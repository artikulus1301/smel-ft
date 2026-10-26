# SMEL FT — Sparse MoE LoRA Fine-Tuning

[English version](README.md)

Дообучение MoE-модели на 35B параметров на **одной потребительской видеокарте на 8 ГБ**. LoRA ставится только на
тех экспертов, которых роутер действительно выбирает на ваших данных, а замороженный низ сети считается один раз
и кэшируется.

Эталонный прогон (подробности — [docs/results.md](docs/results.md)): **Qwen3.6-35B-A3B** на RTX 4070 Laptop (8 ГБ),
24 ГБ RAM, Windows 11. 3 эпохи заняли 18–27 минут при пике видеопамяти меньше 5 ГБ. Перплексия на отложенных
доменных данных упала с 17,88 до 7,97–8,34, на общем тексте ухудшения нет. На выходе — GGUF того же размера и
скорости, что исходный квант.

## Зачем

- В transformers 5.x и в чекпойнте эксперты Qwen3.5/3.6-MoE хранятся **слитыми 3D-тензорами**
  (`gate_up_proj [256, 1024, 2048]`, `down_proj [256, 2048, 512]` на слой). PEFT не может повесить LoRA на
  отдельного эксперта, а bitsandbytes такие тензоры не квантует.
- 4-битный QLoRA для этого семейства не рекомендуется, и ему всё равно нужна вся модель на 35B в видеопамяти.
- В MoE-слое домен в основном использует небольшую часть экспертов, поэтому обучать только их дёшево.

## Как это работает

```
probe    один проход всех 40 слоёв, слой за слоем, по доменному и общему тексту; статистика роутера
select   D_l = KL(p_domain || p_general) по слоям -> точка разреза L; в каждом слое минимальный набор экспертов,
         покрывающий p = 50% доменной gate-массы (не больше k_max = 12); оценка VRAM / RAM / времени
cache    скрытые состояния на входе слоя L для всего датасета (бесплатный побочный продукт probe)
pack     блок L..N-1: замороженные эксперты -> int4 (группа 64), выбранные эксперты и плотные части -> bf16,
         lm_head -> int8
train    LoRA на выбранных экспертах + на shared-эксперте каждого слоя блока, роутер заморожен;
         порядок «слой за слоем»: каждый слой загружается в GPU дважды за шаг оптимизатора
         (forward по всем микробатчам, затем backward с recompute), сколько бы ни было микробатчей
merge    точное вливание LoRA в bf16-веса (неизменённые шарды — жёсткие ссылки)
export   GGUF: копия существующего кванта базовой модели, в которой заново квантованы только обученные эксперты,
         с тем же типом ggml и той же imatrix; остальные байты не меняются
```

Подробнее о реализации:
- **Свой MoE-forward.** Замороженные эксперты считаются в своей autograd-функции: в backward веса деквантуются
  заново, группами, и целый слой экспертов для autograd не хранится. Матричные умножения по экспертам — через
  `torch._grouped_mm`, деквантизация int4 — Triton-ядро. Сборка выходов экспертов детерминированная (без атомарных
  операций).
- **Гибридные слои.** Gated DeltaNet считается ядрами flash-linear-attention, полное внимание — через SDPA.
  Упаковка последовательностей работает через `cu_seqlens`, causal conv с учётом границ и блочно-диагональную
  маску; всё проверено тестами.
- **Лосс.** Чанкнутый cross-entropy по словарю из 248k токенов, без материализации логитов. Лосс считается только
  по ответам ассистента; пустой блок `<think></think>` шаблона исключён.
- **Подгрузка весов.** Двойной буфер с предзагрузкой в отдельном CUDA-стриме, из pinned RAM или через mmap.
- **Продолжение прогонов.** Длинные этапы продолжаются с `--resume`, каждый этап дописывает краткий итог
  в `logs/progress.md`.

**Поддерживаемая модель:** архитектура `qwen3_5_moe` из transformers (Qwen3.5/3.6 MoE). Проверено на
`Qwen/Qwen3.6-35B-A3B`. Размеры берутся из `config.json`, так что другие модели семейства могут работать, но не
проверялись.

## Требования

- Видеокарта NVIDIA с ~6 ГБ свободной видеопамяти (Ampere или новее: bf16 и Triton). Проверено на RTX 4070 Laptop,
  8 ГБ.
- Проверено на 24 ГБ RAM (бюджет RAM задаётся в конфиге, меньше не пробовали); ~110 ГБ на диске под bf16-веса
  (72 ГБ), кэш и упакованный блок.
- Python 3.11+ (проверено на 3.14), torch 2.11+ с CUDA, transformers 5.17+, Triton (`triton-windows` на Windows),
  flash-linear-attention. Работает на Linux и на обычной Windows, WSL не нужен.
- Для экспорта GGUF и оценки: сборка [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`,
  `ggml-base`), существующий GGUF базовой модели и его importance matrix.

```bash
pip install -e ".[gguf,data,dev]"      # сначала поставьте torch с CUDA с pytorch.org
python -m pytest                       # ~30 с; нужна CUDA; скачиваются только файлы токенайзера Qwen
```

## Быстрый старт

```bash
# 0. данные: ваш chat jsonl -> data/domain.jsonl, по диалогу в строке:
#    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
python scripts/download.py                    # bf16 safetensors -> data/Qwen3.6-35B-A3B (72 ГБ, с докачкой)
python scripts/make_general.py                # общий текст -> data/general.jsonl (русская Википедия + чаты);
                                              # или положите туда свой обычный текст

# по желанию: прогон всего конвейера на маленькой случайной модели того же класса (минуты):
python scripts/make_tiny_e2e.py --general data/general.jsonl
#   ... и затем все этапы ниже с --config configs/tiny_e2e.yaml

python scripts/probe.py                       # профиль роутера, ~6 мин; reports/probe.md
python scripts/select_experts.py              # L + эксперты + оценка памяти/времени; configs/selection.yaml
#   задать точку разреза вручную: --set select.L_override=16
python scripts/cache.py                       # вход слоя L (жёсткая ссылка на кэш из probe, если он есть)
python scripts/pack.py                        # упакованный блок + сверка с bf16 на 50 диалогах
python scripts/train.py --max-steps 5         # пробный запуск, затем:
python scripts/train.py --resume --set train.lr=2e-4
python scripts/eval.py --adapter artifacts/adapter_L16
python scripts/merge.py --adapter artifacts/adapter_L16 --out work/merged_L16 --verify
SMEL_LLAMA_CPP_DIR=/path/to/llama.cpp/build/bin \
python scripts/export_gguf.py --adapter artifacts/adapter_L16 --base-gguf base.gguf --imatrix imatrix.gguf
```

Все настройки — в [configs/default.yaml](configs/default.yaml); любую можно переопределить на один запуск,
например `--set train.lr=2e-4 train.epochs=4`.

## Этапы и результаты

| этап | скрипт | что получается |
|---|---|---|
| скачивание | `scripts/download.py` | `paths.model_dir` |
| общий корпус | `scripts/make_general.py` | `paths.general_data` |
| probe | `scripts/probe.py` | `artifacts/routing_stats.pt`, `reports/probe.md`, `reports/probe_divergence.png`, кэш входов кандидатов L |
| select | `scripts/select_experts.py` | `configs/selection.yaml`, `reports/select.md` |
| cache | `scripts/cache.py [--L N]` | `work/cache_L{L}/` |
| pack | `scripts/pack.py [--no-write]` | `work/packed_L{L}/`, `reports/pack.md` |
| train | `scripts/train.py [--max-steps N] [--resume] [--tag T] [--selection F]` | `artifacts/adapter_{T}/`, `logs/train_{T}.jsonl` |
| eval | `scripts/eval.py --adapter A [--packed]` | `reports/eval.md`, `A/eval_ppl.json` |
| merge | `scripts/merge.py --adapter A --out DIR [--verify]` | слитые safetensors |
| GGUF | `scripts/export_gguf.py --adapter A --base-gguf G [--imatrix I] [--reuse]` | пропатченный GGUF |
| абляции | `scripts/make_selection.py --kind random\|shared` | контрольные наборы экспертов |

Инструменты: `scripts/gguf_repro_check.py` проверяет, что наш тракт квантования воспроизводит байты базового GGUF на
неизменённом слое. `scripts/gguf_quant_error.py` и `scripts/bench/` — замеры железа и ядер.
[examples/persona_eval/](examples/persona_eval/) — генерация ответов и слепой судья из эталонного эксперимента.

## Что показал эталонный эксперимент

Варианты отличаются только тем, какие роутируемые эксперты получают LoRA; shared-эксперты обучаются во всех
(перплексия до обучения — 17,88):

| вариант | перплексия домена | судья против базовой модели |
|---|---|---|
| только shared-эксперты | 9,03 | 19 : 20 |
| + 168 случайных экспертов (L=20) | 8,88 | 23 : 15 |
| + 168 экспертов, выбранных по роутеру (L=20) | 8,34 | **27 : 10** |
| + 213 экспертов, выбранных по роутеру (L=16) | **7,97** | 24 : 19 |

Основную часть эффекта дают shared-эксперты. Эксперты, выбранные по роутеру, стабильно добавляют ещё 8–12% по
перплексии (случайные — 2%) и большую часть «голоса» персонажа. Точка разреза ниже помогает, как и подсказывает probe.
Это честные цифры небольшого эксперимента на одном домене; оговорки — в [docs/results.md](docs/results.md).

## Ограничения

- Одна архитектура (`qwen3_5_moe`). Для других MoE-семейств нужен свой код слоя в `smel/layer_runner.py`.
- Роутер заморожен, dropout в LoRA нет.
- Если блок больше файлового кэша ОС, каждый шаг заново читает упакованные веса с диска; нужен быстрый NVMe.
- Экспорт GGUF патчит существующий квант, а не создаёт новый с нуля.

## Структура проекта

```
smel/       библиотека: model_io (послойная загрузка через mmap), moe, quant (int4 + Triton-деквант), lora,
            layer_runner, stream (предзагрузка), loss (чанкнутый CE), data (chat template + маски + упаковка),
            bottom (послойный прогон замороженных слоёв), pack (упаковка + точное слияние), gguf_patch,
            pipeline (конфиг, хранилища)
scripts/    этапы конвейера, scripts/bench/ — бенчмарки
tests/      тесты эквивалентности с transformers на маленькой случайной модели того же класса
examples/   persona_eval: генерация, автометрики и слепой попарный судья через llama-server
docs/       результаты эталонного эксперимента
```

## Лицензия

[Apache-2.0](LICENSE). Базовая модель, её веса и токенайзер не входят в репозиторий и распространяются под своей
лицензией. См. [NOTICE](NOTICE).
