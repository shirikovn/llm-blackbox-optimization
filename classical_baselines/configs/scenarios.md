# Готовые сценарии запуска для Блока 2

Это не конфиг-файлы (Hydra/JSON), а **шпаргалки** — копируй команды в терминал.

## Сценарий A. Минимум — только классика на 2D (быстро, ~1 мин)

```bash
python runner.py tune-lr        --dims 2 --J 10 --K 50
python runner.py run-classical  --dims 2 --J 10 --K 50
python runner.py compare
```

## Сценарий B. Полный прогон классики (2D и 5D)

```bash
python runner.py tune-lr        --dims 2 5 --J 10 --K 50
python runner.py run-classical  --dims 2 5 --J 10 --K 50
python runner.py compare
```

## Сценарий C. Прогон конкретного LLM на одной задаче

```bash
# Например, Claude на Розенброке, n=2, старт #3
python runner.py run-llm \
    --model claude \
    --function rosenbrock \
    --n 2 \
    --start-idx 3 \
    --K 30
```

После прогона все логи и саммари будут в `logs/`, и
`python runner.py compare` подхватит их вместе с классикой.

## Сценарий D. Воспроизвести прошлый LLM-прогон без обращения к модели

```bash
python runner.py run-llm \
    --model claude \
    --function rosenbrock \
    --n 2 \
    --start-idx 3 \
    --K 30 \
    --replay logs/llm_claude_rosenbrock_n2_start3.json
```
