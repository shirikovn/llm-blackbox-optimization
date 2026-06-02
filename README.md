## LLM-blackbox-optimization experiment setup
Project based on Hydra (OmegaConf), check how to use it if not familiar.

###  Usage example
```
python run.py function=sphere optimizer=interactive experiment.steps=5
```

See configs dir to more.

> interactive - special mode to use big LLMs via your browser client.

### Transformation usage
Обычный запуск:
```
python run.py function=transformed_rosenbrock
```

Рандомное преобразование:
```
python run.py \
    function=transformed_rosenbrock \
    function.randomize=true
```

Жёсткая деформация:
```
python run.py \
    function=transformed_ackley \
    function.rotation_deg=137 \
    function.scale=[0.15,6.0] \
    function.shift=[12,-9]
```

Полное аффинное преобразование:
```
python run.py function=transformed_rosenbrock
```

#### Warning
Plots and some functions can be buggy.
