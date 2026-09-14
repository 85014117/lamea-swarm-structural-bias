# LLaMEA swarm structural bias

## Setup

Use Python 3.10 on Linux/macos. Uses os.fork() -> not avaible on windows.

First install & setup [BIAS](https://github.com/nikivanstein/BIAS), [BLADE](https://github.com/XAI-liacs/BLADE), [LLaMEA dataset](https://github.com/haoran-ian/algorithms_LLaMEA)

Folder layout is:

```text
workspace/
├── algorithms_LLaMEA-master/
├── BIAS/
├── BLADE/
└── lamea-swarm-structural-bias/
    ├── settings.py
    ├── 01_sort_dataset.py
    ├── ...
    └── data/
```



Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run


First rebuild the dataset tables:

```bash
python 01_sort_dataset.py
```

Set `objective = "f0"` or `"g0"` in `settings.py`, then run:

```bash
python 02_run_bias.py
python 03_run_deep_bias.py
```

Finaly BLADE:

```bash
python 04_run_blade.py
```


## Outputs

| Files in `data/` | Contents |
| --- | --- |
| `all_algorithms.csv`, `swarm_algorithms.csv` | Dataset metadata and selected swarm candidates |
| `f0_samples.csv`, `g0_samples.csv` | Final solution vectors, one row per run |
| `f0_bias_results.csv`, `g0_bias_results.csv` | BIAS detections, classes and rejection counts |
| `f0_deepbias_results.csv`, `g0_deepbias_results.csv` | Deep-BIAS labels and probabilities averaged across dimensions |
| `paired_algorithms.csv` | Algorithms with valid results under both objectives and their overlap group |
| `blade_*_per_instance.csv` | Individual BLADE scores and execution status |
| `blade_*_summary.csv` | Mean BLADE score when all 32 cases in that suite succeed |
