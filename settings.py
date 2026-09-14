from pathlib import Path

root = Path(__file__).resolve().parent
srcRoot = root.parent
logsDir = srcRoot / "algorithms_LLaMEA-master"
dataDir = root / "data"
dataset = dataDir / "swarm_algorithms.csv"
biasDir = srcRoot / "BIAS"
bladeDir = srcRoot / "BLADE"

objective = "f0"  # f0 g0
runs = 30 # biastoolbox recomands 100 but 30 works fine (i saw that to late)
dim = 10
budget = 100 * dim
g0val = 67.0 # go value
seed = 20260901
timeout = 60
