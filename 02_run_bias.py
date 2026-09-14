import random
import signal
from types import SimpleNamespace
import settings as cfg
from helpers import readCsv, writeCsv, readCode, algoClass, runAlgorithms, groupedSamples
from detectors import makeBias, seedTool
import numpy as np

sampleFile = cfg.dataDir / f"{cfg.objective}_samples.csv"
biasFile = cfg.dataDir / f"{cfg.objective}_bias_results.csv"

xcol = [f"x{i}" for i in range(cfg.dim)]
allcol = ["id", "name", "run", *xcol]
biascol = ["id", "name", "bias_detected", "bias_class", "rejection_count", "status"]

class Objective:
    def __init__(self, run):
        self.bounds = SimpleNamespace(lb=np.zeros(cfg.dim), ub=np.ones(cfg.dim))
        self.rng = np.random.default_rng(cfg.seed + 1_000_000 + run)
        self.i = 0
        self.valid = True

    def __call__(self, x):
        self.i += 1

        if self.i > cfg.budget: raise RuntimeError("Budget exceeded")

        x = np.asarray(x, dtype=float)
        self.valid &= x.shape == (cfg.dim,) and np.isfinite(x).all()

        return float(self.rng.uniform()) if cfg.objective == "f0" else cfg.g0val

def finalPoint(res):
    if isinstance(res, (tuple, list)):
        for v in res:
            x = np.asarray(v, dtype=float)
            if x.shape == (cfg.dim,):
                return x
    return np.asarray(res, dtype=float)

def timeoutHandler(signum, frame):
    raise TimeoutError("nono time")

def sampleAlgo(row):

    Algo = algoClass(readCode(row), row["name"])
    signal.signal(signal.SIGALRM, timeoutHandler)

    pts = []

    for run in range(cfg.runs):
        signal.setitimer(signal.ITIMER_REAL, cfg.timeout)

        try:
            np.random.seed(cfg.seed + run)
            random.seed(cfg.seed + run)
            algo = Algo(budget=cfg.budget, dim=cfg.dim)
            obj = Objective(run)
            x = finalPoint(algo(obj))

            if obj.requests != cfg.budget or not obj.valid:
                return None

            if x.shape != (cfg.dim,) or not ((x >= 0) & (x <= 1)).all():
                return None

            pts += [
                {
                    "id": row["id"], 
                    "name": row["name"], 
                    "run": run, **dict(zip(xcol, x))
                }]

        finally: signal.setitimer(signal.ITIMER_REAL, 0)
    
    return pts

def runSampling():
    data = readCsv(cfg.dataset)
    pts = []

    for row, res in runAlgorithms(data, sampleAlgo, cfg.runs * cfg.timeout + 30):
        if res is not None:
            pts.extend(res)

    writeCsv(sampleFile, pts, allcol)
    print(f"Saved {len(pts) // cfg.runs} eligible algorithms to {sampleFile}")

def runBias():
    idd = groupedSamples(sampleFile)
    biasTool = makeBias()
    res = []

    for row in readCsv(cfg.dataset):

        if row["id"] not in idd:
            continue

        seedTool(row["id"])
        rej, ddd = biasTool.predict(
            idd[row["id"]], 
            corr_method="fdr_by", # defaul uses the other method, paper says use this one
            alpha=0.01,
            show_figure=False, 
            print_type=False
        )

        bias_detected = "none" if ddd == "none" else str(ddd["Class"])
        res += [
            {
                "id": row["id"], 
                "name": row["name"], 
                "bias_detected": bias_detected != "none",
                "bias_class": bias_detected, 
                "rejection_count": int(rej.to_numpy().sum()), 
                "status": "ok"
            }]
        print(f"BIAS {len(res)}/{len(idd)}")

    writeCsv(biasFile, res, biascol)

def main():
    runSampling()
    runBias()

if __name__ == "__main__":
    main()
