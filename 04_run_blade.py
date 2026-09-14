import hashlib, json, math
import random, sys
import signal
import threading

import settings as cfg
from helpers import writeCsv, readCode, algoClass, runAlgorithms
import numpy as np
import pandas as pd

sys.path.insert(0, str(cfg.bladeDir))
import ioh
from ioh import logger as iohLogger
from iohblade.utils import OverBudgetException, aoc_logger, correct_aoc

fids = [1, 3, 5, 8, 15, 20, 21, 23]
iids = [1, 2]
dims = [2, 5]
cases = [(fid, iid, dim) for dim in dims for fid in fids for iid in iids]
budgetMult = 50
suites = ["bbob", "sbox_cost"]
suite = "bbob"
cellTime = 10
suiteLabels = {"bbob": "BBOB", "sbox_cost": "SBOX-COST"}
summaryCols = ["id", "name", "case_type", "suite", "blade_score", "status"]
instCols = ["id", "name", "case_type", "suite", "fid", "iid", "dim", "budget", "auc", "status"]

def pickedRows():
    f0 = pd.read_csv(cfg.dataDir / "f0_bias_results.csv", keep_default_na=False)
    g0 = pd.read_csv(cfg.dataDir / "g0_bias_results.csv", keep_default_na=False)

    pairs = f0[f0.status.eq("ok")][["id", "bias_detected"]
    ].merge(
        g0[g0.status.eq("ok")][["id", "bias_detected"]], 
        on="id", suffixes=("_f0", "_g0")
    )

    data = pd.read_csv(cfg.dataset, keep_default_na=False).merge(pairs, on="id")

    def caseType(row):
        f = str(row["bias_detected_f0"]).lower() == "true"
        g = str(row["bias_detected_g0"]).lower() == "true"
        if f and g: 
            return "both_biased"
        if f: 
            return "f0_only_biased"
        if g: 
            return "g0_only_biased"

        return "neither_biased"

    cols = [*data.columns, "case_type"]
    data = data.sort_values("id").to_dict("records")
    for row in data:
        row["case_type"] = caseType(row)
    writeCsv(cfg.dataDir / "paired_algorithms.csv", data, cols)

    return data

def cellSeed(aid, fid, iid, dim):
    txt = json.dumps(
        {
            "base": cfg.seed, 
            "algorithm_id": aid, 
            "fid": fid, 
            "iid": iid, 
            "dim": dim
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False) # : csv

    return int.from_bytes(hashlib.sha256(txt.encode()).digest()[:4], "big")

class BudgetedObjective:
    def __init__(self, prob, budget, dim):
        self.problem = prob
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.lock = threading.RLock()

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def __call__(self, x):
        with self.lock:
            arr = np.asarray(x, dtype=float)
            single = arr.shape == (self.dim,)
            pts = [arr] if single else arr
            left = self.budget - self.evaluations
            if len(pts) > left:
                raise OverBudgetException
            res = []
            for pt in pts:
                res.append(self.problem(pt))
                self.evaluations += 1
            return res[0] if single else np.asarray(res)

class CellTimedOut(BaseException):
    pass

def timeoutHandler(signum, frame):
    raise CellTimedOut("nono time")

def emptyCell(row, fid, iid, dim):
    return {
        "id": row["id"], 
        "name": row["name"], 
        "case_type": row["case_type"], 
        "suite": suiteLabels[suite],
        "fid": fid, 
        "iid": iid, 
        "dim": dim, 
        "budget": budgetMult * dim, 
        "auc": "", 
        "status": "error"
    }

def runCell(row, fid, iid, dim):
    out = emptyCell(row, fid, iid, dim)
    budget = out["budget"]
    signal.signal(signal.SIGALRM, timeoutHandler)
    signal.setitimer(signal.ITIMER_REAL, cellTime)

    try:
        log = aoc_logger(budget, upper=1e2, triggers=[iohLogger.trigger.ALWAYS])
        seed = cellSeed(row["id"], fid, iid, dim)

        np.random.seed(seed)
        random.seed(seed)

        kind = ioh.ProblemClass.BBOB if suite == "bbob" else ioh.ProblemClass.SBOX

        prob = ioh.get_problem(fid, instance=iid, dimension=dim, problem_class=kind)
        prob.attach_logger(log)
        obj = BudgetedObjective(prob, budget, dim)

        algo = algoClass(
            row["code"], 
            row["name"], 
            {
                "np": np, 
                "ioh": ioh, 
                "math": math
            })

        try:
            algo(budget=budget, dim=dim)(obj)

        except OverBudgetException: pass

        auc = float(correct_aoc(prob, log, budget))

        if 0 <= auc <= 1 and prob.state.evaluations == obj.evaluations:
            out.update(auc=auc, status="ok")

    except (Exception, CellTimedOut): pass

    finally: signal.setitimer(signal.ITIMER_REAL, 0)

    return out

def evalOne(row):
    row = {**row, "code": readCode(row)}
    return [
        runCell(row, fid, iid, dim) 
        for fid, iid, dim in cases
    ]

def main():
    global suite

    data = pickedRows()

    for suite in suites:
        summary = []
        inst = []

        print(f"BLADE {suiteLabels[suite]}: {len(data)} paired algorithms")
        
        for i, (row, res) in enumerate(runAlgorithms(data, evalOne, len(cases) * cellTime + 10), 1):

            cells = res if res is not None else [emptyCell(row, fid, iid, dim) for fid, iid, dim in cases]

            ok = [c for c in cells if c["status"] == "ok"]
            done = len(ok) == len(cases)

            summary += [
                {
                    "id": row["id"], 
                    "name": row["name"], 
                    "case_type": row["case_type"],
                    "suite": suiteLabels[suite], 
                    "blade_score": np.mean([c["auc"] for c in ok]) if done else "",
                    "status": "ok" if done else "partial" if ok else "error"
                }]

            inst.extend(cells)

            if i % 25 == 0 or i == len(data):
                print(f"BLADE {i}/{len(data)}")

        writeCsv(cfg.dataDir / f"blade_{suite}_summary.csv", summary, summaryCols)
        writeCsv(cfg.dataDir / f"blade_{suite}_per_instance.csv", inst, instCols)

if __name__ == "__main__":
    main()
