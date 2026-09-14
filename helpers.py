import csv
import json
import multiprocessing as mp

import settings as cfg

def writeCsv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

def readCsv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def readCode(row):
    path = cfg.srcRoot / row["log_path"]
    entry = json.loads(path.read_text(encoding="utf-8").splitlines()[int(row["line_no"]) - 1])
    code = entry.get("solution") or entry.get("code") or ""
    
    return code if code.strip() else (cfg.srcRoot / row["code_path"]).read_text(encoding="utf-8")

def algoClass(code, name, extra=None):
    ns = {"__name__": "__generated__", **(extra or {})}

    exec(code, ns) # hope the code is safe

    if name in ns and isinstance(ns[name], type):
        return ns[name]

    return [
        v for v in ns.values() 
            if isinstance(v, type) and v.__module__ == "__generated__"
        ][-1]

def runChild(func, row, conn):
    try:
        conn.send(func(row))
    except:
        conn.send(None)
    finally:
        conn.close()

def runAlgorithms(rows, func, timeout):
    ctx = mp.get_context("fork")

    for row in rows:

        recv, send = ctx.Pipe(duplex=False)
        p = ctx.Process(target=runChild, args=(func, row, send))
        p.start()
        send.close()
        res = None

        try:

            if recv.poll(timeout):
                res = recv.recv()

        except EOFError: pass

        finally:
            p.join(timeout=0.1)

            if p.is_alive():
                p.kill()
                p.join()
                
            recv.close()
        yield row, res


def groupedSamples(path):
    import pandas as pd
    df = pd.read_csv(path)
    cols = [f"x{i}" for i in range(cfg.dim)]

    return {
        aid: part.sort_values("run")[cols].to_numpy(float) 
        for aid, part 
        in df.groupby("id")
    }
