import settings as cfg
from helpers import readCsv, writeCsv, groupedSamples
from detectors import makeBias, seedTool
import numpy as np

sampleFile = cfg.dataDir / f"{cfg.objective}_samples.csv"
biasFile = cfg.dataDir / f"{cfg.objective}_bias_results.csv"
deepFile = cfg.dataDir / f"{cfg.objective}_deepbias_results.csv"
probcol = ["prob_unif", "prob_centre", "prob_disc", "prob_bounds", "prob_gaps_clusters"]
deepcool = ["id", "name", "deepbias_detected", "deep_label", *probcol, "status"]

def main():
    ptsById = groupedSamples(sampleFile)
    data = [r for r in readCsv(biasFile) if r["bias_detected"].lower() == "true"]
    biasTool = makeBias()
    res = []

    for row in data:
        seedTool(row["id"])
        deepbias_detected, pred = biasTool.predict_deep(ptsById[row["id"]])
        bb = np.asarray(pred).reshape(-1, 5).mean(axis=0)

        res += [
            {
                "id": row["id"], 
                "name": row["name"], 
                "deepbias_detected": str(deepbias_detected) != "unif",
                "deep_label": str(deepbias_detected), **dict(zip(probcol, bb)), 
                "status": "ok"
            }]
        print(f"Deep-BIAS {len(res)}/{len(data)}")

    writeCsv(deepFile, res, deepcool)

if __name__ == "__main__":
    main()