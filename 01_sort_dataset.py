import json
from settings import srcRoot as root, logsDir, dataDir as outDir
from helpers import writeCsv

keys = ("pso", "swarm")

cols = [
    "id",
    "name",
    "description",
    "fitness",
    "generation",
    "source_folder",
    "line_no",
    "log_path",
    "code_path",
]

def readRows():
    data = []
    for logFile in logsDir.glob("*/log.jsonl"):
        logDir = logFile.parent
        for i, line in enumerate(logFile.open(encoding="utf-8"), start=1):
            if not line.strip():
                continue
            data2 = json.loads(line)
            name = data2.get("name", "")
            gen = data2.get("generation", "")
            patt = logDir / "code" / f"try-{gen}-{name}.py"
            data += [
                {
                    "id": data2.get("id", ""),
                    "name": name,
                    "description": data2.get("description", ""),
                    "fitness": data2.get("fitness", ""),
                    "generation": gen,
                    "source_folder": logDir.name,
                    "line_no": i,
                    "log_path": str(logFile.relative_to(root)),
                    "code_path": str(patt.relative_to(root)) if patt.exists() else "",
                }
            ]
    return data

def isSwarm(row):
    txt = f"{row['name']} {row['description']}".lower()
    return any(k in txt for k in keys)

def main():
    data = readRows()
    swarm = [r for r in data if isSwarm(r)]
    writeCsv(outDir / "all_algorithms.csv", data, cols)
    writeCsv(outDir / "swarm_algorithms.csv", swarm, cols)
    print(f"Wrote {len(data)} total algos to {outDir}/all_algorithms.csv")
    print(f"Wrote {len(swarm)} swarm algos to {outDir}/swarm_algorithms.csv")

if __name__ == "__main__":
    main()
