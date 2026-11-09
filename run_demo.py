"""Run only the clearly identified synthetic CPU example."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time

def fs_path(path):
    p = Path(path).absolute()  # Preserve useful short junction paths on Windows.
    text = str(p)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        if text.startswith("\\\\"):
            text = "\\\\?\\UNC\\" + text[2:]
        else:
            text = "\\\\?\\" + text
    return Path(text)

ROOT = fs_path(Path(__file__).parent)
sys.path.insert(0,str(ROOT))
from src.core import run
from data.generate import generate

def main():
    parser = argparse.ArgumentParser(description="SYNTHETIC MECHANISM DEMO - not paper results")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"synthetic_run")
    args = parser.parse_args()
    out = fs_path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    cfg = json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    if cfg.get("synthetic_only") is not True:
        raise ValueError("This educational delivery may not be labelled as real-data evidence")
    sample = ROOT/"data"/"SYNTHETIC_sample.npz"
    if not sample.exists():
        generate(sample,cfg["seed"])
    print("SYNTHETIC DATA ONLY - outputs are not publication evidence.",flush=True)
    start=time.perf_counter()
    metrics=run(sample,out,cfg)
    metrics.update({"algorithm":"SPFD mechanism demo","synthetic_smoke_test":True,
                    "publication_evidence":False,"seed":cfg["seed"],
                    "data_sha256":hashlib.sha256(sample.read_bytes()).hexdigest(),
                    "elapsed_seconds":time.perf_counter()-start})
    (out/"synthetic_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2,
                                                       allow_nan=False),encoding="utf-8")
    (out/"config_used.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed synthetic mechanism demo",
                      "publication_evidence":False,"output":str(out),
                      "elapsed_seconds":metrics["elapsed_seconds"]},ensure_ascii=False))
if __name__=="__main__":
    main()
