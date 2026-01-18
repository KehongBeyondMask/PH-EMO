from __future__ import annotations
import json
from typing import List, Dict, Any
from tqdm import tqdm

from .metrics import compute_rrs, compute_acc_f1
from .pipeline import PHEMOPipeline

def safe_parse_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"_parse_error": True, "raw": s}

def run_eval(pipeline: PHEMOPipeline, dataset_iter, out_jsonl: str):
    y_true, y_pred = [], []
    r_ij = []

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for sample in tqdm(dataset_iter):
            res = pipeline.run_one(sample)

            # parse mapping
            mp = safe_parse_json(res["mapping_raw"])
            pred = mp.get("label", None)
            if pred is None:
                pred = "UNKNOWN"

            # parse judge -> r_i,1..3
            r_i = []
            for raw in res["judge_raw"]:
                jj = safe_parse_json(raw)
                r = jj.get("consistent", 0)
                r_i.append(int(r) if str(r).isdigit() else 0)

            y_true.append(res["y_true"])
            y_pred.append(pred)
            r_ij.append(r_i)

            f.write(json.dumps({**res, "y_pred": pred, "r_i": r_i}, ensure_ascii=False) + "\n")

    cls = compute_acc_f1(y_true, y_pred)
    rrs = compute_rrs(r_ij)

    return {"acc": cls["acc"], "f1": cls["f1"], "rrs": rrs}
