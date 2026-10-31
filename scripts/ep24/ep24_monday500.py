#!/usr/bin/env python3
"""CSV-first Monday 500 recovery. Original EP24 Mongo collections are never modified."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Direct execution from scripts/ep24 must resolve repository-root modules.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

COUNTRIES = ("finland", "poland", "portugal")
NAMES = {1:"preprocess", 2:"frame", 3:"video", 4:"summary", 5:"postprocess", 6:"discourse_analysis", 7:"discourse_network_analysis"}
STATUS = {1:"preprocess_status",2:"frame_analysis_status",3:"vllm_video_status",4:"summary_quality_status",5:"processing_status",6:"laclau_status",7:"dna_status"}
RESULT = {1:"asr_transcript",2:"frame_analysis_1",3:"vllm_video_analysis",4:"summary_analysis",5:"postprocess_summary_md",6:"formula_of_populism_analysis",7:"dna_analysis_markdown"}
GOOD = {"ok","cached","complete","no_evidence"}
EMPTY = {"","nan","none","null"}

def present(value):
    return value is not None and str(value).strip().lower() not in EMPTY

def valid(record, step):
    state = str(record.get(STATUS[step], "")).strip().lower()
    output = present(record.get(RESULT[step], ""))
    if step == 1:
        return state in GOOD and (output or present(record.get("frame_file", "")))
    if step in (2,3,6):
        return state in GOOD and output
    if step in (4,5):
        return output and state not in {"error","failed","delete","reprocess"}
    return state in GOOD and (output or present(record.get("dna_statements_json", "")))

def build(outputs, country, count):
    candidates = {}
    source_counts = {}
    for step,name in NAMES.items():
        path = outputs/country/f"step_{step:02d}_{name}.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        if "video_id" not in df.columns:
            raise ValueError(f"{path}: missing video_id")
        source_counts[str(step)] = len(df)
        for row in df.to_dict("records"):
            vid = str(row.get("video_id", "")).strip()
            if not vid:
                continue
            existing = candidates.setdefault(vid, {})
            old = existing.get(step)
            score = lambda x: (int(valid(x,step)), sum(present(v) for v in x.values()))
            if old is None or score(row) > score(old):
                existing[step] = row
    selected = []
    for vid,stage_rows in candidates.items():
        through = 0
        for step in range(1,8):
            if step in stage_rows and valid(stage_rows[step],step):
                through = step
            else:
                break
        if not through:
            continue
        merged = {}
        for step in range(1,through+1):
            for key,value in stage_rows[step].items():
                if present(value) and not key.startswith("_pipeline"):
                    merged[key] = value
        if not present(merged.get("allas_filename","")):
            continue
        merged["video_id"] = vid
        merged["country"] = country
        merged["_storage_id"] = "monday500:"+country+":"+hashlib.sha256(vid.encode()).hexdigest()[:24]
        merged["_monday500_reused_through"] = through
        merged["_pipeline"] = {f"step_{step:02d}":{"status":"complete"} for step in range(1,through+1)}
        selected.append((through, vid, merged))
    selected.sort(key=lambda row:(-row[0],row[1]))
    if len(selected) < count:
        raise RuntimeError(f"{country}: only {len(selected)} usable unique video_ids; need {count}")
    chosen = selected[:count]
    return [x[2] for x in chosen], {
        "selected": count, "available":len(selected),
        "source_rows":source_counts,
        "reused_stages":{str(i):sum(x[0]>=i for x in chosen) for i in range(1,8)}
    }

def storage(country,dataset):
    from roihu_storage import StorageConfig, MongoStorage
    os.environ["LACLAUGPT_COUNTRY"] = country
    os.environ["LACLAUGPT_DATASET"] = dataset
    config = StorageConfig.from_env()
    if not config.mongo_enabled or not config.mongo_uri:
        raise RuntimeError("Mongo is required. Source scripts/roihu/roihu_job_common.sh")
    return MongoStorage(config)

def prepare(args):
    if not args.dataset.startswith("ep24_monday500_"):
        raise ValueError("Use isolated ep24_monday500_* dataset only")
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest = {"dataset":args.dataset,"countries":{},"source_collections_untouched":True}
    prepared = {}
    for country in COUNTRIES:
        rows,report=build(Path(args.outputs),country,args.count)
        store=storage(country,args.dataset)
        try:
            target=store.db[store.collection_name("dataframe")]
            if target.count_documents({}):
                raise RuntimeError(f"Refusing nonempty target {target.full_name}")
            manifest["countries"][country]={**report,"collection":target.full_name}
        finally:
            store.close()
        prepared[country]=rows
    for country,rows in prepared.items():
        pd.DataFrame([{k:v for k,v in row.items() if k != "_pipeline"} for row in rows]).to_csv(out/f"{country}_selected.csv",index=False)
        (out/f"{country}_ids.txt").write_text("".join(row["_storage_id"]+"\n" for row in rows))
        if args.apply:
            from pymongo import InsertOne
            store=storage(country,args.dataset)
            try:
                coll=store.db[store.collection_name("dataframe")]
                if coll.count_documents({}):
                    raise RuntimeError(f"Target was populated concurrently: {coll.full_name}")
                coll.bulk_write([InsertOne(row) for row in rows],ordered=False)
                if coll.count_documents({}) != args.count:
                    raise RuntimeError(f"{country}: unexpected inserted row count")
            finally:
                store.close()
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    print("IMPORTED" if args.apply else "DRY RUN ONLY")

def export(args):
    out=Path(args.run_dir).resolve()
    manifest=json.loads((out/"manifest.json").read_text())
    frames=[]
    for country in COUNTRIES:
        ids=(out/f"{country}_ids.txt").read_text().splitlines()
        store=storage(country,manifest["dataset"])
        try:
            docs=list(store.db[store.collection_name("dataframe")].find({"_storage_id":{"$in":ids}},{"_id":0}))
        finally:
            store.close()
        if len(docs)!=args.count or len({d["_storage_id"] for d in docs})!=args.count:
            raise RuntimeError(f"{country}: identity mismatch in export: {len(docs)}")
        df=pd.json_normalize(docs,sep=".")
        df["dashboard_country"]=country
        df["dashboard_stage7_complete"]=df.get("_pipeline.step_07.status",pd.Series("",index=df.index)).eq("complete")
        df["dashboard_step7_has_text"]=df.get("dna_analysis_markdown",pd.Series("",index=df.index)).map(present)
        df.to_csv(out/f"{country}_final.csv",index=False)
        frames.append(df)
        print(country, int(df["dashboard_stage7_complete"].sum()),"/",args.count,"Step 7 complete")
    final=pd.concat(frames,ignore_index=True)
    if len(final) != 3*args.count:
        raise RuntimeError("Export row count mismatch")
    final.to_csv(out/"ep24_research_dashboard_1500.csv",index=False)
    print("FINAL",out/"ep24_research_dashboard_1500.csv")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=["prepare","export"])
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--outputs",default="/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs")
    ap.add_argument("--dataset",default="ep24_monday500_20261031")
    ap.add_argument("--count",type=int,default=500)
    ap.add_argument("--apply",action="store_true")
    args=ap.parse_args()
    if args.action=="prepare":prepare(args)
    else:export(args)

if __name__=="__main__":
    main()
