#!/usr/bin/env python3
"""Read-only diagnostics for the EP24 MongoDB Step 6 eligibility gate."""
import argparse
import os
from collections import Counter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=os.environ.get("LACLAUGPT_EP24_ENV_FILE"))
    parser.add_argument("--country", default="finland")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        if not Path(args.env_file).is_file():
            parser.error("Environment file does not exist")
        load_dotenv(args.env_file, override=False)
    from pymongo import MongoClient
    uri = os.environ.get("LACLAUGPT_MONGO_URI") or os.environ.get("MONGODB_URI")
    if not uri:
        parser.error("LACLAUGPT_MONGO_URI is not configured")
    db = os.environ.get("LACLAUGPT_MONGO_DATABASE", "spectacleScraper")
    dataset = os.environ.get("LACLAUGPT_DATASET", "ep2024_reprocess")
    collection = f"laclaugpt_{dataset}_{args.country}_dataframe"
    with MongoClient(uri, serverSelectionTimeoutMS=10000) as client:
        c = client[db][collection]
        print(f"Database: {db}; collection: {collection}; total: {c.count_documents({})}")
        for n in range(1,7):
            key = f"_pipeline.step_{n:02d}.status"
            results = c.aggregate([{"$group":{"_id":f"${key}","n":{"$sum":1}}},{"$sort":{"n":-1}}])
            print(f"Step {n}: "+str({str(r["_id"]):r["n"] for r in results}))
        results = c.aggregate([{"$group":{"_id":"$processing_status","n":{"$sum":1}}},{"$sort":{"n":-1}}])
        print("Processing status: "+str({str(r["_id"]):r["n"] for r in results}))
        step5 = {"_pipeline.step_05.status":"complete"}
        ok = {"processing_status":{"$in":["OK","ok"]}}
        print("Step 5 complete:",c.count_documents(step5))
        print("Quality OK:",c.count_documents(ok))
        print("Step 5 complete and quality OK:",c.count_documents({**step5,**ok}))
        eligible = {**step5,**ok,"$or":[{"_pipeline.step_06.status":{"$exists":False}},{"_pipeline.step_06.status":{"$in":["pending","retry"]}}]}
        print("Step 6 eligible for normal run:",c.count_documents(eligible))
if __name__ == "__main__":
    main()
