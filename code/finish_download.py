"""finishes the nlstseg ct download in small batches, safe to rerun"""
import glob, os, time
import pandas as pd
from idc_index import IDCClient

ROOT = r"C:\MetaSeg_data"
TMPL = "%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID"
BATCH, MAX_PASSES = 50, 8

c = IDCClient()
m = pd.read_csv(os.path.join(ROOT, "nlstseg_manifest.csv")).dropna(subset=["ct_series"]).drop_duplicates("ct_series")

def incomplete():
    out = []
    for _, r in m.iterrows():
        d = glob.glob(os.path.join(ROOT, "raw", "nlst", str(r.PatientID), "*", f"CT_{r.ct_series}"))
        n = len(os.listdir(d[0])) if d else 0
        if n < r.instanceCount:
            out.append(r.ct_series)
    return out

for p in range(1, MAX_PASSES + 1):
    todo = incomplete()
    print(f"pass {p}: {len(todo)} of {len(m)} CT series incomplete", flush=True)
    if not todo:
        break
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            c.download_from_selection(downloadDir=os.path.join(ROOT, "raw", "nlst"), seriesInstanceUID=batch,
                                      dirTemplate=TMPL, use_s5cmd_sync=True, show_progress_bar=False, quiet=True,
                                      source_bucket_location="gcs")
        except Exception as e:
            print(f"  batch {i//BATCH} error: {e}", flush=True)
            time.sleep(20)
        print(f"  pass {p}: batch {i//BATCH + 1}/{(len(todo)+BATCH-1)//BATCH} done", flush=True)
left = incomplete()
print(f"FINISHED: {len(m) - len(left)} complete, {len(left)} incomplete", flush=True)
