"""downloads the second reconstructions listed in kernel_pairs.csv"""
import glob, os, time
import pandas as pd
from idc_index import IDCClient

ROOT = r"C:\MetaSeg_data"
TMPL = "%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID"
BATCH, MAX_PASSES = 50, 8

c = IDCClient()
pairs = pd.read_csv(os.path.join(ROOT, "kernel_pairs.csv")).drop_duplicates("paired_series")
uids = "','".join(pairs.paired_series)
meta = c.sql_query(f"SELECT SeriesInstanceUID, PatientID, instanceCount FROM index WHERE SeriesInstanceUID IN ('{uids}')")
print(f"{len(meta)} paired series for {meta.PatientID.nunique()} patients", flush=True)


def incomplete():
    out = []
    for _, r in meta.iterrows():
        d = glob.glob(os.path.join(ROOT, "raw", "nlst", str(r.PatientID), "*", f"CT_{r.SeriesInstanceUID}"))
        if (len(os.listdir(d[0])) if d else 0) < r.instanceCount:
            out.append(r.SeriesInstanceUID)
    return out


for p in range(1, MAX_PASSES + 1):
    todo = incomplete()
    print(f"pass {p}: {len(todo)} of {len(meta)} paired series incomplete", flush=True)
    if not todo:
        break
    for i in range(0, len(todo), BATCH):
        try:
            c.download_from_selection(downloadDir=os.path.join(ROOT, "raw", "nlst"), seriesInstanceUID=todo[i:i + BATCH],
                                      dirTemplate=TMPL, use_s5cmd_sync=True, show_progress_bar=False, quiet=True,
                                      source_bucket_location="gcs")
        except Exception as e:
            print(f"  batch error: {e}", flush=True)
            time.sleep(20)
        print(f"  pass {p}: batch {i//BATCH + 1}/{(len(todo)+BATCH-1)//BATCH} done", flush=True)
left = incomplete()
print(f"FINISHED: {len(meta) - len(left)} complete, {len(left)} incomplete", flush=True)
