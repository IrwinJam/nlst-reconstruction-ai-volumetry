"""downloads the LIDC-annot-NLST501 reader segs for our patients"""
import io, os, sys, time, zipfile
import pandas as pd, requests

ROOT = r"C:\MetaSeg_data"
BASE = os.path.join(ROOT, "raw", "lidc_annot_nlst501")
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"


def fetch(uid, dest, tries=4):
    for t in range(tries):
        try:
            r = requests.get(API, params={"SeriesInstanceUID": uid}, timeout=300)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".dcm")]
                if not names:
                    raise ValueError("zip has no DICOM files")
                os.makedirs(dest, exist_ok=True)
                for n in names:
                    with open(os.path.join(dest, os.path.basename(n)), "wb") as f:
                        f.write(z.read(n))
            return len(names), len(r.content) / 1e6
        except Exception as e:
            print(f"  retry {t + 1}/{tries} for {uid}: {type(e).__name__}: {e}", flush=True)
            time.sleep(10 * (t + 1))
    return 0, 0.0


def main(limit=None):
    sub = pd.read_csv(os.path.join(BASE, "our_seg_subset.csv"), dtype={"patient_id": str})
    if limit:
        sub = sub.head(limit)
    rows, t0 = [], time.time()
    for i, r in enumerate(sub.itertuples(), 1):
        dest = os.path.join(BASE, "seg", r.patient_id, r.seg_uid)
        if os.path.isdir(dest) and os.listdir(dest):
            rows.append({"patient_id": r.patient_id, "seg_uid": r.seg_uid, "files": len(os.listdir(dest)), "status": "present"})
            continue
        n, mb = fetch(r.seg_uid, dest)
        rows.append({"patient_id": r.patient_id, "seg_uid": r.seg_uid, "files": n, "MB": round(mb, 2),
                     "status": "ok" if n else "FAILED"})
        if i % 20 == 0 or i == len(sub):
            print(f"{i}/{len(sub)} series, {time.time() - t0:.0f} s", flush=True)
    log = pd.DataFrame(rows)
    log.to_csv(os.path.join(BASE, "download_log.csv"), index=False)
    print(log.status.value_counts().to_dict(), f"patients: {log[log.files > 0].patient_id.nunique()}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
