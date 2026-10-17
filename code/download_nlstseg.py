"""builds the nlstseg manifest and downloads masks + ct from IDC"""
from __future__ import annotations
import sys, time
from pathlib import Path
from idc_index import IDCClient

ROOT = Path(r"C:\MetaSeg_data")
OUT = ROOT / "raw" / "nlst"
MANIFEST = ROOT / "nlstseg_manifest.csv"


def main():
    c = IDCClient()
    c.fetch_index("seg_index")
    m = c.sql_query("""
        SELECT sg.PatientID, sg.SeriesInstanceUID AS seg_series, ct.SeriesInstanceUID AS ct_series,
               ct.StudyInstanceUID, ct.StudyDate, ct.Manufacturer, ct.ManufacturerModelName,
               ct.SeriesDescription, ct.instanceCount, ct.series_size_MB, s.total_segments
        FROM index sg
        JOIN seg_index s ON s.SeriesInstanceUID = sg.SeriesInstanceUID
        LEFT JOIN index ct ON ct.SeriesInstanceUID = s.segmented_SeriesInstanceUID
        WHERE sg.analysis_result_id = 'nlstseg' AND sg.Modality = 'SEG'
        ORDER BY sg.PatientID
    """)
    # series descriptions look like "1,OPA,GE,HSQX,STANDARD,310,2.5,120,..."
    # field 0 = screen round, 4 = kernel, 6 = slice thickness mm
    parts = m["SeriesDescription"].fillna("").str.split(",")
    m["screen_round"] = parts.str[0]
    m["kernel"] = parts.str[4]
    m["slice_thickness_mm"] = parts.str[6]
    m.to_csv(MANIFEST, index=False)
    missing = m["ct_series"].isna().sum()
    print(f"manifest: {len(m)} masks, {m['PatientID'].nunique()} patients, "
          f"{m['ct_series'].nunique()} CT series, {m['series_size_MB'].sum()/1024:.1f} GB of CT; "
          f"{missing} masks whose CT is not in the index -> {MANIFEST}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tmpl = "%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID"
    t0 = time.time()
    print("[1/2] downloading masks ...", flush=True)
    c.download_from_selection(downloadDir=str(OUT), seriesInstanceUID=m["seg_series"].tolist(),
                              dirTemplate=tmpl, use_s5cmd_sync=True, show_progress_bar=False, quiet=False)
    print(f"[1/2] masks done ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("[2/2] downloading CT series ...", flush=True)
    c.download_from_selection(downloadDir=str(OUT), seriesInstanceUID=m["ct_series"].dropna().unique().tolist(),
                              dirTemplate=tmpl, use_s5cmd_sync=True, show_progress_bar=False, quiet=False)
    print(f"[2/2] CT done ({(time.time()-t0)/60:.1f} min total)", flush=True)
    print("DOWNLOAD COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
