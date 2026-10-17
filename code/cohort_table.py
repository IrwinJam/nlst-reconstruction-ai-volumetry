"""table 1, participant characteristics from the nlst clinical tables (D11)"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from idc_index import IDCClient

ROOT = Path(r"C:\MetaSeg_data")
OUT = ROOT / "results" / "cohort"
# histology groups from the icd-o-3 codes in de_type
HIST = {"Adenocarcinoma": {8140, 8250, 8251, 8252, 8253, 8254, 8255, 8260, 8310, 8323, 8480, 8481, 8490, 8550, 8551, 8560},
        "Squamous cell carcinoma": {8070, 8071, 8072, 8073, 8074, 8075, 8083},
        "Small cell carcinoma": {8041, 8042, 8043, 8044, 8045},
        "Non-small cell carcinoma, NOS": {8046},
        "Large cell carcinoma": {8012, 8013, 8014}}
STAGE = {"110": "I", "120": "I", "210": "II", "220": "II", "310": "III", "320": "III", "400": "IV"}


def hist_group(x):
    try:
        code = int(str(x).strip())
    except ValueError:
        return "Other / unknown"
    for g, s in HIST.items():
        if code in s:
            return g
    return "Other / unknown"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = IDCClient(); c.fetch_index("clinical_index")
    prsn = c.get_clinical_table("nlst_prsn")
    prsn["pid"] = prsn["dicom_patient_id"].astype(str)
    prsn = prsn.drop_duplicates("pid").set_index("pid")
    sp = json.loads((ROOT / "results" / "split.json").read_text())
    excl = set(pd.read_csv(ROOT / "results" / "qc" / "exclusions_D5.csv", dtype={"patient_id": str}).patient_id)
    pts = pd.read_csv(ROOT / "cache" / "NLST_v2" / "patients.csv", dtype={"patient_id": str}).set_index("patient_id")
    groups = {"Training": sp["train"], "Validation": sp["val"], "Test": sp["test"],
              "Paired reconstructions (Aim 1)": [p for p in sp["aim1"] if p not in excl]}
    rows, cols = [], {}

    def add(label, fn):
        rows.append(label)
        for g, ids in groups.items():
            cols.setdefault(g, []).append(fn(ids))

    pct = lambda n, d: f"{n} ({100 * n / d:.0f})" if d else "0"
    linked = lambda ids: [p for p in ids if p in prsn.index]
    add("Participants, n", lambda ids: str(len(ids)))
    add("Linked to NLST clinical data, n", lambda ids: str(len(linked(ids))))
    add("Age at randomization, y, mean ± SD", lambda ids: f"{prsn.loc[linked(ids), 'age'].astype(float).mean():.1f} ± {prsn.loc[linked(ids), 'age'].astype(float).std():.1f}")
    add("Male sex", lambda ids: pct(int((prsn.loc[linked(ids), 'gender'].astype(str) == '1').sum()), len(linked(ids))))
    add("Current smoker at baseline", lambda ids: pct(int((prsn.loc[linked(ids), 'cigsmok'].astype(str) == '1').sum()), len(linked(ids))))
    add("White race", lambda ids: pct(int((prsn.loc[linked(ids), 'race'].astype(str) == '1').sum()), len(linked(ids))))
    for h in list(HIST) + ["Other / unknown"]:
        add(f"Histology: {h}", lambda ids, h=h: pct(int((prsn.loc[linked(ids), 'de_type'].map(hist_group) == h).sum()), len(linked(ids))))
    for s_ in ["I", "II", "III", "IV", "Other / unknown"]:
        add(f"Stage (AJCC 7th): {s_}", lambda ids, s_=s_: pct(int((prsn.loc[linked(ids), 'de_stag_7thed'].astype(str).map(STAGE).fillna("Other / unknown") == s_).sum()), len(linked(ids))))
    for v in ["GE MEDICAL SYSTEMS", "SIEMENS", "TOSHIBA", "Philips"]:
        add(f"Vendor: {dict(zip(['GE MEDICAL SYSTEMS','SIEMENS','TOSHIBA','Philips'],['GE','Siemens','Toshiba','Philips']))[v]}", lambda ids, v=v: pct(int((pts.loc[ids, 'vendor'] == v).sum()), len(ids)))
    add("Slice thickness, mm, median (range)", lambda ids: f"{pts.loc[ids, 'slice_thickness_mm'].median():.1f} ({pts.loc[ids, 'slice_thickness_mm'].min():.2f}–{pts.loc[ids, 'slice_thickness_mm'].max():.1f})")
    add("Expert tumor volume, mL, median (IQR)", lambda ids: f"{pts.loc[ids, 'expert_volume_ml_native'].median():.2f} ({pts.loc[ids, 'expert_volume_ml_native'].quantile(.25):.2f}–{pts.loc[ids, 'expert_volume_ml_native'].quantile(.75):.2f})")
    t = pd.DataFrame(cols, index=rows)
    t.to_csv(OUT / "table1.csv")
    md = ["| Characteristic | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
    md += [f"| {r} | " + " | ".join(t.loc[r]) + " |" for r in t.index]
    (OUT / "table1.md").write_text("\n".join(md) + "\n\nValues are n (%) unless stated. Percentages of linked participants for clinical rows.\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
