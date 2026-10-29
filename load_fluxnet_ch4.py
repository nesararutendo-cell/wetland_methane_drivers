"""
Load FLUXNET-CH4 daily (DD) data for the wetland, rice and drained sites
and merge them into one table.

How to run:
  1. Put this script in the folder with the metadata CSV (FLX_AA-Flx_CH4-META_....csv)
     and the site files. The site files can be zips, or already unzipped
     (daily CSV files with _DD_ in the name), and can sit in subfolders.
  2. In a terminal in that folder:  python load_fluxnet_ch4.py
     (or paste it into a Jupyter notebook cell in that folder)

Outputs (in the same folder):
  - wetland_ch4_daily.csv   one row per site per day
  - site_summary.csv        what each site contributed
"""
import glob
import io
import zipfile

import numpy as np
import pandas as pd

# Site types we keep, and how they map to the project's land-use groups
GROUPS = {
    "Bog": "Natural", "Fen": "Natural", "Marsh": "Natural", "Swamp": "Natural",
    "Rice": "Human-modified", "Drained": "Human-modified",
}
# Delta wetlands built on formerly drained farmland
RESTORED = {"US-Myb", "US-Tw1", "US-Tw4", "US-Tw5", "US-Sne"}
DELTA = RESTORED | {"US-Snd", "US-Tw3", "US-Bi1", "US-Bi2", "US-Twt"}

# Column names to look for, in order of preference
CANDIDATES = {
    "FCH4": ["FCH4_F", "FCH4_F_ANNOPTLM", "FCH4"],  # nmol CH4 m-2 s-1
    "TA":   ["TA_F", "TA"],                         # air temperature, deg C
    "TS":   ["TS_1", "TS_F", "TS", "TS_2"],         # shallowest soil temp, deg C
    "WTD":  ["WTD_F", "WTD"],                       # water table depth, m (negative = below surface)
    "P":    ["P_F", "P"],                           # precipitation, mm
    "SW_IN": ["SW_IN_F", "SW_IN"],                  # incoming shortwave radiation, W m-2
    # optional extras for the next model version (left empty if a site lacks them)
    "GPP":  ["GPP_NT", "GPP_DT", "GPP_F", "GPP"],   # gross primary productivity
    "LE":   ["LE_F", "LE"],                         # latent heat flux, W m-2
    "FCH4_QC": ["FCH4_F_ANNOPTLM_QC", "FCH4_F_QC", "FCH4_QC"],  # gap-filling quality flag
}


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


# ---- 1. Metadata: decide which sites belong in the project ----
meta_csv = glob.glob("**/*CH4-META*.csv", recursive=True)
if meta_csv:
    meta = pd.read_csv(meta_csv[0])
else:  # metadata still inside the AA-Flx zip
    zname = glob.glob("*AA-Flx*.zip")[0]
    with zipfile.ZipFile(zname) as z:
        inner = [n for n in z.namelist() if "CH4-META" in n and n.endswith(".csv")][0]
        meta = pd.read_csv(io.BytesIO(z.read(inner)))
meta = meta[(meta["SITE_CLASSIFICATION"].isin(GROUPS))
            & (meta["FLUXNET-CH4_DATA_POLICY"] == "CCBY4.0")].copy()
meta["LAND_USE"] = meta["SITE_CLASSIFICATION"].map(GROUPS)
meta.loc[meta["SITE_ID"].isin(RESTORED), "LAND_USE"] = "Restored"
meta["DELTA"] = meta["SITE_ID"].isin(DELTA)
wanted = set(meta["SITE_ID"])

# ---- 2. Find each site's daily (DD) file: inside a zip, or already unzipped ----
def daily_sources():
    """Yield (filename, loader) for every daily file in this folder or its subfolders."""
    for zpath in sorted(glob.glob("**/*.zip", recursive=True)):
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if "_DD_" in n and n.endswith(".csv"):
                    yield n, (lambda zp=zpath, nm=n: pd.read_csv(
                        io.BytesIO(zipfile.ZipFile(zp).read(nm)), na_values=[-9999]))
    for f in sorted(glob.glob("**/*_DD_*.csv", recursive=True)):
        yield f, (lambda fp=f: pd.read_csv(fp, na_values=[-9999]))
    for f in sorted(glob.glob("**/*_DD_*.xlsx", recursive=True)):
        yield f, (lambda fp=f: pd.read_excel(fp, na_values=[-9999]))


frames, summary, found = [], [], set()
for name, load in daily_sources():
    base = name.replace("\\", "/").split("/")[-1]
    site = base.split("_")[1]  # FLX_US-Myb_FLUXNET-CH4_DD_... -> US-Myb
    if site not in wanted or site in found:
        continue
    raw = load()

    out = pd.DataFrame({"SITE_ID": site,
                        "DATE": pd.to_datetime(raw["TIMESTAMP"].astype(str).str[:8], format="%Y%m%d")})
    used = {}
    for var, names in CANDIDATES.items():
        col = pick(raw, names)
        out[var] = raw[col].values if col else np.nan
        used[var] = col
    frames.append(out)
    found.add(site)
    summary.append({"SITE_ID": site, "days": len(out),
                    "days_with_FCH4": int(out["FCH4"].notna().sum()),
                    **{f"col_{k}": v for k, v in used.items()}})

if not frames:
    raise SystemExit("No daily (DD) files found. Check that the site files are in this folder.")

data = pd.concat(frames, ignore_index=True)
data = data.merge(meta[["SITE_ID", "SITE_NAME", "SITE_CLASSIFICATION", "LAND_USE", "DELTA",
                        "LAT", "LON", "KOPPEN", "IGBP", "DOM_VEG"]], on="SITE_ID")
data.to_csv("wetland_ch4_daily.csv", index=False)

summ = pd.DataFrame(summary).merge(meta[["SITE_ID", "SITE_CLASSIFICATION", "LAND_USE"]], on="SITE_ID")
summ.to_csv("site_summary.csv", index=False)

# ---- 3. Report ----
print(f"Loaded {len(found)} of {len(wanted)} project sites, {len(data):,} site-days.")
print(summ.groupby("LAND_USE")[["days", "days_with_FCH4"]].sum())
missing = sorted(wanted - found)
if missing:
    print("\nNot found in this folder (download these if you want them):")
    for s in missing:
        row = meta.set_index("SITE_ID").loc[s]
        print(f"  {s:8s} {row['SITE_CLASSIFICATION']:8s} {row['SITE_NAME']}")
