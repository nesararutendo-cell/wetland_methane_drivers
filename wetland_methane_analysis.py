# Hydroclimatic and Human Controls on Wetland Methane Emissions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import shap

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()
DATA_FILE = BASE / "wetland_ch4_daily.csv"

try: 
    get_ipython 
    show = lambda fig: plt.show()
except NameError:
    show = lambda fig: plt.close(fig)

FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
print("Figures will be saved in", FIG)
GROUP_ORDER = ["Natural", "Restored", "Rice", "Drained"]
COLORS = {"Natural": "#2a78d6", "Drained": "#eb6834", "Restored": "#1baf7a", "Rice": "#eda100"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 10})

# 1. Load and clean
d = pd.read_csv(DATA_FILE)
if "DATE" not in d.columns:
    d = pd.read_csv(DATA_FILE, sep=";")
if "DATE" not in d.columns:
    raise SystemExit(f"No DATE column in {DATA_FILE}. Columns found: {list(d.columns)[:8]}")
d["DATE"] = pd.to_datetime(d["DATE"])
d = d.sort_values(["SITE_ID", "DATE"]).reset_index(drop=True)
d["F"] = d["FCH4"] * 86400 * 16.04e-6  # nmol m-2 s-1 to mg CH4 m-2 d-1
d["GROUP"] = np.where(d["LAND_USE"] == "Human-modified", d["SITE_CLASSIFICATION"], d["LAND_USE"])
d.loc[d["SITE_ID"] == "FI-Lom", "P"] = np.nan  # implausible precipitation record
d["T"] = d["TS"].fillna(d["TA"])
d["DELTA"] = d["DELTA"].astype(str).str.lower().eq("true")

sites = (d.groupby("SITE_ID")
           .agg(group=("GROUP", "first"), name=("SITE_NAME", "first"), delta=("DELTA", "first"),
                lat=("LAT", "first"), days_ch4=("F", "count"), mean_F=("F", "mean"),
                has_WTD=("WTD", lambda x: x.notna().any())))
print(f"{len(sites)} sites, {d['F'].notna().sum():,} site-days with methane data")
print(sites.groupby("group").agg(sites=("name", "size"), site_days=("days_ch4", "sum"),
                                 with_WTD=("has_WTD", "sum")).loc[GROUP_ORDER])

# 2. How much methane does each land-use group emit?

site_means = sites.reset_index()
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, g in enumerate(GROUP_ORDER):
    v = site_means.loc[site_means["group"] == g, "mean_F"]
    x = i + np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
    ax.scatter(x, v, s=46, color=COLORS[g], edgecolor="white", linewidth=1.2, zorder=3)
    ax.hlines(v.median(), i - 0.28, i + 0.28, color="#0b0b0b", linewidth=2, zorder=4)
    ax.text(i + 0.31, v.median(), f"{v.median():.0f}", va="center", fontsize=9, color="#52514e")
ax.set_yscale("log")
ax.set_xticks(range(4), [f"{g}\n(n={(site_means.group == g).sum()})" for g in GROUP_ORDER])
ax.set_ylabel("Mean CH₄ flux (mg CH₄ m⁻² d⁻¹, log scale)")
ax.set_title("Mean methane flux per site, by land use (bar = median)", loc="left", fontsize=11)
ax.set_ylim(0.5, 1500)
fig.tight_layout(); fig.savefig(FIG / "fig1_flux_by_land_use.png", dpi=200)
show(fig)
print(site_means.groupby("group")["mean_F"].describe().loc[GROUP_ORDER].round(1))

# 3. Temperature sensitivity (apparent Q10)

rows = []
for s, g in d[(d["F"] > 0) & d["T"].notna()].groupby("SITE_ID"):
    b = np.polyfit(g["T"], np.log(g["F"]), 1)[0]
    rows.append({"SITE_ID": s, "group": g["GROUP"].iloc[0], "Q10": np.exp(10 * b),
                 "r": np.corrcoef(g["T"], np.log(g["F"]))[0, 1], "n_days": len(g)})
q10 = pd.DataFrame(rows)

q10["short_record"] = q10["SITE_ID"].isin(["US-HRA", "US-HRC"])

fig, ax = plt.subplots(figsize=(7, 4.2))
for i, g in enumerate(GROUP_ORDER):
    sub = q10[q10["group"] == g]
    x = i + np.random.default_rng(1).uniform(-0.12, 0.12, len(sub))
    y = sub["Q10"].clip(upper=30)
    ax.scatter(x[~sub.short_record], y[~sub.short_record], s=46, color=COLORS[g],
               edgecolor="white", linewidth=1.2, zorder=3)
    ax.scatter(x[sub.short_record], y[sub.short_record], s=46, facecolor="none",
               edgecolor=COLORS[g], linewidth=1.5, zorder=3)
    med = sub.loc[~sub.short_record, "Q10"].median()
    ax.hlines(med, i - 0.28, i + 0.28, color="#0b0b0b", linewidth=2, zorder=4)
    ax.text(i + 0.31, med, f"{med:.1f}", va="center", fontsize=9, color="#52514e")
ax.axhline(1, color="#52514e", linewidth=1, linestyle="--")
ax.text(3.45, 1.08, "no response", fontsize=8, color="#52514e", ha="right")
ax.set_yscale("log"); ax.set_ylim(0.08, 40)
ax.set_xticks(range(4), GROUP_ORDER)
ax.set_ylabel("Apparent Q10 (log scale)")
ax.set_title("Temperature sensitivity of methane, by land use (bar = median)", loc="left", fontsize=11)
ax.text(0.01, 0.02, "Hollow = single-season rice records (US-HRA, US-HRC), capped at 30, excluded from median",
        transform=ax.transAxes, fontsize=7.5, color="#52514e")
fig.tight_layout(); fig.savefig(FIG / "fig2_temperature_sensitivity.png", dpi=200)
show(fig)
print(q10[~q10.short_record].groupby("group")["Q10"].describe().loc[GROUP_ORDER].round(2))

# 4. Case study: one climate, three land uses (Sacramento–San Joaquin Delta)

delta = d[d["DELTA"] & d["F"].notna()].copy()
delta["month"] = delta["DATE"].dt.month

monthly = (delta.groupby(["GROUP", "SITE_ID", "month"])["F"].mean()
                .groupby(["GROUP", "month"]).mean().unstack(0))
fig, ax = plt.subplots(figsize=(7, 4.2))
for g in ["Restored", "Rice", "Drained"]:
    ax.plot(monthly.index, monthly[g], color=COLORS[g], linewidth=2, marker="o", markersize=5)
    ax.text(12.25, monthly[g].iloc[-1], g, color="#0b0b0b", va="center", fontsize=9)
ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
ax.set_xlim(0.7, 13.3)
ax.set_ylabel("CH₄ flux (mg CH₄ m⁻² d⁻¹)")
ax.set_title("Delta sites: monthly methane flux by land use (mean of site means)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "fig3_delta_case_study.png", dpi=200)
show(fig)
annual = delta.groupby(["GROUP", "SITE_ID"])["F"].mean().groupby("GROUP").agg(["mean", "count"])
annual["g_CH4_m2_yr"] = annual["mean"] * 365 / 1000
print(annual.round(1))

# 5. Machine learning: which drivers matter, at sites the model has never seen?
g = d.groupby("SITE_ID")
d["TA_7d"]  = g["TA"].transform(lambda x: x.rolling(7, min_periods=4).mean())
d["TA_30d"] = g["TA"].transform(lambda x: x.rolling(30, min_periods=15).mean())
d["P_30d"]  = g["P"].transform(lambda x: x.rolling(30, min_periods=15).sum())
d["SW_7d"]  = g["SW_IN"].transform(lambda x: x.rolling(7, min_periods=4).mean())
d["ABS_LAT"] = d["LAT"].abs()
for k in ["Restored", "Rice", "Drained"]:
    d[f"is_{k}"] = (d["GROUP"] == k).astype(int)

FEATURES = ["TS", "TA", "TA_7d", "TA_30d", "WTD", "P_30d", "SW_7d", "ABS_LAT",
            "is_Restored", "is_Rice", "is_Drained"]
LABELS = {"TS": "Soil temperature", "TA": "Air temperature", "TA_7d": "Air temp, 7-day mean",
          "TA_30d": "Air temp, 30-day mean", "WTD": "Water table depth", "P_30d": "Precipitation, 30-day",
          "SW_7d": "Sunlight, 7-day mean", "ABS_LAT": "Latitude",
          "is_Restored": "Land use: restored", "is_Rice": "Land use: rice", "is_Drained": "Land use: drained"}

m = d[d["F"].notna()].reset_index(drop=True)
X, y, groups = m[FEATURES], m["F"].values, m["SITE_ID"].values

models = {
    "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5,
                                           n_jobs=-1, random_state=0),
    "Gradient boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                       min_samples_leaf=50, random_state=0),
}
cv = GroupKFold(n_splits=5)
results, preds = [], {}
for name, model in models.items():
    p = np.zeros_like(y)
    for tr, te in cv.split(X, y, groups):
        model.fit(X.iloc[tr], y[tr]); p[te] = model.predict(X.iloc[te])
    preds[name] = p
    site_r = pd.Series({s: np.corrcoef(grp["F"], grp["p"])[0, 1]
                        for s, grp in m.assign(p=p).groupby("SITE_ID")})
    results.append({"model": name, "R² (all held-out days)": r2_score(y, p),
                    "median r within site (timing)": site_r.median(),
                    "sites with r > 0.5": f"{(site_r > 0.5).sum()} / {len(site_r)}"})
print(pd.DataFrame(results).round(2).to_string(index=False))


rf = models["Random forest"].fit(X, y)
sample = X.sample(3000, random_state=0)
sv = shap.TreeExplainer(rf).shap_values(sample)
imp = (pd.Series(np.abs(sv).mean(0), index=FEATURES)
         .rename(LABELS).sort_values())

fig, ax = plt.subplots(figsize=(7, 4.4))
cols = ["#2a78d6" if not k.startswith("Land use") else "#eb6834" for k in imp.index]
ax.barh(imp.index, imp.values, color=cols, height=0.6)
for i, v in enumerate(imp.values):
    ax.text(v + imp.max() * 0.01, i, f"{v:.0f}", va="center", fontsize=8.5, color="#52514e")
ax.set_xlabel("Mean |SHAP| (mg CH₄ m⁻² d⁻¹)")
ax.set_title("What drives daily methane? SHAP importance (orange = land use)", loc="left", fontsize=11)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "fig4_shap_importance.png", dpi=200)
show(fig)
print(imp.sort_values(ascending=False).round(1))

# How the soil-temperature effect differs by land use
sg = m.loc[sample.index, "GROUP"].values
ts_col = FEATURES.index("TS")
fig, ax = plt.subplots(figsize=(7, 4.2))
for gname in GROUP_ORDER:
    k = (sg == gname) & sample["TS"].notna().values
    ax.scatter(sample["TS"].values[k], sv[k, ts_col], s=10, alpha=0.55, color=COLORS[gname],
               label=gname, edgecolor="none")
ax.axhline(0, color="#52514e", linewidth=1)
ax.set_xlabel("Soil temperature (°C)")
ax.set_ylabel("SHAP value of soil temperature\n(mg CH₄ m⁻² d⁻¹)")
ax.set_title("Soil-temperature effect on predicted methane, by land use", loc="left", fontsize=11)
ax.legend(frameon=False, markerscale=2.5, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig5_shap_temperature_by_land_use.png", dpi=200)
show(fig)

print("\nDone. All five figures are in", FIG)
