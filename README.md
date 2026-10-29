# Hydroclimatic and Human Controls on Wetland Methane Emissions

How do temperature, water table and human land use together control daily methane (CH₄) emissions from wetlands? This project analyses **44,000+ site-days** of eddy covariance data from **37 sites** in the [FLUXNET-CH₄ Community Product](https://fluxnet.org/data/fluxnet-ch4-community-product/), comparing natural, restored, rice and drained wetlands, and uses machine learning with SHAP to rank the drivers.

**Author:** Rutendo Nesara, MS Applied Data Science, Clarkson University

\---

## Key findings

**1. Draining a wetland removes its temperature response.**
In natural and restored wetlands, methane rises about 3.3–3.4× for every 10 °C of warming (median apparent Q10). At drained sites the median Q10 is 0.7: warming no longer increases emissions.

!\[Temperature sensitivity by land use](figures/fig2\_temperature\_sensitivity.png)

**2. In one climate, land use sets emissions.**
In California's Sacramento–San Joaquin Delta, ten sites share a Mediterranean climate but differ in land use. Wetlands restored on formerly drained farmland emit about **66 g CH₄ m⁻² yr⁻¹**, rice about **17** and drained farmland about **3.5**, so rewetting raised emissions about 19-fold. Rice follows its flooding schedule rather than the seasons.

!\[Delta case study](figures/fig3\_delta\_case\_study.png)

**3. Weather explains *when* wetlands emit, not <i>how much</i>.**
Random forest models tested only on sites they never saw during training capture seasonal timing well (median within-site r = 0.66; 27 of 37 sites above 0.5) but predict emission size poorly (R² = 0.21). Size depends on factors outside these features: vegetation, soil carbon and water management.

!\[SHAP driver importance](figures/fig4\_shap\_importance.png)

|Group|Sites|Median site-mean flux (mg CH₄ m⁻² d⁻¹)|Median apparent Q10|
|-|-|-|-|
|Natural (bog, fen, marsh, swamp)|18|68|3.4|
|Restored|5|172|3.3|
|Rice|7|81|2.8\*|
|Drained|7|16|0.7|

\*Excludes two single-season rice records (US-HRA, US-HRC) whose Q10 reflects flooding management.

More figures: [flux by land use](figures/fig1_flux_by_land_use.png), [temperature effect by land use (SHAP)](figures/fig5_shap_temperature_by_land_use.png).

\---

## Data

* **Source:** FLUXNET-CH₄ Community Product, CC-BY-4.0 subset, daily (DD) files (Delwiche et al., 2021).
* **Sites:** 37 sites classified in the FLUXNET-CH₄ metadata as bog, fen, marsh or swamp (natural), rice or drained. Five Delta marshes built on formerly drained land (US-Myb, US-Tw1, US-Tw4, US-Tw5, US-Sne) are grouped as *restored*. Upland, lake, tundra, salt marsh and mangrove sites are excluded.
* **Variables:** gap-filled CH₄ flux (FCH4\_F), air and soil temperature, water table depth, precipitation, incoming shortwave radiation.
* **Quality control:** −9999 treated as missing; FI-Lom precipitation (\~5,500 mm yr⁻¹, about ten times realistic) set to missing.

The data are not redistributed in this repository. To reproduce, download them yourself (free account) from [fluxnet.org](https://fluxnet.org/data/download-data/) → FLUXNET-CH₄ CC-BY-4.0 Data, including the `AA-Flx` metadata file.

## Methods

1. **Unit conversion:** nmol m⁻² s⁻¹ → mg CH₄ m⁻² d⁻¹.
2. **Apparent Q10 per site:** linear fit of ln(CH₄) on soil temperature (air temperature where soil is missing), Q10 = exp(10 × slope).
3. **Machine learning:** random forest and histogram gradient boosting (scikit-learn) predicting daily CH₄ flux from temperature (daily, 7- and 30-day means), water table depth, 30-day precipitation, 7-day sunlight, latitude and land-use group.
4. **Validation:** 5-fold `GroupKFold` by site, so every prediction is for a site held out of training.
5. **Interpretation:** SHAP `TreeExplainer` on the random forest.

|Model|R² (held-out sites)|Median within-site r|
|-|-|-|
|Random forest|0.21|0.66|
|Gradient boosting|0.14|0.66|

## Reproduce

```bash
pip install -r requirements.txt

\# put the downloaded site zips and the AA-Flx metadata in data/

python load\_fluxnet\_ch4.py        # writes wetland\_ch4\_daily.csv and site\_summary.csv

python wetland\_methane\_analysis.py

```

Runs in about 5 minutes on a laptop.

## Limitations and next steps

* Gap-filled fluxes include days estimated from meteorology; a stricter version would keep mostly measured days.
* Plant productivity (GPP), a known methane driver, is not yet included and should improve cross-site skill.
* Water table depth is missing at 13 of 37 sites, and its reference level differs between sites.
* **Next:** add GPP, then MODIS NDVI, land surface temperature and land cover from Google Earth Engine, to test whether satellite data can replace tower measurements for mapping emissions where no towers exist.

## Repository

```
load\_fluxnet\_ch4.py               load, filter and merge the daily site files

wetland\_methane\_analysis.py       full analysis: statistics, models, figures

figures/                          exported figures

data/                             put downloaded FLUXNET-CH4 files here (not tracked)

```

## Citation and acknowledgements

Delwiche, K. B., Knox, S. H., Malhotra, A., et al. (2021). FLUXNET-CH₄: a global, multi-ecosystem dataset and analysis of methane seasonality from freshwater wetlands. *Earth System Science Data*, 13, 3607–3689. https://doi.org/10.5194/essd-13-3607-2021

This work uses data from the FLUXNET-CH₄ Community Product. I thank all site teams for collecting and sharing their data; individual site PIs are listed in the FLUXNET-CH₄ metadata.

Code: MIT License.

