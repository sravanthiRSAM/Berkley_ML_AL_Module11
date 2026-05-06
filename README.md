# What Drives the Price of a Used Car?

A regression analysis of ~427K used-car listings to identify the features that most strongly influence sale price, with actionable recommendations for used-car dealers.

**Notebook:** [prompt_II.ipynb](prompt_II.ipynb)

---

## Business Question
*What features of a used car most strongly drive its sale price, and how can a dealer use this insight to fine-tune inventory?*

## Approach (CRISP-DM)
1. **Business Understanding** — Reframed as a supervised regression problem with `price` as the continuous target.
2. **Data Understanding** — EDA on 427K listings; identified extreme outliers, heavy missingness in several columns, and useless identifiers.
3. **Data Preparation** — Filtered listings to the realistic range ($1K–$100K, ≤300K miles, year ≥1990), dropped useless / high-cardinality columns (`id`, `VIN`, `model`, `region`, `size`), engineered an `age` feature, imputed categorical NaN as `'unknown'`, and log-transformed the price target. Final cleaned dataset: ~363K rows.
4. **Modeling** — Built three regression models (Linear, Ridge, Lasso) inside an sklearn `Pipeline` with `StandardScaler` for numeric features and `OneHotEncoder` for categoricals. Tuned `alpha` for Ridge and Lasso via 5-fold `GridSearchCV`.
5. **Evaluation** — Compared models on R², MAE (in dollars), and RMSE (in dollars).
6. **Deployment** — Translated coefficient findings into dealer-facing recommendations.

## Key Results

| Model | R² (test) | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.754 | $4,839 | $7,723 |
| Ridge (α tuned) | 0.754 | $4,839 | $7,723 |
| Lasso (α tuned) | 0.753 | $4,839 | $7,711 |

The model explains **~75% of the variation in used-car prices** with a typical error of about **$4,800 per vehicle**.

## Top Drivers of Used-Car Price

**Increase price (vs. a baseline reference vehicle):**
- Luxury/specialty brands: Porsche (+69%), Tesla (+65%), Lexus (+26%), Land Rover (+18%)
- Body type: off-road (+34%), convertible (+27%), trucks (+21%), pickups (+21%)
- Powertrain: 12-cylinder (+34%), manual transmission (+16%), diesel fuel (baseline / premium category)

**Decrease price:**
- Condition: "fair" (−47%), "salvage" (−42%)
- Title status: missing/salvage title (−42%)
- Small engines: 3-cylinder (−39%), 4-cylinder (−37%)
- Age and mileage (steady linear depreciation)

## Recommendations for Dealers

1. **Prioritize inventory under ~10 years old with under 100K miles** — age and mileage dominate price.
2. **Stock trucks, pickups, and 4WD SUVs (especially diesel)** — the strongest body-type and fuel premiums.
3. **Carry select luxury inventory** (Porsche, Tesla, Lexus, Land Rover) for high-margin opportunities.
4. **Avoid salvage titles and "fair" condition** unless deeply discounted; the price penalty is severe.
5. **Invest in light reconditioning** to move vehicles from "good" → "excellent"; the price lift covers reasonable costs.
6. **Don't overpay for cosmetic factors** like paint color or region — coefficients are small.

## Project Files

- [prompt_II.ipynb](prompt_II.ipynb) — main analysis notebook (CRISP-DM phases, modeling, findings).
- [data/vehicles.csv](data/vehicles.csv) — input dataset (~427K used-car listings, 18 columns).
- [build_notebook.py](build_notebook.py) — script that generates the notebook structure.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook prompt_II.ipynb
```
