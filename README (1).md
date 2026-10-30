# Hispanic Heritage Historical Data Explorer

A Streamlit classroom app for retrieving World Bank historical indicators, editing a validated working dataset, fitting centered-time polynomial models, exploring user-defined functions, and exporting CSV, Excel, and PDF reports.

## Run locally

Use Python 3.10 or newer (Python 3.11 recommended).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```

The app retrieves data from the official World Bank Indicators API, so internet access is required for fresh retrieval. No credentials are required.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Open Streamlit Community Cloud and connect the repository.
4. Select `app.py` as the main file and deploy.
5. Open the deployed app and verify the data source, editing/reset, model, and downloads.

## Deploy to Render

Create a **Python Web Service** linked to the GitHub repository.

- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

## What the app does

- Offers Mexico, Brazil, Argentina, Colombia, Chile, Peru, Venezuela, Ecuador, and Uruguay.
- Requests years from current year minus 70 through current year minus 1 and reports actual non-null coverage and count.
- Keeps retrieved source data separate from the editable working table; reset returns the table to source observations.
- Fits degree 3–8 polynomials using `t = year - base_year`, reports in-sample R²/MAE/residuals, and offers a chronological holdout when enough data remain.
- Supports single-year and batch model estimates, country comparisons, function entry and derivatives, graph controls, contextual interpretations, and CSV/XLSX/PDF exports.

## Verification checklist

1. Compare the country, indicator code, and values against the World Bank API link shown in the app.
2. Confirm the requested window and actual coverage are different concepts; missing years must not be filled with fabricated values.
3. Edit a value and confirm the user-modified scenario warning appears. Click **Reset to source data** and confirm it clears.
4. Remove rows until fewer than `degree + 1` distinct years remain; fitting should be disabled with an explanation.
5. Create duplicate or non-integer years; validation should report them.
6. Confirm the displayed equation uses centered time and the same base year for single and batch predictions.
7. Enter the same year in the single-year calculator and custom batch years; estimates should agree to rounding.
8. Try `1/(x-2)` and a logarithm such as `log(x)`; inspect undefined regions and the domain notes.
9. Test invalid expressions such as `x + unknown_name`; the app should show a parse error.
10. Change each selector, degree, sample increment, horizon, comparison country, course level, graph visibility checkbox, context, units, and interval; confirm the corresponding display updates.
11. Download CSV, XLSX, and PDF and open each with its intended reader. Excel should contain Summary, Working Data, and Calculated Years sheets.
12. On deployment, inspect the Streamlit logs if API requests time out, dependencies fail, or an export errors.

## Known limitations

This is a complete classroom-oriented starting app, but it has not been executed in this file-generation environment. Do not describe it as runtime-tested until you run the checklist above. The World Bank API can return missing observations or temporary errors. The polynomial model is descriptive and not a causal model. High-degree fits and extrapolations can be unstable. Calculus root finding is a numerical/symbolic aid, not a proof system for every possible expression; absolute extrema are numerically approximated on a grid. The “Play one step” control advances the simulation one step per click; it is not continuous timed animation. The normalized contextual gauge is illustrative, not a physical simulation. No validated 95% prediction intervals are claimed.
