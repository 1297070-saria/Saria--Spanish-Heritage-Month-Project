# Hispanic Heritage Historical Data Explorer

A single-file Streamlit app for exploring World Bank historical indicators for Mexico, Brazil, Argentina, Colombia, Chile, Peru, Venezuela, Ecuador, and Uruguay; fitting centered-time polynomials; editing/validating working data; comparing countries; analyzing custom functions; and exporting CSV, Excel, and PDF reports.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m streamlit run app.py
```

The app retrieves observations for the current year minus 70 through the previous year. The number of usable observations depends on World Bank reporting and is shown in the app. Requests are cached for one hour. No credentials are needed.

## Deploy on GitHub + Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Open Streamlit Community Cloud, connect GitHub, choose the repository and branch, and set the app file to `app.py`.
4. Deploy. The app needs outbound internet access to reach the World Bank API.
5. If deployment fails, open the app's **Manage app → Logs** and check the Python version, dependency installation, traceback, and network/API errors.

## Deploy on Render

Create a **Python Web Service** connected to the GitHub repository.

- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

No secret environment variables are required by this app.

## Controls and data flow

1. Country and indicator controls select a World Bank country/indicator endpoint.
2. `fetch_world_bank` requests JSON and caches successful results for one hour.
3. The original API table is kept separately from the editable working table. Validation rejects missing/non-numeric cells, fractional years, duplicate years, non-finite values, and selected indicator-specific invalid ranges. **Reset to source** restores the fetched table.
4. A degree 3–8 polynomial is fit to values using centered time `t = year - base_year`, reducing numerical conditioning problems. At least degree + 1 distinct years are required.
5. Single-year and batch estimates use the same stored polynomial coefficients. Results are invalidated when the data, country, indicator, or degree changes.
6. Country comparisons retrieve the same indicator for each selected country and plot available values without fabricating missing observations.
7. The function lab supports polynomial, rational, exponential, logarithmic, and trigonometric forms, parameterized suggestions, and user-entered expressions. SymPy provides derivatives; numerical sign-change checks give approximate calculus features.
8. CSV, XLSX, and PDF exports are generated as actual files and reflect the current working data, model summary, and calculated results.

## Verification checklist

- Confirm indicator code, units, definition, source URL, and retrieval date.
- Confirm the displayed coverage and count can be fewer than 70.
- Edit a year/value, confirm scenario status changes, then reset and confirm the source values return.
- Try a duplicate year, a fractional year, a blank cell, and a nonnumeric value; each should be rejected with a visible error.
- Select a degree that is too high for the number of distinct years; fitting should fail clearly.
- Compare a single-year result with the same year in batch mode; estimates should agree because both use `poly_predict`.
- Try `1/(x-2)` near x = 2 and `log(x)` for x <= 0; undefined values should not be plotted as ordinary finite values.
- Change country, indicator, degree, sample increment, horizon, comparison countries, course mode, context, function source, interval, and point controls; verify their associated graph, model, or explanations update.
- Download CSV, XLSX, and PDF; confirm Excel has Summary, Working Data, and Calculated Years sheets, and PDF includes metadata, equation/metrics when available, chart, preview, results, and limitations.
- After deployment, test the app on a fresh browser session and inspect logs for API timeouts or dependency errors.

## Mathematical limitations

Polynomial fitting is descriptive, not causal. R² and MAE are in-sample unless marked chronological holdout. MAE is an error summary, not a prediction interval. High-degree polynomials can oscillate and extrapolate implausibly; the app warns but does not clip results. The calculus root search is numerical and may miss even-multiplicity roots, discontinuities, or narrow features. Numerical extrema are approximations and are not exact proofs. The contextual visual is a normalized teaching illustration, not a physical or clinical simulator.

## Important implementation note

The UI offers slider-driven synchronized graph and visual updates. Automatic timed playback is not implemented: reliable browser-side animation requires a dedicated component or timer mechanism that is not part of this dependency-light Streamlit implementation. The slider can be moved manually to simulate playback.
