import io
import math
import re
import time
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor,
)
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
WB_API = "https://api.worldbank.org/v2"
COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current age-specific mortality rates remained constant."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment; national estimate."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population, expressed in current US dollars; not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual number of live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants during the period; not a count of emigration alone."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a share of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP."),
}
CONTEXTS = {
    "Business": ("Revenue model", "time", "years", "revenue", "currency units", "A business indicator such as revenue over time."),
    "Engineering": ("Position model", "time", "seconds", "position", "meters", "Position of a moving component over time."),
    "Medical": ("Concentration model", "time", "hours", "concentration", "mg/L", "A conceptual concentration curve; not medical advice or dosing guidance."),
    "Sports science": ("Performance model", "time", "seconds", "performance", "score units", "A performance measure over time."),
    "Biology": ("Population model", "time", "days", "population", "organisms", "A population or cell-colony model."),
    "Bioengineering": ("Bioprocess model", "time", "hours", "output", "units", "A conceptual biological process output."),
    "Economic forecasting": ("Economic indicator", "time", "years", "indicator", "index units", "A hypothetical economic indicator."),
    "Weather": ("Temperature model", "time", "hours", "temperature", "°C", "A conceptual temperature trend."),
    "Custom": ("Custom model", "x", "x-units", "f(x)", "y-units", "Describe the meaning and limitations of your model."),
}
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
x = sp.symbols("x", real=True)

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title("📊 " + APP_TITLE)
st.caption("HISPANIC HERITAGE DATA LAB · historical data, mathematical modeling, and interactive function analysis")
st.info("Brazil is included as a Latin American comparison country. Country inclusion does not imply that every resident identifies as Hispanic; Brazil is predominantly Portuguese-speaking.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code, indicator_code, start_year, end_year):
    url = f"{WB_API}/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("The World Bank API returned an unexpected or empty response.")
    rows = []
    for item in payload[1]:
        if item and item.get("date") and item.get("value") is not None:
            try:
                rows.append({"year": int(item["date"]), "value": float(item["value"])})
            except (ValueError, TypeError):
                continue
    frame = pd.DataFrame(rows, columns=["year", "value"])
    if not frame.empty:
        frame = frame.sort_values("year").drop_duplicates("year", keep="last").reset_index(drop=True)
    return frame

def fit_poly(frame, degree):
    clean = validate_frame(frame)
    if clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years; only {clean['year'].nunique()} are available.")
    base = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base
    yv = clean["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, yv, degree)
    pred = np.polyval(coeff, t)
    return {"coeff": coeff, "base": base, "degree": degree, "years": clean["year"].to_numpy(),
            "values": yv, "pred": pred, "r2": float(r2_score(yv, pred)),
            "mae": float(mean_absolute_error(yv, pred)), "mean": float(np.mean(yv)),
            "residuals": yv-pred, "min_year": int(clean.year.min()), "max_year": int(clean.year.max())}

def poly_predict(model, years):
    return np.polyval(model["coeff"], np.asarray(years, dtype=float) - model["base"])

def polynomial_equation(model, digits=8):
    terms = []
    d = model["degree"]
    for i, c in enumerate(model["coeff"]):
        power = d-i
        cs = f"{c:.{digits}g}"
        if power == 0: terms.append(cs)
        elif power == 1: terms.append(f"({cs})t")
        else: terms.append(f"({cs})t^{power}")
    return "y(t) = " + " + ".join(terms) + f", where t = year − {model['base']}"

def validate_frame(frame, indicator_name=None):
    if not {"year", "value"}.issubset(frame.columns):
        raise ValueError("The working table must contain year and value columns.")
    out = frame[["year", "value"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if out.isna().any().any():
        raise ValueError("Years and values must be numeric and cannot be blank.")
    if not np.all(np.isclose(out["year"], np.round(out["year"]))):
        raise ValueError("Years must be integers.")
    out["year"] = out["year"].astype(int)
    if out["year"].duplicated().any():
        raise ValueError("Duplicate years are not allowed. Keep one row per year.")
    if (out["year"] < 1).any() or (out["year"] > 3000).any():
        raise ValueError("Years must be between 1 and 3000.")
    if not np.isfinite(out["value"].to_numpy(dtype=float)).all():
        raise ValueError("Values must be finite numbers.")
    if indicator_name == "Life expectancy at birth" and ((out.value < 0) | (out.value > 120)).any():
        raise ValueError("Life expectancy must be between 0 and 120 years.")
    if indicator_name == "Unemployment, national estimate" and ((out.value < 0) | (out.value > 100)).any():
        raise ValueError("Unemployment percentage must be between 0 and 100.")
    if indicator_name == "Consumer price inflation" and (out.value <= -100).any():
        raise ValueError("Annual CPI inflation must be greater than -100%.")
    return out.sort_values("year").reset_index(drop=True)

def safe_parse(expr_text):
    allowed = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
               "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi,
               "E": sp.E, "x": x}
    if len(expr_text) > 180:
        raise ValueError("Function expression is too long.")
    if re.search(r"[^0-9a-zA-Z_+\-*/^().,\s]", expr_text):
        raise ValueError("Use numbers, x, + − * / ^, parentheses, and supported functions sin, cos, tan, exp, log, sqrt, abs.")
    expr = parse_expr(expr_text, local_dict=allowed, transformations=TRANSFORMS, evaluate=True)
    if expr.has(sp.Symbol) and expr.free_symbols - {x}:
        raise ValueError("Only x may be used as a variable.")
    return sp.simplify(expr)

def numeric_function(expr, vals):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        result = np.asarray(fn(vals), dtype=float)
    if result.shape == ():
        result = np.full(np.asarray(vals).shape, float(result))
    return result

def calc_features(expr, interval, point, secant_a, secant_b):
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    f = sp.lambdify(x, expr, "numpy")
    f1, f2 = sp.lambdify(x, d1, "numpy"), sp.lambdify(x, d2, "numpy")
    def scalar(fn, z):
        try:
            q = complex(fn(float(z)))
            return float(q.real) if abs(q.imag) < 1e-8 and math.isfinite(q.real) else None
        except Exception:
            return None
    ya, yb = scalar(f, secant_a), scalar(f, secant_b)
    avg = (yb-ya)/(secant_b-secant_a) if ya is not None and yb is not None and secant_b != secant_a else None
    yp, slope = scalar(f, point), scalar(f1, point)
    return d1, d2, scalar, f, f1, f2, avg, ya, yb, yp, slope

def context_meaning(context, yname, yunit, xname, xunit, point, value, slope, curvature):
    if value is None:
        return "The function is undefined at this input, so a contextual interpretation is not available."
    lines = [f"At {xname} = {point:g} {xunit}, the modeled {yname} is {value:.6g} {yunit}."]
    if slope is not None:
        lines.append(f"The first derivative is {slope:.6g} {yunit}/{xunit}; locally, the model is changing at this rate.")
    if curvature is not None:
        lines.append(f"The second derivative is {curvature:.6g} {yunit}/{xunit}²; it describes how the rate itself is changing.")
    if context == "Engineering":
        lines.append("If the output is position, the first derivative is velocity and the second derivative is acceleration.")
    elif context in ("Business", "Economic forecasting"):
        lines.append("A positive first derivative means modeled growth; a negative one means decline. The second derivative indicates whether that rate is increasing or decreasing.")
    elif context in ("Biology", "Bioengineering"):
        lines.append("Interpret this as a mathematical scenario, not as verified biological measurements.")
    elif context == "Medical":
        lines.append("This is an educational model only, not clinical guidance.")
    return " ".join(lines)

# Sidebar controls
with st.sidebar:
    st.header("Data and model")
    country_name = st.selectbox("Country", list(COUNTRIES))
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 3)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_names = st.multiselect("Compare same indicator with", [n for n in COUNTRIES if n != country_name])
    printer_mode = st.checkbox("Printer-friendly report view", False)
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    st.caption("Source: World Bank Indicators API. Requests are cached for one hour.")

country_code = COUNTRIES[country_name]
indicator_code, units, definition = INDICATORS[indicator_name]
current_year = date.today().year
start_year, end_year = current_year-70, current_year-1
retrieved_at = date.today().isoformat()
source_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_year}:{end_year}&format=json"

try:
    original = fetch_world_bank(country_code, indicator_code, start_year, end_year)
    source_error = None
except Exception as exc:
    original = pd.DataFrame(columns=["year", "value"])
    source_error = f"{type(exc).__name__}: {exc}"

data_key = f"{country_code}|{indicator_code}|{start_year}|{end_year}"
if st.session_state.get("_data_key") != data_key:
    st.session_state["_data_key"] = data_key
    st.session_state["working_data"] = original.copy()
    st.session_state["scenario_edited"] = False
    st.session_state.pop("calc_results", None)

st.subheader("1. Historical data")
st.write(f"**{country_name} · {indicator_name}** — {definition}")
st.caption(f"Units: {units} · Indicator: {indicator_code} · Requested years: {start_year}–{end_year} · Retrieved: {retrieved_at}")
st.markdown(f"[Open World Bank API source]({source_url})")
if source_error:
    st.error("Could not retrieve live data. No substitute or random values were generated. Check your internet connection, API status, and deployment logs. " + source_error)
elif original.empty:
    st.warning("The API returned no non-null observations for this country, indicator, and period.")
else:
    st.success(f"Actual coverage: {original.year.min()}–{original.year.max()} · {len(original)} non-null observations (not necessarily 70).")

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("Reset to source", disabled=original.empty):
        st.session_state["working_data"] = original.copy()
        st.session_state["scenario_edited"] = False
        st.session_state.pop("calc_results", None)
        st.rerun()
with c2:
    if st.session_state.get("scenario_edited", False):
        st.warning("User-modified scenario: calculations and exports use the editable working table, not untouched source values.")

if not original.empty:
    st.write("Editable working data — edit year/value cells, then use the validated table below.")
    edited = st.data_editor(st.session_state["working_data"], num_rows="dynamic", use_container_width=True,
                            key=f"editor_{data_key}", column_config={
                                "year": st.column_config.NumberColumn("Year", step=1, format="%d"),
                                "value": st.column_config.NumberColumn(f"Value ({units})", format="%.8g")
                            })
    try:
        valid_working = validate_frame(edited, indicator_name)
        source_compare = valid_working.equals(validate_frame(original, indicator_name))
        st.session_state["working_data"] = valid_working.copy()
        st.session_state["scenario_edited"] = not source_compare
        st.caption(f"Validated working table: {len(valid_working)} rows, {valid_working.year.nunique()} distinct years.")
        working_error = None
    except Exception as exc:
        working_error = str(exc)
        valid_working = pd.DataFrame(columns=["year", "value"])
        st.error("Working table validation failed: " + working_error)
else:
    valid_working = pd.DataFrame(columns=["year", "value"])
    working_error = "No source data available."

model = None
if not valid_working.empty and working_error is None:
    try:
        model = fit_poly(valid_working, degree)
    except Exception as exc:
        st.error("Model could not be fitted: " + str(exc))

st.subheader("2. Polynomial fit")
if model:
    st.latex(r"t = \text{year} - " + str(model["base"]))
    st.code(polynomial_equation(model))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (in-sample)", f"{model['r2']:.5f}")
    m2.metric("MAE (in-sample)", f"{model['mae']:.5g} {units}")
    m3.metric("Observation count", len(model["years"]))
    m4.metric("Historical mean", f"{model['mean']:.6g} {units}")
    st.caption(f"Fitted range: {model['min_year']}–{model['max_year']}. Metrics are in-sample and are not evidence of causal relationships.")
    residual_frame = pd.DataFrame({"year": model["years"], "actual": model["values"], "fitted": model["pred"], "residual (actual − fitted)": model["residuals"]})
    with st.expander("Residuals and fitted values"):
        st.dataframe(residual_frame, use_container_width=True)
    # Chronological holdout: last 20% only when enough data for a stable training fit
    if len(valid_working) >= max(degree + 5, 10):
        n_test = max(2, int(round(len(valid_working)*0.2)))
        train = valid_working.iloc[:-n_test]
        test = valid_working.iloc[-n_test:]
        if train.year.nunique() >= degree+1:
            try:
                hold = fit_poly(train, degree)
                hold_pred = poly_predict(hold, test.year.to_numpy())
                st.write(f"Chronological holdout MAE: **{mean_absolute_error(test.value, hold_pred):.6g} {units}** on {len(test)} later observations. This is separate from in-sample MAE.")
            except Exception:
                st.caption("Chronological holdout score unavailable for this dataset/degree.")
    else:
        st.caption("Chronological holdout score not shown: more observations are needed for this degree.")

    years_curve = np.linspace(model["min_year"], model["max_year"]+horizon, max(250, (model["max_year"]-model["min_year"]+horizon)*3))
    vals_curve = poly_predict(model, years_curve)
    sampled_years = np.arange(model["min_year"], model["max_year"]+1, sample_step)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(valid_working.year, valid_working.value, label="Observed / working data", zorder=3)
    hist_mask = years_curve <= model["max_year"]
    ax.plot(years_curve[hist_mask], vals_curve[hist_mask], label="Fitted curve (historical domain)")
    if horizon > 0:
        ax.plot(years_curve[~hist_mask], vals_curve[~hist_mask], linestyle="--", label="Polynomial extrapolation")
        ax.axvline(model["max_year"], color="gray", linestyle=":", label="Historical boundary")
    ax.scatter(sampled_years, poly_predict(model, sampled_years), marker="x", label=f"Model markers every {sample_step} year(s)")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({units})")
    ax.set_title(f"{country_name}: {indicator_name}")
    ax.grid(True, alpha=.25); ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    if horizon > 0:
        future_vals = poly_predict(model, np.array([model["max_year"]+horizon]))
        if not np.isfinite(future_vals).all() or abs(float(future_vals[0])) > max(1e8, abs(model["mean"])*1e5):
            st.warning("The polynomial extrapolation becomes extremely large or non-finite. Treat it as an implausible mathematical extrapolation, not a forecast; values are not clipped.")
    st.write("**Uncertainty and limitations:** the residuals and MAE summarize errors on the fitted observations. They are not 95% prediction intervals. Polynomial extrapolation can behave implausibly outside the observed range; the model is descriptive, not causal.")

    st.markdown("### Calculate a single year")
    single_year = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=max(1950, min(2100, model["max_year"])), step=1, key="single_year")
    if st.button("Calculate Value"):
        val = float(poly_predict(model, np.array([single_year]))[0])
        st.session_state["calc_results"] = {"years": [int(single_year)], "values": [val], "signature": (country_code, indicator_code, degree, tuple(valid_working.year), tuple(valid_working.value))}
    sig = (country_code, indicator_code, degree, tuple(valid_working.year), tuple(valid_working.value))
    results = st.session_state.get("calc_results")
    if results and results.get("signature") == sig:
        yr, val = results["years"][0], results["values"][0]
        status = "within fitted historical range" if model["min_year"] <= yr <= model["max_year"] else "outside fitted historical range (extrapolation)"
        st.write(f"**Year {yr}: {val:.10g} {units}.** This is a fitted model value, not an observed value; {status}.")
    elif results:
        st.session_state.pop("calc_results", None)
        st.caption("Previous calculated results were invalidated because the model/data changed.")

    st.markdown("### Batch year calculations")
    batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
    batch_years = []
    invalid_batch = []
    if batch_mode == "Year range":
        ba, bb = st.columns(2)
        with ba: first_year = st.number_input("Start year", 1950, 2100, max(1950, min(2100, model["min_year"])), key="batch_start")
        with bb: last_year = st.number_input("End year", 1950, 2100, max(1950, min(2100, model["max_year"])), key="batch_end")
        step = st.slider("Batch range step", 1, 10, 1)
        if first_year <= last_year:
            batch_years = list(range(int(first_year), int(last_year)+1, int(step)))
        else:
            invalid_batch.append("Start year must be less than or equal to end year.")
    else:
        custom_years = st.text_input("Comma-separated years", "2000, 2010, 2020")
        for part in custom_years.split(","):
            part = part.strip()
            if not part: continue
            try:
                number = float(part)
                if not number.is_integer() or not 1950 <= number <= 2100:
                    invalid_batch.append(f"Invalid year: {part!r} (enter an integer from 1950 to 2100).")
                else: batch_years.append(int(number))
            except ValueError:
                invalid_batch.append(f"Invalid year: {part!r} (not a number).")
        if len(set(batch_years)) != len(batch_years):
            invalid_batch.append("Duplicate requested years were found.")
        batch_years = list(dict.fromkeys(batch_years))
    if invalid_batch:
        for msg in invalid_batch: st.error(msg)
    if st.button("Calculate batch values"):
        if not batch_years or invalid_batch:
            st.error("Fix batch input errors before calculating.")
        else:
            vals = poly_predict(model, np.asarray(batch_years))
            batch_df = pd.DataFrame({"year": batch_years, "estimated value": vals,
                                     "classification": ["within fitted range" if model["min_year"] <= yy <= model["max_year"] else "outside fitted range" for yy in batch_years]})
            st.session_state["batch_results"] = {"frame": batch_df, "signature": sig}
    batch_result = st.session_state.get("batch_results")
    if batch_result and batch_result.get("signature") == sig:
        st.dataframe(batch_result["frame"], use_container_width=True)
        if len(batch_result["frame"]):
            bfig, bax = plt.subplots(figsize=(8, 3))
            bax.plot(years_curve, vals_curve, label="Fitted curve")
            bax.scatter(batch_result["frame"]["year"], batch_result["frame"]["estimated value"], label="Batch estimates", marker="D")
            bax.set_xlabel("Year"); bax.set_ylabel(f"Estimate ({units})"); bax.grid(True, alpha=.25); bax.legend()
            st.pyplot(bfig, use_container_width=True); plt.close(bfig)

# Same-indicator country comparison
st.subheader("3. Country comparison")
if compare_names:
    compare_frames = []
    compare_errors = []
    for cname in compare_names:
        try:
            cf = fetch_world_bank(COUNTRIES[cname], indicator_code, start_year, end_year)
            cf["country"] = cname
            compare_frames.append(cf)
        except Exception as exc:
            compare_errors.append(f"{cname}: {exc}")
    if compare_frames:
        figc, axc = plt.subplots(figsize=(10, 4.2))
        for cf in compare_frames:
            axc.plot(cf.year, cf.value, marker=".", linewidth=1.5, label=cf.country)
        if not valid_working.empty:
            axc.plot(valid_working.year, valid_working.value, marker=".", linewidth=1.5, label=country_name)
        axc.set_xlabel("Year"); axc.set_ylabel(f"{indicator_name} ({units})"); axc.set_title("Same indicator; coverage can differ by country")
        axc.grid(True, alpha=.25); axc.legend()
        st.pyplot(figc, use_container_width=True); plt.close(figc)
        coverage = [{"country": c, "observations": len(f), "first year": int(f.year.min()) if len(f) else None, "last year": int(f.year.max()) if len(f) else None} for c, f in [(country_name, valid_working)] + [(f.country.iloc[0], f.drop(columns="country")) for f in compare_frames]]
        st.dataframe(pd.DataFrame(coverage), use_container_width=True)
        st.caption("Gaps and different date ranges reflect available reported observations; missing values are not filled with fabricated data.")
    for err in compare_errors: st.warning(err)
else:
    st.caption("Choose one or more comparison countries in the sidebar to plot the same indicator.")

# Function and calculus lab
st.subheader("4. Function lab and calculus analysis")
context = st.selectbox("Context", list(CONTEXTS))
_, default_xname, default_xunit, default_yname, default_yunit, default_desc = CONTEXTS[context]
context_desc = st.text_input("Context description", default_desc)
cc1, cc2, cc3, cc4 = st.columns(4)
with cc1: xname = st.text_input("Independent variable", default_xname)
with cc2: xunit = st.text_input("Independent units", default_xunit)
with cc3: yname = st.text_input("Dependent variable", default_yname)
with cc4: yunit = st.text_input("Dependent units", default_yunit)

function_source = st.radio("Function source", ["Curve of best fit", "Suggested function", "Build with parameters", "Enter my own function"], horizontal=True)
suggestions = {
    "Business": "a*x^2 + b*x + c", "Engineering": "a*sin(b*x) + c", "Medical": "a*exp(-b*x) + c",
    "Sports science": "a*x^2 + b*x + c", "Biology": "a*exp(b*x)", "Bioengineering": "a*(1-exp(-b*x))",
    "Economic forecasting": "a*x + b", "Weather": "a*cos(b*x) + c", "Custom": "x^2 - 4*x + 3",
}
if function_source == "Curve of best fit":
    if model:
        # use x as calendar year; model uses centered t
        expr = sum(sp.Float(float(c), 14) * (x-model["base"])**(model["degree"]-i) for i, c in enumerate(model["coeff"]))
        expr_error = None
        st.caption("Using the fitted polynomial with x as calendar year and centered time internally.")
    else:
        expr, expr_error = None, "Fit a valid data model first."
elif function_source == "Suggested function":
    selected_suggestion = st.selectbox("Suggested form", list(suggestions.values()), index=list(suggestions.values()).index(suggestions[context]))
    params = st.columns(3)
    with params[0]: a = st.number_input("a", value=1.0, key="suggest_a")
    with params[1]: b = st.number_input("b", value=0.2, key="suggest_b")
    with params[2]: c = st.number_input("c", value=0.0, key="suggest_c")
    expr_text = re.sub(r"\\ba\\b", f"({a})", selected_suggestion)
    expr_text = re.sub(r"\\bb\\b", f"({b})", expr_text)
    expr_text = re.sub(r"\\bc\\b", f"({c})", expr_text)
    try: expr, expr_error = safe_parse(expr_text), None
    except Exception as exc: expr, expr_error = None, str(exc)
elif function_source == "Build with parameters":
    form = st.selectbox("Function family", ["Polynomial (quadratic)", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
    pcols = st.columns(3)
    with pcols[0]: a = st.number_input("Parameter a", value=1.0, key="build_a")
    with pcols[1]: b = st.number_input("Parameter b", value=1.0, key="build_b")
    with pcols[2]: c = st.number_input("Parameter c", value=0.0, key="build_c")
    forms = {"Polynomial (quadratic)": f"({a})*x^2+({b})*x+({c})",
             "Rational": f"({a})/(x-({b}))+({c})",
             "Exponential": f"({a})*exp(({b})*x)+({c})",
             "Logarithmic": f"({a})*log(x-({b}))+({c})",
             "Trigonometric": f"({a})*sin(({b})*x)+({c})"}
    try: expr, expr_error = safe_parse(forms[form]), None
    except Exception as exc: expr, expr_error = None, str(exc)
else:
    custom_expr = st.text_input("Function f(x)", "x^3 - 3*x")
    st.caption("Supported notation: +, -, *, /, ^, parentheses, sin, cos, tan, exp, log, sqrt, abs, pi, E. Example: x^3 - 3*x.")
    try: expr, expr_error = safe_parse(custom_expr), None
    except Exception as exc: expr, expr_error = None, str(exc)

if expr_error:
    st.error("Function unavailable: " + expr_error)
if expr is not None:
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    st.markdown("**Function:**")
    st.latex(sp.latex(sp.Eq(sp.Function("f")(x), expr)))
    if course == "Calculus":
        st.markdown("**First derivative:**"); st.latex(sp.latex(sp.Eq(sp.Function("f'")(x), d1)))
        st.markdown("**Second derivative:**"); st.latex(sp.latex(sp.Eq(sp.Function("f''")(x), d2)))
    elif course == "Precalculus":
        st.caption("Precalculus focus: degree and coefficients shape the graph; turning points and concavity can be explored numerically below.")
        st.latex(sp.latex(d1)); st.latex(sp.latex(d2))
    else:
        st.caption("Honors Algebra 2 focus: domain, range, end behavior, graph trends, and average rates of change.")

    with st.expander("Function interval and moving point"):
        default_lo, default_hi = (-5.0, 5.0)
        if model and function_source == "Curve of best fit":
            default_lo, default_hi = float(model["min_year"]), float(min(2100, model["max_year"]+horizon))
            if default_hi <= default_lo: default_hi = default_lo+1
        lo = st.number_input("Analysis interval start", value=float(default_lo), key="analysis_lo")
        hi = st.number_input("Analysis interval end", value=float(default_hi), key="analysis_hi")
    if hi <= lo:
        st.error("Analysis interval end must be greater than start.")
    else:
        point = st.slider("Current input value", min_value=float(lo), max_value=float(hi), value=float((lo+hi)/2), step=max((hi-lo)/1000, 0.001), key="moving_point")
        acol1, acol2 = st.columns(2)
        with acol1:
            sec_a = st.slider("Average-rate interval start", min_value=float(lo), max_value=float(hi), value=float(lo), key="secant_a")
        with acol2:
            sec_b = st.slider("Average-rate interval end", min_value=float(lo), max_value=float(hi), value=float(hi), key="secant_b")
        show_f = st.checkbox("Show function", True)
        show_d1 = st.checkbox("Show first derivative", course == "Calculus")
        show_d2 = st.checkbox("Show second derivative", False)
        try:
            d1, d2, scalar, f, f1, f2, avg, ya, yb, yp, slope = calc_features(expr, (lo, hi), point, sec_a, sec_b)
            curvature = scalar(f2, point)
            xgrid = np.linspace(lo, hi, 1600)
            ygrid = numeric_function(expr, xgrid)
            # Avoid connecting across invalid values / discontinuities
            ygrid[~np.isfinite(ygrid)] = np.nan
            figf, axf = plt.subplots(figsize=(10, 5))
            if show_f: axf.plot(xgrid, ygrid, label="f(x)")
            if show_d1: axf.plot(xgrid, numeric_function(d1, xgrid), label="f′(x)")
            if show_d2: axf.plot(xgrid, numeric_function(d2, xgrid), label="f″(x)")
            if yp is not None:
                axf.scatter([point], [yp], label="Moving point", zorder=5)
                if slope is not None:
                    tangent_x = np.linspace(max(lo, point-(hi-lo)*.15), min(hi, point+(hi-lo)*.15), 100)
                    axf.plot(tangent_x, yp+slope*(tangent_x-point), linestyle="--", label="Tangent line")
            if avg is not None:
                secx = np.linspace(sec_a, sec_b, 100)
                axf.plot(secx, ya+avg*(secx-sec_a), linestyle=":", label="Secant line")
                axf.scatter([sec_a, sec_b], [ya, yb], marker="o")
            axf.axvline(point, color="gray", alpha=.25)
            axf.set_xlabel(f"{xname} ({xunit})"); axf.set_ylabel(f"{yname} ({yunit})")
            axf.set_title(context_desc); axf.grid(True, alpha=.25); axf.legend()
            st.pyplot(figf, use_container_width=True); plt.close(figf)
            st.write(f"**f({point:.6g})** = {yp:.8g} {yunit}" if yp is not None else f"f({point:.6g}) is undefined or non-real.")
            st.write(f"**Instantaneous rate:** {slope:.8g} {yunit}/{xunit}" if slope is not None else "Instantaneous rate is undefined at this point.")
            st.write(f"**Average rate from {sec_a:g} to {sec_b:g}:** {avg:.8g} {yunit}/{xunit}" if avg is not None else "Average rate unavailable: endpoint is undefined or interval length is zero.")
            st.write(context_meaning(context, yname, yunit, xname, xunit, point, yp, slope, curvature))
            st.write(f"**Domain note:** {sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)}")
            if course in ("Precalculus", "Calculus"):
                # Numeric sign changes and near-zero checks; explicitly label approximations
                def roots_in_interval(derivative):
                    try:
                        candidates = sp.calculus.util.function_range(derivative, x, sp.Interval(lo, hi))
                    except Exception:
                        pass
                    fn = sp.lambdify(x, derivative, "numpy")
                    grid = np.linspace(lo, hi, 2500)
                    with np.errstate(all="ignore"):
                        vals = np.asarray(fn(grid), dtype=float)
                    roots = []
                    for i in range(len(grid)-1):
                        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1] < 0:
                            left, right = grid[i], grid[i+1]
                            for _ in range(35):
                                mid = (left+right)/2
                                vm = float(fn(mid))
                                if not math.isfinite(vm): break
                                if float(fn(left))*vm <= 0: right = mid
                                else: left = mid
                            roots.append((left+right)/2)
                    return roots
                critical = roots_in_interval(d1)
                st.write("**Approximate critical points (zeros of f′ found by sign changes):**", [round(v, 6) for v in critical] or "None detected numerically.")
                if course == "Calculus":
                    inc, dec, extrema, infl = [], [], [], []
                    breaks = [lo] + critical + [hi]
                    for aa, bb in zip(breaks[:-1], breaks[1:]):
                        mid = (aa+bb)/2
                        sv = scalar(sp.lambdify(x, d1, "numpy"), mid)
                        if sv is not None: (inc if sv > 0 else dec if sv < 0 else []).append((aa,bb))
                    for r in critical:
                        leftv, rightv = scalar(f1, r-1e-5), scalar(f1, r+1e-5)
                        if leftv is not None and rightv is not None:
                            if leftv > 0 and rightv < 0: extrema.append((r, "local maximum"))
                            elif leftv < 0 and rightv > 0: extrema.append((r, "local minimum"))
                    inflection_candidates = roots_in_interval(d2)
                    for r in inflection_candidates:
                        lv, rv = scalar(f2, r-1e-5), scalar(f2, r+1e-5)
                        if lv is not None and rv is not None and lv*rv < 0:
                            infl.append(r)
                    st.write("**Increasing intervals (numerical):**", inc or "None detected.")
                    st.write("**Decreasing intervals (numerical):**", dec or "None detected.")
                    st.write("**Local extrema (first-derivative sign test):**", [(round(r,6), kind) for r,kind in extrema] or "None confirmed.")
                    st.write("**Inflection points (f″ sign change verified numerically):**", [round(r,6) for r in infl] or "None confirmed.")
                    # Closed interval extrema using dense grid plus endpoints; numerical only
                    grid = np.linspace(lo, hi, 10000)
                    vals = numeric_function(expr, grid)
                    ok = np.isfinite(vals)
                    if ok.any():
                        idx = np.where(ok)[0]
                        imin, imax = idx[np.argmin(vals[ok])], idx[np.argmax(vals[ok])]
                        st.write(f"**Approximate absolute extrema on sampled interval:** min ≈ ({grid[imin]:.6g}, {vals[imin]:.6g}), max ≈ ({grid[imax]:.6g}, {vals[imax]:.6g}). Numerical grid only; discontinuities or narrow features may be missed.")
                    st.caption("Root and extrema results are numerical approximations, not exact proofs. Zeros of f′ or f″ are not automatically extrema or inflection points; sign-change checks are used where possible.")
                st.write("**Concavity:** determined by the sign of f″; numerical analysis can miss roots or discontinuities.")
            if course == "Honors Algebra 2":
                st.write("**Average rate of change** is the secant-line slope. **End behavior** can be examined by comparing graph values near the interval endpoints; results outside the displayed interval are not automatically justified.")
        except Exception as exc:
            st.error("Could not evaluate/plot this function on the selected interval. Check its domain and parameters. Details: " + str(exc))

# Animation / contextual visual
st.subheader("5. Visual simulation")
visual = st.selectbox("Visual simulation", ["Walking person with prosthetic leg (conceptual)", "Moving object / projectile", "Tank or concentration indicator", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
st.caption("The visual is a conceptual teaching aid. Its scale is normalized to the current analysis interval and does not represent a real person, medical device, projectile, patient, or forecast.")
if expr is not None and 'lo' in locals() and 'hi' in locals() and hi > lo:
    sim_x = st.slider("Simulation input", float(lo), float(hi), float(point if 'point' in locals() else (lo+hi)/2), key="sim_point")
    sim_val = None
    try: sim_val = float(numeric_function(expr, np.array([sim_x]))[0])
    except Exception: pass
    if sim_val is None or not math.isfinite(sim_val):
        st.warning("Simulation is paused at an undefined/non-finite function value.")
    else:
        # normalized visual scale
        all_y = numeric_function(expr, np.linspace(lo, hi, 300))
        finite = all_y[np.isfinite(all_y)]
        if len(finite):
            ymin, ymax = float(np.min(finite)), float(np.max(finite))
            norm = 0.5 if ymax == ymin else float(np.clip((sim_val-ymin)/(ymax-ymin), 0, 1))
            if visual.startswith("Walking"):
                st.write(f"**Conceptual gait phase:** {norm*100:.0f}% of the normalized vertical scale. This is not a prosthesis biomechanics model.")
                st.progress(norm)
                st.markdown("🚶  🦿  →  The marker responds to the selected function's normalized value; no real gait or clinical inference is represented.")
            elif visual.startswith("Moving"):
                st.write(f"Object vertical position (normalized): {norm:.3f}")
                st.markdown("⬅️ " + "　" * int(norm*12) + "🔵")
            elif visual.startswith("Tank"):
                st.write(f"Normalized tank/concentration indicator: {norm:.1%}")
                st.progress(norm)
            elif visual.startswith("Population"):
                st.write(f"Normalized population/cell-colony indicator: {norm:.1%}")
                st.progress(norm)
            elif visual.startswith("Thermometer"):
                st.write(f"Normalized temperature indicator: {norm:.1%}")
                st.progress(norm)
            else:
                st.write(f"Normalized business/economic indicator: {norm:.1%}")
                st.progress(norm)
        st.caption("Use the Simulation input slider to move the visual. Streamlit reruns the app on each control change; playback is manual through the slider to avoid brittle browser-side timers.")

# Exports
st.subheader("6. Export report")
scenario_status = "User-modified scenario" if st.session_state.get("scenario_edited", False) else "Source data (unmodified)"
summary_rows = [
    ("App", APP_TITLE), ("Country", country_name), ("Country code", country_code),
    ("Indicator", indicator_name), ("Indicator code", indicator_code), ("Definition", definition),
    ("Units", units), ("Source URL", source_url), ("Retrieved date", retrieved_at),
    ("Requested range", f"{start_year}-{end_year}"), ("Data status", scenario_status),
    ("Mathematics pathway", course), ("Polynomial degree", degree),
]
if model:
    summary_rows += [("Equation", polynomial_equation(model)), ("Base year", model["base"]),
                     ("R-squared (in-sample)", model["r2"]), ("MAE (in-sample)", model["mae"]),
                     ("Historical range", f"{model['min_year']}-{model['max_year']}"), ("Observation count", len(model["years"]))]
summary_df = pd.DataFrame(summary_rows, columns=["field", "value"])
working_export = valid_working.copy() if not valid_working.empty else pd.DataFrame(columns=["year","value"])
single_export = pd.DataFrame()
if model and st.session_state.get("calc_results") and st.session_state["calc_results"].get("signature") == (country_code, indicator_code, degree, tuple(valid_working.year), tuple(valid_working.value)):
    rr = st.session_state["calc_results"]
    single_export = pd.DataFrame({"year": rr["years"], "estimated value": rr["values"]})
batch_export = pd.DataFrame()
br = st.session_state.get("batch_results")
if br and model and br.get("signature") == (country_code, indicator_code, degree, tuple(valid_working.year), tuple(valid_working.value)):
    batch_export = br["frame"]
combined_results = pd.concat([single_export.assign(result_type="single"), batch_export.assign(result_type="batch")], ignore_index=True, sort=False)
csv_data = working_export.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", csv_data, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
try:
    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        working_export.to_excel(writer, sheet_name="Working Data", index=False)
        combined_results.to_excel(writer, sheet_name="Calculated Years", index=False)
        for sheet in ["Summary", "Working Data", "Calculated Years"]:
            ws = writer.sheets[sheet]
            ws.set_column(0, 0, 28); ws.set_column(1, 1, 90)
    st.download_button("Download Excel workbook (.xlsx)", xlsx_buffer.getvalue(), file_name="hispanic_heritage_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as exc:
    st.error("Excel export unavailable: " + str(exc))
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=landscape(letter), rightMargin=0.4*inch, leftMargin=0.4*inch, topMargin=0.4*inch, bottomMargin=0.4*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(APP_TITLE, styles["Title"]), Paragraph(f"{country_name} · {indicator_name}", styles["Heading2"]),
             Paragraph(f"Retrieved {retrieved_at} · {scenario_status} · Units: {units}", styles["BodyText"]), Spacer(1, 8)]
    story.append(Paragraph(definition, styles["BodyText"]))
    if model:
        story += [Spacer(1, 6), Paragraph(polynomial_equation(model), styles["BodyText"]),
                  Paragraph(f"In-sample R²={model['r2']:.6g}; in-sample MAE={model['mae']:.6g}; observations={len(model['years'])}; range={model['min_year']}–{model['max_year']}. Metrics are not prediction intervals.", styles["BodyText"])]
        pdf_fig, pdf_ax = plt.subplots(figsize=(8, 2.5))
        pdf_ax.scatter(valid_working.year, valid_working.value, label="Observed")
        pdf_ax.plot(years_curve, vals_curve, label="Fit/extrapolation")
        pdf_ax.axvline(model["max_year"], color="gray", linestyle=":")
        pdf_ax.set_xlabel("Year"); pdf_ax.set_ylabel(units); pdf_ax.legend(fontsize=7); pdf_ax.grid(True, alpha=.2)
        image_buf = io.BytesIO(); pdf_fig.savefig(image_buf, format="png", dpi=130, bbox_inches="tight"); plt.close(pdf_fig); image_buf.seek(0)
        story += [Spacer(1, 8), RLImage(image_buf, width=7.5*inch, height=2.4*inch)]
    story += [Spacer(1, 8), Paragraph("Data preview", styles["Heading3"])]
    preview = working_export.head(18)
    table_data = [list(preview.columns)] + preview.astype(str).values.tolist()
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(tbl)
    if len(combined_results):
        story += [Spacer(1, 8), Paragraph("Calculated results", styles["Heading3"])]
        tdata = [list(combined_results.columns)] + combined_results.head(30).astype(str).values.tolist()
        rt = Table(tdata, repeatRows=1)
        rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(rt)
    story += [Spacer(1, 8), Paragraph("Limitations: World Bank observations can be missing or revised. Polynomial fits are descriptive, sensitive to degree, and may extrapolate implausibly. MAE and R² are in-sample unless explicitly labeled holdout; no validated prediction interval is provided. Do not infer historical causes from this curve alone.", styles["BodyText"])]
    doc.build(story)
    st.download_button("Download PDF report", pdf_buf.getvalue(), file_name="hispanic_heritage_report.pdf", mime="application/pdf")
except Exception as exc:
    st.error("PDF export unavailable: " + str(exc))

if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly report")
    st.write(f"{country_name} — {indicator_name} ({units})")
    st.write(definition)
    st.write(f"Source: {source_url}")
    st.write(f"Retrieved: {retrieved_at} · Status: {scenario_status}")
    if model:
        st.write(polynomial_equation(model))
        st.write(f"In-sample R² = {model['r2']:.6g}; in-sample MAE = {model['mae']:.6g}; n = {len(model['years'])}.")
    st.dataframe(working_export.head(25), use_container_width=True)

st.markdown("---")
st.caption("Educational use only. The World Bank is the source for historical indicators; user-entered functions and scenarios are not official forecasts. Check sources.md for API and indicator references.")
