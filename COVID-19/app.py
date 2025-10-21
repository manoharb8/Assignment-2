import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

# Caching data for 1 hour
@st.cache_data(ttl=3600)
def load_countries_summary():
    url = "https://disease.sh/v3/covid-19/countries"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())

@st.cache_data(ttl=3600)
def load_global_historical(lastdays="all"):
    url = f"https://disease.sh/v3/covid-19/historical?lastdays={lastdays}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=3600)
def load_country_historical(country, lastdays="all"):
    # Use the country-specific historical endpoint
    url = f"https://disease.sh/v3/covid-19/historical/{country}?lastdays={lastdays}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

# Load data
with st.spinner("Loading summary data..."):
    try:
        df = load_countries_summary()
    except Exception as e:
        st.error(f"Failed to load summary data: {e}")
        st.stop()

st.sidebar.title("COVID-19 Dashboard")
st.sidebar.markdown("Select options to filter the data and update visualizations.")

# Sidebar controls
country_list = sorted(df['country'].dropna().unique())
country = st.sidebar.selectbox("Select country", ["Global"] + country_list, index=0)

metric_option = st.sidebar.selectbox("Metric for time series", ["cases", "deaths", "recovered", "tests"])
ma_window = st.sidebar.slider("Moving average window (days)", 1, 30, 7)

# Top-level KPIs
st.title("COVID-19 Interactive Dashboard")
st.markdown("Data source: disease.sh (aggregated Johns Hopkins / other public sources) — used for demonstration and assignment EDA.")

if country == "Global":
    # Aggregate global values
    total_cases = int(df['cases'].sum())
    total_deaths = int(df['deaths'].sum())
    total_recovered = int(df['recovered'].sum())
    total_tests = int(df['tests'].dropna().sum()) if 'tests' in df.columns else None
else:
    row = df[df['country'] == country].iloc[0]
    total_cases = int(row['cases'])
    total_deaths = int(row['deaths'])
    total_recovered = int(row['recovered'])
    total_tests = int(row['tests']) if 'tests' in row and not pd.isna(row['tests']) else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", f"{total_cases:,}")
col2.metric("Total Deaths", f"{total_deaths:,}")
col3.metric("Total Recovered", f"{total_recovered:,}")
col4.metric("Total Tests", f"{total_tests:,}" if total_tests is not None else "N/A")

st.markdown("---")

# Correlation heatmap across countries (numeric features)
st.header("Cross-country statistics & correlation")
numeric_cols = ['cases', 'deaths', 'recovered', 'active', 'population']
corr_df = df[numeric_cols].fillna(0)
corr = corr_df.corr()

fig_corr = px.imshow(corr,
                     x=corr.columns,
                     y=corr.columns,
                     text_auto=True,
                     title="Correlation matrix (cases, deaths, recovered, active, population)")
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# Scatter: Cases vs Deaths by country (interactive)
st.header("Cases vs Deaths (by country)")
fig_scatter = px.scatter(df,
                         x="cases",
                         y="deaths",
                         size="population",
                         color="continent" if 'continent' in df.columns else None,
                         hover_name="country",
                         log_x=True,
                         title="COVID-19 Cases vs Deaths by Country (log scale on X)")
fig_scatter.update_layout(autosize=True)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# World Choropleth map of cases
st.header("Global map: Cases by country")
if 'countryInfo' in df.columns and isinstance(df.loc[0, 'countryInfo'], dict):
    # Extract ISO2 or lat/lon where possible
    df['iso2'] = df['countryInfo'].apply(lambda x: x.get('iso2') if isinstance(x, dict) else None)
else:
    df['iso2'] = None

fig_map = px.choropleth(df,
                        locations="iso2",
                        color="cases",
                        hover_name="country",
                        title="Choropleth: Total Cases by Country",
                        color_continuous_scale="Reds")
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# Time series for selected country (historical)
st.header(f"Time series: {country}")

# Get historical data:
if country == "Global":
    # If Global selected, aggregate country historical data (slower). We'll fetch global historical and sum per day.
    with st.spinner("Loading historical data for all countries (may take a few seconds)..."):
        try:
            hist_json = load_global_historical(lastdays="all")
            # hist_json is a list of country objects; build a global time series for metric_option
            # We'll sum the 'timeline' -> 'cases'/'deaths'/'recovered' across countries for each date.
            # Build a DataFrame of dates x countries then sum across columns
            df_hist_list = []
            for c in hist_json:
                timeline = c.get('timeline') or c.get('timeline', {})
                if not timeline:
                    continue
                metric_series = timeline.get(metric_option, {})
                if metric_series:
                    s = pd.Series(metric_series)
                    s.name = c.get('country', 'unknown')
                    df_hist_list.append(s)
            if len(df_hist_list) == 0:
                st.info("No historical data available to show.")
                ts = pd.DataFrame()
            else:
                df_hist = pd.concat(df_hist_list, axis=1).fillna(0)
                ts = df_hist.sum(axis=1).rename('value')
                ts.index = pd.to_datetime(ts.index, format="%m/%d/%y")
                ts = ts.sort_index()
        except Exception as e:
            st.error(f"Failed to load global historical data: {e}")
            ts = pd.Series(dtype=float)
else:
    with st.spinner(f"Loading historical data for {country}..."):
        try:
            hist = load_country_historical(country, lastdays="all")
            # hist has structure {'country':..., 'timeline': {'cases':{date:val,...}, 'deaths':{...}, ...}}
            timeline = hist.get('timeline', {})
            metric_series = timeline.get(metric_option, {})
            ts = pd.Series(metric_series).rename(metric_option)
            ts.index = pd.to_datetime(ts.index, format="%m/%d/%y")
            ts = ts.sort_index()
        except Exception as e:
            st.error(f"Failed to load historical data for {country}: {e}")
            ts = pd.Series(dtype=float)

if ts.empty:
    st.info("No time series data available for the selected options.")
else:
    df_ts = pd.DataFrame({'date': ts.index, 'value': ts.values})
    df_ts['new'] = df_ts['value'].diff().fillna(0)
    df_ts['ma'] = df_ts['new'].rolling(window=ma_window, min_periods=1).mean()

    # Plotly time series with range selector
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df_ts['date'], y=df_ts['value'],
                                mode='lines', name=f"Cumulative {metric_option.title()}"))
    fig_ts.add_trace(go.Bar(x=df_ts['date'], y=df_ts['new'], name=f"Daily new {metric_option}"))
    fig_ts.add_trace(go.Scatter(x=df_ts['date'], y=df_ts['ma'], mode='lines',
                                name=f"{ma_window}-day MA (new)", line=dict(width=3, dash='dash')))
    fig_ts.update_layout(title=f"{metric_option.title()} over time ({country})",
                         xaxis=dict(rangeselector=dict(buttons=list([
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=3, label="3m", step="month", stepmode="backward"),
                             dict(count=6, label="6m", step="month", stepmode="backward"),
                             dict(step="all")
                         ])),
                                     rangeslider=dict(visible=True),
                                     type="date"))
    st.plotly_chart(fig_ts, use_container_width=True)

    # Time series decomposition on daily new series (if long enough)
    if len(df_ts) >= 30:
        try:
            series_for_decomp = df_ts.set_index('date')['new'].astype(float).asfreq('D').fillna(0)
            result = seasonal_decompose(series_for_decomp, model='additive', period=7, extrapolate_trend='freq')
            st.subheader("Time series decomposition (daily new cases)")
            # Display decomposition as a small matplotlib figure
            fig_decomp, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(result.observed); ax[0].set_ylabel('observed')
            ax[1].plot(result.trend); ax[1].set_ylabel('trend')
            ax[2].plot(result.seasonal); ax[2].set_ylabel('seasonal')
            ax[3].plot(result.resid); ax[3].set_ylabel('resid')
            fig_decomp.tight_layout()
            st.pyplot(fig_decomp)
        except Exception as e:
            st.info("Decomposition not available: " + str(e))

st.markdown("---")
st.header("Download & Notes")
st.write("You can download the country-level summary CSV used in this dashboard.")
csv = df.to_csv(index=False)
st.download_button("Download summary CSV", data=csv, file_name="covid_summary.csv", mime="text/csv")

st.caption("Notes: This dashboard uses the disease.sh API for demonstration. For academic or production work, cross-check data with official repositories (Johns Hopkins CSSE, Our World in Data).")
