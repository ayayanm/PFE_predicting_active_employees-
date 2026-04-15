import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CMR Workforce Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
workforce_df = pd.read_csv("workforce_df.csv")

# Replace these with your real final forecast values
forecast_df = pd.DataFrame({
    "year": [2026, 2027, 2028, 2029, 2030],
    "naive_forecast": [616380, 616380, 616380, 616380, 616380],
    "linear_fe_safe_forecast": [625463, 633286, 641514, 649509, 657552],
    "linear_trend_forecast": [679676, 690170, 700664, 711158, 721652]
})

results_df = pd.DataFrame({
    "Model": [
        "Naive",
        "Linear Regression",
        "Ridge",
        "Lasso",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "ARIMA",
        "Exponential Smoothing",
        "Exponential Smoothing Damped",
        "LSTM",
        "LSTM FE",
        "LSTM Tuned",
        "Linear FE Safe",
        "Linear Trend"
    ],
    "MAPE": [
        3.20,
        4.11,
        4.11,
        4.26,
        8.06,
        7.28,
        8.89,
        10.01,
        12.13,
        12.65,
        12.23,
        4.08,
        4.24,
        5.77,
        4.05,
        4.11
    ]
}).sort_values("MAPE")

# Derived indicators
workforce_df["net_change"] = workforce_df["entries"] - workforce_df["exits"]
workforce_df["growth_rate"] = workforce_df["active_employees"].pct_change() * 100

latest_year = int(workforce_df["year"].max())
latest_workforce = int(workforce_df["active_employees"].iloc[-1])
avg_growth = workforce_df["growth_rate"].dropna().mean()
best_model = results_df.iloc[0]["Model"]
best_mape = results_df.iloc[0]["MAPE"]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Historical Analysis", "Model Comparison", "Forecast Scenarios", "Project Notes"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

min_year = int(workforce_df["year"].min())
max_year = int(workforce_df["year"].max())

year_range = st.sidebar.slider(
    "Select year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

filtered_df = workforce_df[
    (workforce_df["year"] >= year_range[0]) &
    (workforce_df["year"] <= year_range[1])
].copy()

# -----------------------------
# HELPERS
# -----------------------------
def plot_lines(df, x_col, y_cols, title, ylabel):
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in y_cols:
        ax.plot(df[x_col], df[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def metric_card(label, value):
    st.metric(label, value)

# -----------------------------
# DASHBOARD
# -----------------------------
if page == "Dashboard":
    st.title("📊 CMR Workforce Forecast Dashboard")
    st.markdown("""
    This interactive dashboard presents the historical evolution of public sector employees in Morocco,
    the comparison of forecasting models, and future workforce scenarios.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Latest Year", latest_year)
    with c2:
        metric_card("Active Employees (2025)", f"{latest_workforce:,}")
    with c3:
        metric_card("Average Growth Rate", f"{avg_growth:.2f}%")
    with c4:
        metric_card("Best Model", f"{best_model} ({best_mape:.2f}%)")

    st.markdown("### Workforce Evolution")
    selected_main = st.multiselect(
        "Select indicators",
        ["active_employees", "entries", "exits", "net_change"],
        default=["active_employees", "entries", "exits"]
    )
    if selected_main:
        plot_lines(
            filtered_df,
            "year",
            selected_main,
            "Historical Workforce Evolution",
            "Count"
        )

    st.markdown("### Quick Insight")
    with st.expander("Read interpretation"):
        st.write("""
        - The workforce shows a long-term upward evolution.
        - Entries and exits explain yearly changes in active employees.
        - The year 2021 appears as a structural shock in the series.
        - Simple models performed better than many complex ones due to the dataset size and strong inertia.
        """)

# -----------------------------
# HISTORICAL ANALYSIS
# -----------------------------
elif page == "Historical Analysis":
    st.title("📈 Historical Analysis")

    tab1, tab2, tab3 = st.tabs(["Workforce", "Dynamics", "Growth"])

    with tab1:
        plot_lines(
            filtered_df,
            "year",
            ["active_employees"],
            "Active Employees Over Time",
            "Employees"
        )

    with tab2:
        plot_lines(
            filtered_df,
            "year",
            ["entries", "exits", "net_change"],
            "Entries, Exits, and Net Change",
            "Count"
        )

    with tab3:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(filtered_df["year"], filtered_df["growth_rate"], marker="o")
        ax.set_title("Growth Rate of Workforce")
        ax.set_xlabel("Year")
        ax.set_ylabel("Growth Rate (%)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.dataframe(filtered_df, use_container_width=True)

# -----------------------------
# MODEL COMPARISON
# -----------------------------
elif page == "Model Comparison":
    st.title("🤖 Model Comparison")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Performance Table")
        st.dataframe(results_df, use_container_width=True)

    with c2:
        st.subheader("MAPE Comparison")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(results_df["Model"], results_df["MAPE"])
        ax.set_xlabel("MAPE (%)")
        ax.set_title("Lower is Better")
        ax.invert_yaxis()
        st.pyplot(fig)

    with st.expander("Interpretation"):
        st.write("""
        - The Naive model is the best baseline.
        - Linear models remain competitive and interpretable.
        - Tree-based and boosting models performed worse.
        - LSTM provided an interesting advanced benchmark, but did not outperform the simplest approaches.
        """)

# -----------------------------
# FORECAST SCENARIOS
# -----------------------------
elif page == "Forecast Scenarios":
    st.title("🔮 Forecast Scenarios")

    scenario_options = {
        "Naive": "naive_forecast",
        "Linear FE Safe": "linear_fe_safe_forecast",
        "Linear Trend": "linear_trend_forecast"
    }

    selected_scenarios = st.multiselect(
        "Choose forecast scenarios to display",
        list(scenario_options.keys()),
        default=["Naive", "Linear FE Safe", "Linear Trend"]
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(workforce_df["year"], workforce_df["active_employees"], marker="o", linewidth=2.5, label="Historical")

    for label in selected_scenarios:
        col = scenario_options[label]
        ax.plot(forecast_df["year"], forecast_df[col], marker="o", linestyle="--", label=label)

    ax.set_title("Forecast Comparison (2026–2030)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Employees")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Forecast Table")
    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast table as CSV",
        data=csv,
        file_name="forecast_scenarios.csv",
        mime="text/csv"
    )

    with st.expander("Scenario interpretation"):
        st.write("""
        - **Naive:** stability scenario, assumes the workforce remains unchanged.
        - **Linear FE Safe:** moderate growth scenario using lag-based dynamics.
        - **Linear Trend:** long-term growth scenario based on overall historical trend.
        """)

# -----------------------------
# PROJECT NOTES
# -----------------------------
elif page == "Project Notes":
    st.title("📝 Project Notes")

    st.markdown("""
    ### Objective
    Forecast the evolution of public sector employees in Morocco to support planning and budget decisions.

    ### Methodology
    - Data cleaning and validation
    - Construction of yearly workforce dataset
    - Feature engineering
    - Comparison of statistical, machine learning, and deep learning models
    - Scenario-based forecasting

    ### Final Positioning
    - **Naive** kept as baseline
    - **Linear FE Safe** kept as main practical model
    - **Linear Trend** kept as strategic trend scenario
    """)

    with st.expander("Key takeaway"):
        st.write("""
        In this project, simple and interpretable models outperformed many complex ones.
        This shows that model choice should depend on data characteristics rather than complexity alone.
        """)
