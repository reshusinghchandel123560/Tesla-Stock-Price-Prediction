"""
Tesla Stock Price Prediction - Streamlit Application
======================================================
Interactive app for exploring TSLA historical data, model performance
(SimpleRNN vs LSTM vs Tuned LSTM), and generating 1/5/10-day-ahead
closing-price forecasts using the trained best model.

Run locally with:
    streamlit run app.py

All paths are relative, so the app works regardless of where the
project folder is placed (local machine, Streamlit Cloud, Render, Railway, etc.)
as long as the folder structure below is preserved:

    project_root/
        app/app.py          <- this file
        data/TSLA.csv
        data/TSLA_processed.csv
        models/best_model.keras
        models/scaler.pkl
        models/metadata.pkl
        models/streamlit_assets.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model

# ------------------------------------------------------------------
# PATHS (relative to this file -> portable across environments)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

RAW_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'TSLA.csv')
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'TSLA_processed.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'best_model.keras')
SCALER_PATH = os.path.join(ROOT_DIR, 'models', 'scaler.pkl')
METADATA_PATH = os.path.join(ROOT_DIR, 'models', 'metadata.pkl')
ASSETS_PATH = os.path.join(ROOT_DIR, 'models', 'streamlit_assets.pkl')

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').drop_duplicates(subset='Date').set_index('Date')
    df = df.ffill().bfill()
    return df

@st.cache_data
def load_processed_data():
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    return df

@st.cache_resource
def load_model_assets():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, scaler, metadata

@st.cache_data
def load_streamlit_assets():
    return joblib.load(ASSETS_PATH)


def inverse_transform_target(scaled_values, scaler, target_idx, n_features):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]


def make_forecast(model, scaler, metadata, feature_df):
    feature_cols = metadata['feature_columns']
    target_idx = metadata['target_col_idx']
    lookback = metadata['lookback']
    horizons = metadata['horizons']
    n_features = len(feature_cols)

    model_data = feature_df[feature_cols].values
    last_window_raw = model_data[-lookback:]
    last_window_scaled = scaler.transform(last_window_raw).reshape(1, lookback, n_features)

    preds_scaled = model.predict(last_window_scaled, verbose=0)[0]
    preds = inverse_transform_target(preds_scaled, scaler, target_idx, n_features)
    return dict(zip(horizons, preds))


# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("📈 TSLA Stock Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Dataset Overview", "EDA", "Model Performance", "Forecasting"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with SimpleRNN & LSTM deep learning models trained on historical "
    "Tesla OHLCV data. Educational / portfolio project — not financial advice."
)

# ------------------------------------------------------------------
# PAGE: HOME
# ------------------------------------------------------------------
if page == "Home":
    st.title("Tesla (TSLA) Stock Price Prediction")
    st.subheader("SimpleRNN vs. LSTM — 1-Day, 5-Day & 10-Day Forecasts")

    st.markdown("""
    This application accompanies the **Tesla Stock Price Prediction** deep learning project.
    It uses historical Tesla OHLCV (Open, High, Low, Close, Adjusted Close, Volume) data to:

    - Explore historical trends, distributions, and volatility (**EDA** page)
    - Compare **SimpleRNN**, **LSTM**, and a **hyperparameter-tuned LSTM** on multiple
      forecast horizons (**Model Performance** page)
    - Generate **1-day, 5-day, and 10-day** ahead closing price forecasts using the
      best-performing model (**Forecasting** page)

    ### How it works
    1. The model looks at the last **60 trading days** of engineered features
       (price, moving averages, returns, volatility, lags, volume).
    2. A trained recurrent neural network (SimpleRNN or LSTM) predicts the closing
       price 1, 5, and 10 trading days ahead.
    3. Predictions are inverse-scaled back to dollar values for interpretation.

    > ⚠️ **Disclaimer:** This tool is for educational and portfolio demonstration purposes
    > only. Stock price predictions from historical data alone should **never** be the sole
    > basis for real trading or investment decisions.
    """)

    try:
        raw_df = load_raw_data()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", f"${raw_df['Close'].iloc[-1]:.2f}")
        col2.metric("52-Week High", f"${raw_df['Close'].tail(252).max():.2f}")
        col3.metric("52-Week Low", f"${raw_df['Close'].tail(252).min():.2f}")
        col4.metric("Total Trading Days", f"{len(raw_df):,}")

        fig = px.line(raw_df.reset_index(), x='Date', y='Close',
                       title='TSLA Closing Price — Full History')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning("Dataset not found. Make sure `data/TSLA.csv` exists.")

# ------------------------------------------------------------------
# PAGE: DATASET OVERVIEW
# ------------------------------------------------------------------
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")

    raw_df = load_raw_data()

    st.markdown(f"**Shape:** {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    st.markdown(f"**Date Range:** {raw_df.index.min().date()} → {raw_df.index.max().date()}")

    tab1, tab2, tab3 = st.tabs(["Head", "Tail", "Statistical Summary"])
    with tab1:
        st.dataframe(raw_df.head(10))
    with tab2:
        st.dataframe(raw_df.tail(10))
    with tab3:
        st.dataframe(raw_df.describe())

    st.markdown("### Column Data Types")
    st.dataframe(raw_df.dtypes.astype(str).rename("dtype"))

    st.markdown("### Missing Values (after cleaning)")
    st.dataframe(raw_df.isna().sum().rename("missing_count"))

# ------------------------------------------------------------------
# PAGE: EDA
# ------------------------------------------------------------------
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    raw_df = load_raw_data()
    proc_df = load_processed_data()

    st.markdown("### Price Trends")
    price_cols = st.multiselect(
        "Select price series to plot",
        options=['Open', 'High', 'Low', 'Close', 'Adj Close'],
        default=['Close']
    )
    if price_cols:
        fig = px.line(raw_df.reset_index(), x='Date', y=price_cols,
                       title='TSLA Price Trends')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Trading Volume")
    fig = px.line(raw_df.reset_index(), x='Date', y='Volume', title='TSLA Trading Volume')
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Close Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(raw_df['Close'], bins=50, kde=True, color='royalblue', ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("### Daily Returns Distribution")
        returns = raw_df['Close'].pct_change().dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(returns, bins=80, kde=True, color='purple', ax=ax)
        st.pyplot(fig)

    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = raw_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    st.markdown("### Moving Averages")
    if {'MA20', 'MA50', 'MA100', 'Close'}.issubset(proc_df.columns):
        ma_choice = st.multiselect(
            "Select moving averages to overlay",
            options=['MA20', 'MA50', 'MA100'],
            default=['MA20', 'MA50']
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=proc_df.index, y=proc_df['Close'], name='Close', opacity=0.6))
        for ma in ma_choice:
            fig.add_trace(go.Scatter(x=proc_df.index, y=proc_df[ma], name=ma))
        fig.update_layout(title='Close Price with Moving Averages', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Rolling Volatility (20-day)")
    if 'Volatility_20' in proc_df.columns:
        fig = px.line(proc_df.reset_index(), x=proc_df.index.name or 'index', y='Volatility_20',
                       title='20-Day Rolling Volatility')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# ------------------------------------------------------------------
elif page == "Model Performance":
    st.title("🧠 Model Performance: SimpleRNN vs. LSTM vs. Tuned LSTM")

    try:
        assets = load_streamlit_assets()
        comparison_df = assets['comparison_df']
        best_model_name = assets['best_model_name']
        avg_rmse = assets['avg_rmse']

        st.success(f"Best model on the test set (lowest average RMSE): **{best_model_name}**")

        st.markdown("### Average RMSE per Model (across 1/5/10-day horizons)")
        st.dataframe(pd.DataFrame(avg_rmse.items(), columns=['Model', 'Avg RMSE']))

        st.markdown("### Full Comparison Table")
        st.dataframe(comparison_df)

        st.markdown("### Metric Comparison Chart")
        metric_choice = st.selectbox("Select metric", ['RMSE', 'MSE', 'MAE', 'MAPE (%)', 'R2'])
        fig = px.bar(comparison_df, x='Horizon', y=metric_choice, color='Model',
                      barmode='group', title=f'{metric_choice} by Model and Horizon')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Actual vs. Predicted (Test Set)")
        horizon_choice = st.selectbox("Select horizon", ['1-day', '5-day', '10-day'])
        model_choice = st.selectbox("Select model", ['SimpleRNN', 'LSTM', 'Tuned LSTM'])

        results_map = {
            'SimpleRNN': assets['rnn_results'],
            'LSTM': assets['lstm_results'],
            'Tuned LSTM': assets['tuned_results']
        }
        res = results_map[model_choice][horizon_choice]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=res['y_true'], name='Actual', line=dict(color='black')))
        fig.add_trace(go.Scatter(y=res['y_pred'], name=f'{model_choice} Predicted'))
        fig.update_layout(
            title=f'{model_choice}: Actual vs Predicted Close Price ({horizon_choice} ahead)',
            xaxis_title='Test Sample', yaxis_title='Close Price ($)', template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning(
            "Model performance assets not found. Run the notebook "
            "`notebooks/Tesla_Stock_Price_Prediction.ipynb` end-to-end first "
            "to generate `models/streamlit_assets.pkl`."
        )

# ------------------------------------------------------------------
# PAGE: FORECASTING
# ------------------------------------------------------------------
elif page == "Forecasting":
    st.title("🔮 Future Price Forecasting")

    try:
        model, scaler, metadata = load_model_assets()
        proc_df = load_processed_data()
        assets = load_streamlit_assets()

        st.markdown(f"**Model in use:** `{assets['best_model_name']}` "
                     f"(automatically selected as the best performer in the notebook)")
        st.markdown(f"**Lookback window:** {metadata['lookback']} trading days")
        st.markdown(f"**Last known close price:** ${assets['last_close']:.2f}")

        if st.button("Generate Forecast", type="primary"):
            forecast = make_forecast(model, scaler, metadata, proc_df)

            forecast_df = pd.DataFrame({
                'Horizon': [f'{h}-Day' for h in forecast.keys()],
                'Predicted Close Price ($)': [round(v, 2) for v in forecast.values()]
            })

            st.markdown("### Forecast Table")
            st.dataframe(forecast_df, use_container_width=True)

            st.markdown("### Individual Forecast Horizons")

            col1, col2, col3 = st.columns(3)

            with col1:
                fig = go.Figure()
                fig.add_bar(
                    x=["Today", "1 Day"],
                    y=[assets['last_close'], forecast[1]]
                )
                fig.update_layout(title="1-Day Forecast")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_bar(
                    x=["Today", "5 Day"],
                    y=[assets['last_close'], forecast[5]]
                )
                fig.update_layout(title="5-Day Forecast")
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                fig = go.Figure()
                fig.add_bar(
                    x=["Today", "10 Day"],
                    y=[assets['last_close'], forecast[10]]
                )
                fig.update_layout(title="10-Day Forecast")
                st.plotly_chart(fig, use_container_width=True)
            horizons_x = [0] + list(forecast.keys())
            values_y = [assets['last_close']] + list(forecast.values())
            labels = ['Today'] + [f'+{h} Day{"s" if h > 1 else ""}' for h in forecast.keys()]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labels, y=values_y, mode='lines+markers+text',
                text=[f"${v:.2f}" for v in values_y], textposition='top center',
                line=dict(color='seagreen', dash='dash')
            ))
            fig.update_layout(
                title=f"TSLA Forecast — Next 1/5/10 Trading Days ({assets['best_model_name']})",
                xaxis_title="Horizon", yaxis_title="Predicted Close Price ($)",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Forecasts are generated using the last 60 trading days of data found in "
                "`data/TSLA_processed.csv`. To forecast from a more recent date, regenerate "
                "this file by re-running the notebook with updated raw data."
            )
        else:
            st.info("Click **Generate Forecast** to predict the next 1, 5, and 10-day closing prices.")

    except FileNotFoundError as e:
        st.warning(
            "Required model files not found. Please run the notebook "
            "`notebooks/Tesla_Stock_Price_Prediction.ipynb` end-to-end first to generate "
            "the `models/` directory contents.\n\n"
            f"Missing: {e}"
        )
