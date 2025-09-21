import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import requests
import json

@st.cache_data(show_spinner=False)
def load_train_forecast(gru_path_csv: str = None, lstm_path_csv: str = None) -> pd.DataFrame:
    gru_forecast_df = pd.read_csv(
        gru_path_csv, index_col=0, parse_dates=True).reset_index()
    lstm_forecast_df = pd.read_csv(
        lstm_path_csv, index_col=0, parse_dates=True).reset_index()

    gru_forecast_df.rename(columns={"index": "Date"}, inplace=True)
    lstm_forecast_df.rename(columns={"index": "Date"}, inplace=True)

    summary_forecast_df = pd.merge(
        gru_forecast_df, lstm_forecast_df, on='Date', suffixes=('_GRU', '_LSTM'))

    summary_forecast_df.drop(columns=['Close_GRU'], inplace=True)
    summary_forecast_df.rename(
        columns={'Close_LSTM': 'Close_Price'}, inplace=True)
    return summary_forecast_df


@st.cache_data(show_spinner=False)
def load_data(path_csv: str = None) -> pd.DataFrame:
    history_df = pd.read_csv(path_csv)
    return history_df


def calculate_metrics(y_true, y_pred):
    scaler = MinMaxScaler(feature_range=(1e-1, 1))
    y_true_scaled = scaler.fit_transform(y_true.values.reshape(-1, 1))
    y_pred_scaled = scaler.transform(y_pred.values.reshape(-1, 1))

    rmse = root_mean_squared_error(y_true_scaled, y_pred_scaled)
    mae = mean_absolute_error(y_true_scaled, y_pred_scaled)
    mape = mean_absolute_percentage_error(y_true_scaled, y_pred_scaled)
    return {
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'mape': round(mape, 3)
    }

def calc_percentage_delta_between_to_point(value, relative_value):
    return round(((value - relative_value) / relative_value) * 100, 2)

def forecast_chart(df, title_chart):
    # Time Horizon Selection
    time_horizon = st.date_input(
        "Select the time horizon for prediction (in days):",
        value=(df['Date'][0],
               df['Date'][800]),
        min_value=df['Date'][0],
        max_value=df['Date'].iloc[-1]
    )
    st.markdown(
        f"Selected time horizon :green-badge[{time_horizon[0]}] to :green-badge[{time_horizon[1]}]"
    )

    # Chart Traiingng Forecast
    start_date = pd.to_datetime(time_horizon[0])
    end_date = pd.to_datetime(time_horizon[1])

    # filter data within selected time horizon
    mask = (df['Date'] >= start_date) & (
        df['Date'] <= end_date)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig = px.line(
        data_frame=df[mask],
        x="Date",
        y=df.columns[1:],
        title=title_chart,
        color_discrete_sequence=colors,
    )
    st.plotly_chart(fig, use_container_width=True)

    # metrics evaluation within selected time horizon
    lstm_metrics = calculate_metrics(
        y_true=df[mask]['Close_Price'], y_pred=df[mask]['Close_Forecast_LSTM'])
    gru_metrics = calculate_metrics(
        y_true=df[mask]['Close_Price'], y_pred=df[mask]['Close_Forecast_GRU'])
    print('lstm metric', lstm_metrics)


    metric_col1, metric_col2, metric_col3 = st.columns(3, border=True)
    with metric_col1:
        st.subheader("RMSE")
        lstm, gru = st.columns(2)
        with lstm:
            lstm_rmse_delta = calc_percentage_delta_between_to_point(
                value=lstm_metrics['rmse'],
                relative_value=gru_metrics['rmse']
            )

            st.metric(
                label="LSTM",
                value=lstm_metrics['rmse'],
                delta=f"{lstm_rmse_delta}%",
                border=False
            )

        with gru:
            gru_rmse_delta = calc_percentage_delta_between_to_point(
                value=gru_metrics['rmse'],
                relative_value=lstm_metrics['rmse']
            )

            st.metric(
                label="GRU",
                value=gru_metrics['rmse'],
                delta=f"{gru_rmse_delta}%",
                border=False
            )

    with metric_col2:
        st.subheader("MAE")
        lstm, gru = st.columns(2)
        with lstm:
            lstm_mae_delta = calc_percentage_delta_between_to_point(
                value=lstm_metrics['mae'],
                relative_value=gru_metrics['mae']
            )

            st.metric(
                label="LSTM",
                value=lstm_metrics['mae'],
                delta=f"{lstm_mae_delta}%",
                border=False
            )
        with gru:
            gru_mae_delta = calc_percentage_delta_between_to_point(
                value=gru_metrics['mae'],
                relative_value=lstm_metrics['mae']
            )

            st.metric(
                label="GRU",
                value=gru_metrics['mae'],
                delta=f"{gru_mae_delta}%",
                border=False
            )

    with metric_col3:
        st.subheader("MAPE %")
        lstm, gru = st.columns(2)

        lstm_mape_delta = calc_percentage_delta_between_to_point(
            value=lstm_metrics['mape'],
            relative_value=gru_metrics['mape']
        )

        with lstm:
            st.metric(
                label="LSTM",
                value=f"{lstm_metrics['mape']}%",
                delta=f"{lstm_mape_delta}%",
                border=False
            )

        with gru:
            gru_mape_delta = calc_percentage_delta_between_to_point(
                value=gru_metrics['mape'],
                relative_value=lstm_metrics['mape']
            )

            st.metric(
                label="GRU",
                value=f"{gru_metrics['mape']}%",
                delta=f"{gru_mape_delta}%",
                border=False
            )


def history_chart(df, title_chart):
    fig = px.line(
        data_frame=df,
        x="epochs",
        y=[col for col in df.columns if col not in [
            'epochs', 'loss', 'val_loss', 'learning_rate']],
        title=title_chart,

    )
    st.plotly_chart(fig, use_container_width=True)

def predict_price(data):
    headers = {"Content-Type": "application/json"}
    json_data = json.dumps({"signature_name": "serving_default","instances": data.tolist()})
    print(json_data)
    response = requests.post(
        url='http://localhost:8501/v1/models/forecasts_gru_serving:predict',
        data=json_data,
        headers=headers
    )