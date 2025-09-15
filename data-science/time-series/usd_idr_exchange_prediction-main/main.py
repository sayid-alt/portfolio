import streamlit as st
import pandas as pd
from functions import (
    load_train_forecast,
    calculate_metrics,
    forecast_chart,
    load_data,
    history_chart
)

st.set_page_config(
    page_title="Idr=X Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# load train forecast data
train_forecast_df = load_train_forecast(
    gru_path_csv='data/train_forecast_gru.csv', lstm_path_csv='data/train_forecast_lstm.csv')
valid_forecast_df = load_train_forecast(
    gru_path_csv='data/valid_forecast_gru.csv', lstm_path_csv='data/valid_forecast_lstm.csv')


def main():
    # custom css for streamlit
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {
            font-size: 25px;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("20 years IDR=X price prediction with LSTM and GRU!")
    scores_col1, scores_col2 = st.columns(
        2, border=True)

    with scores_col1:
        st.subheader("Training Data Size")
        st.metric(label="Rows", value=train_forecast_df.shape[0], border=False)

    with scores_col2:
        st.subheader("Validation Data Size")
        st.metric(label="Rows", value=valid_forecast_df.shape[0], border=False)

    train_forecast_chart, valid_forecast_chart = st.columns(2, border=True)

    with train_forecast_chart:
        st.subheader("Training Data Forecast")
        forecast_chart(
            train_forecast_df, title_chart="LSTM & GRU Close Price Prediction vs True Price on Training Data")

    with valid_forecast_chart:
        st.subheader("Validation Data Forecast")
        forecast_chart(
            valid_forecast_df, title_chart="LSTM & GRU Close Price Prediction vs True Price on Validation Data")

    raw_train_col, raw_valid_col = st.columns(2, border=True)
    with raw_train_col:
        st.subheader("Raw Data - Training vs Forecast")
        st.dataframe(train_forecast_df, use_container_width=True)

    with raw_valid_col:
        st.subheader("Raw Data - Validation vs Forecast")
        st.dataframe(valid_forecast_df, use_container_width=True)

    learning_history = st.container()
    with learning_history:
        st.subheader("Learning History")
        left_col, right_col = st.columns(2, border=True)
        with left_col:
            history_chart(
                df=load_data('data/best_gru_history.csv'),
                title_chart='GRU Model Training & Validation Loss History'
            )
        with right_col:
            history_chart(
                df=load_data('data/best_lstm_history.csv'),
                title_chart='LSTM Model Training & Validation Loss History'
            )


if __name__ == "__main__":
    main()
