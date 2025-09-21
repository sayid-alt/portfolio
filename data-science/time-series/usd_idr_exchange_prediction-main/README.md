# USD-IDR Exchange Rate Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

## 📊 Project Overview

This project implements time series forecasting models to predict USD to Indonesian Rupiah (IDR) exchange rates. It uses both LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) neural networks to capture temporal patterns in historical exchange rate data and make future predictions.

![USD-IDR Exchange Rate Prediction](https://img.shields.io/badge/USD--IDR-Exchange%20Prediction-brightgreen?style=for-the-badge)

## 🔍 Key Features

- **Dual Model Approach**: Implements both LSTM and GRU models for comparative analysis
- **Interactive Visualization**: Streamlit-based dashboard for exploring predictions
- **Performance Metrics**: RMSE, MAE, and MAPE evaluation for model comparison
- **Time Horizon Selection**: Adjustable date ranges for prediction visualization
- **Historical Data Analysis**: Uses 20 years of USD-IDR exchange rate data
- **Hyperparameter Tuning**: Optimized model parameters using Bayesian optimization

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for building and training models
- **Streamlit**: Web application framework for interactive dashboard
- **Pandas & NumPy**: Data manipulation and numerical computations
- **Plotly**: Interactive data visualization
- **yfinance**: Yahoo Finance API for retrieving financial data
- **scikit-learn**: For data preprocessing and metrics calculation
- **Keras Tuner**: For hyperparameter optimization

## 📈 Model Architecture

### LSTM Model
Long Short-Term Memory networks are specialized RNNs capable of learning long-term dependencies in time series data. Our implementation includes:
- Multiple LSTM layers with dropout for regularization
- Dense output layers for prediction
- Optimized hyperparameters for USD-IDR exchange rate prediction

### GRU Model
Gated Recurrent Units are simpler than LSTMs but often perform similarly well for many tasks. Our GRU model features:
- Multiple GRU layers with optimized units
- Dropout layers to prevent overfitting
- Dense output layers for final prediction

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/usd_idr_exchange_prediction.git
   cd usd_idr_exchange_prediction-main
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install -e .
   ```

### Running the Application

```bash
streamlit run main.py
```

The application will be available at http://localhost:8501 in your web browser.

## 📊 Dashboard Features

1. **Forecast Visualization**: Compare LSTM and GRU predictions against actual exchange rates
2. **Time Horizon Selection**: Select specific date ranges to focus on particular periods
3. **Performance Metrics**: View RMSE, MAE, and MAPE for both models
4. **Training History**: Visualize the learning curves for both models
5. **Data Size Information**: See the size of training and validation datasets

## 📝 Project Structure

```
├── data/                      # Contains CSV files with forecasts and model history
├── models/                    # Saved trained models
│   ├── best_gru_model_tuned_bayes_trained_10.keras
│   └── best_lstm_model_tuned_bayes_trained_10.keras
├── functions.py               # Utility functions for data processing and visualization
├── main.py                    # Streamlit application entry point
├── notebook.ipynb             # Jupyter notebook with model development and analysis
└── README.md                  # Project documentation
```

## 📈 Performance Evaluation

Both LSTM and GRU models are evaluated using:

- **RMSE (Root Mean Squared Error)**: Measures the standard deviation of prediction errors
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **MAPE (Mean Absolute Percentage Error)**: Average percentage difference between predicted and actual values

## 🔮 Future Improvements

- Implement ensemble methods combining LSTM and GRU predictions
- Add more external features (economic indicators, global events)
- Extend the model to predict other currency pairs
- Implement real-time prediction updates using live data feeds
- Add confidence intervals for predictions

## 📚 References

- [TensorFlow Time Series Forecasting Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.