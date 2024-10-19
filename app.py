




import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

model_path = r'C:\Python\Stock\Stock Predictions Model.keras'
model = load_model(model_path)

# Page config with a modern theme and wider layout
st.set_page_config(page_title='StockInsightAI - Market Predictor', layout='wide', page_icon="📈")

# Custom CSS for fonts and background
st.markdown("""
    <style>
        body {
            background-color: #F0F2F6;
        }
        .main {
            font-family: 'Verdana', sans-serif;
        }
        .title h1 {
            color: #2B547E;
        }
        .sidebar .sidebar-content {
            background-color: #D8E3E7;
        }
    </style>
    """, unsafe_allow_html=True)

st.title('📈 StockInsightAI - Predict Stock Prices with AI')

# Input for stock symbol
stock_symbol = st.text_input('Enter Stock Symbol:', 'GOOG')

start_date = '2012-01-01'
end_date = '2022-12-31'

# Fetching stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)

if data.empty:
    st.error("No data found for the stock symbol provided. Please try a different symbol.")
else:
    st.subheader('📊 Stock Data')
    st.dataframe(data)

    data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
    data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    last_100_days = data_train.tail(100)
    data_test = pd.concat([last_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.fit_transform(data_test)

    # Moving Averages using Plotly for interactive charts
    def plot_moving_averages(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=50).mean(), mode='lines', name='MA 50', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=100).mean(), mode='lines', name='MA 100', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=200).mean(), mode='lines', name='MA 200', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='black')))
        fig.update_layout(title=f'Moving Averages for {stock_symbol}', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
        st.plotly_chart(fig)

    plot_moving_averages(data)

    x = []
    y = []
    for i in range(100, data_test_scaled.shape[0]):
        x.append(data_test_scaled[i-100:i])
        y.append(data_test_scaled[i, 0])

    x, y = np.array(x), np.array(y)

    predicted_prices = model.predict(x)
    scale = 1 / scaler.scale_[0]
    predicted_prices = predicted_prices * scale
    y = y * scale

    # Plot the actual vs predicted prices using Plotly
    def plot_predictions():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(predicted_prices)), y=predicted_prices.flatten(), mode='lines', name='Predicted Prices', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y.flatten(), mode='lines', name='Actual Prices', line=dict(color='green')))
        fig.update_layout(title='Original Price vs Predicted Price', xaxis_title='Days', yaxis_title='Price', template='plotly_white')
        st.plotly_chart(fig)

    plot_predictions()

    st.subheader('📈 Prediction Statistics')
    st.write(f"Mean Absolute Error (MAE): {np.mean(np.abs(predicted_prices - y)):.2f}")
    st.write(f"Mean Squared Error (MSE): {np.mean((predicted_prices - y)**2):.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(np.mean((predicted_prices - y)**2)):.2f}")

    # Adding stock daily returns
    st.subheader('📊 Additional Insights')
    data['Returns'] = data['Close'].pct_change()
    fig = go.Figure([go.Scatter(x=data.index, y=data['Returns'], mode='lines', name='Daily Returns', line=dict(color='purple'))])
    fig.update_layout(title=f'{stock_symbol} Daily Returns', xaxis_title='Date', yaxis_title='Daily Returns', template='plotly_white')
    st.plotly_chart(fig)

    # Volatility chart
    vol_window = st.slider('Select Volatility Window (in days)', 10, 120, 30)
    data['Volatility'] = data['Returns'].rolling(vol_window).std()
    fig = go.Figure([go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility', line=dict(color='orange'))])
    fig.update_layout(title=f'{stock_symbol} {vol_window}-day Volatility', xaxis_title='Date', yaxis_title=f'{vol_window}-day Volatility', template='plotly_white')
    st.plotly_chart(fig)

