# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model # type: ignore
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # Load the pre-trained model
# model_path = r'C:\Python\Stock\Stock Predictions Model.keras'
# model = load_model(model_path)

# # Set Streamlit page configuration
# st.set_page_config(page_title='Stock Market Predictor', layout='wide')

# # Application Title
# st.title('📈 Stock Market Predictor')

# # Stock Symbol Input
# stock_symbol = st.text_input('Enter Stock Symbol:', 'GOOG')
# start_date = '2012-01-01'
# end_date = '2022-12-31'

# # Fetch stock data
# data = yf.download(stock_symbol, start=start_date, end=end_date)

# if data.empty:
#     st.error("No data found for the stock symbol provided. Please try a different symbol.")
# else:
#     st.subheader('📊 Stock Data')
#     st.dataframe(data)

#     # Prepare the training and testing datasets
#     data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
#     data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

#     # Scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     last_100_days = data_train.tail(100)
#     data_test = pd.concat([last_100_days, data_test], ignore_index=True)
#     data_test_scaled = scaler.fit_transform(data_test)

#     # Plotting Moving Averages
#     def plot_moving_averages(data):
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(data['Close'].rolling(window=50).mean(), label='MA 50', color='red')
#         ax.plot(data['Close'].rolling(window=100).mean(), label='MA 100', color='blue')
#         ax.plot(data['Close'].rolling(window=200).mean(), label='MA 200', color='green')
#         ax.plot(data['Close'], label='Close Price', color='black')
#         ax.set_title(f'Moving Averages for {stock_symbol}')
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Price')
#         ax.legend()
#         st.pyplot(fig)

#     plot_moving_averages(data)

#     # Prepare the test data for predictions
#     x = []
#     y = []
#     for i in range(100, data_test_scaled.shape[0]):
#         x.append(data_test_scaled[i-100:i])
#         y.append(data_test_scaled[i, 0])

#     x, y = np.array(x), np.array(y)

#     # Make predictions
#     predicted_prices = model.predict(x)
#     scale = 1 / scaler.scale_[0]
#     predicted_prices = predicted_prices * scale
#     y = y * scale

#     # Plotting the original vs predicted prices
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(predicted_prices, color='red', label='Predicted Prices')
#     ax.plot(y, color='green', label='Actual Prices')
#     ax.set_title('Original Price vs Predicted Price')
#     ax.set_xlabel('Days')
#     ax.set_ylabel('Price')
#     ax.legend()
#     st.pyplot(fig)

#     # Additional Visualization: Prediction Statistics
#     st.subheader('📈 Prediction Statistics')
#     st.write(f"Mean Absolute Error (MAE): {np.mean(np.abs(predicted_prices - y)):.2f}")
#     st.write(f"Mean Squared Error (MSE): {np.mean((predicted_prices - y)**2):.2f}")
#     st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(np.mean((predicted_prices - y)**2)):.2f}")

# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# model_path = r'C:\Python\Stock\Stock Predictions Model.keras'
# model = load_model(model_path)

# st.set_page_config(page_title='Stock Market Predictor', layout='wide')
# st.title('📈 Stock Market Predictor')

# stock_symbol = st.text_input('Enter Stock Symbol:', 'GOOG')
# start_date = '2012-01-01'
# end_date = '2022-12-31'

# data = yf.download(stock_symbol, start=start_date, end=end_date)

# if data.empty:
#     st.error("No data found for the stock symbol provided. Please try a different symbol.")
# else:
#     st.subheader('📊 Stock Data')
#     st.dataframe(data)

#     data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
#     data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     last_100_days = data_train.tail(100)
#     data_test = pd.concat([last_100_days, data_test], ignore_index=True)
#     data_test_scaled = scaler.fit_transform(data_test)

#     def plot_moving_averages(data):
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(data['Close'].rolling(window=50).mean(), label='MA 50', color='red')
#         ax.plot(data['Close'].rolling(window=100).mean(), label='MA 100', color='blue')
#         ax.plot(data['Close'].rolling(window=200).mean(), label='MA 200', color='green')
#         ax.plot(data['Close'], label='Close Price', color='black')
#         ax.set_title(f'Moving Averages for {stock_symbol}')
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Price')
#         ax.legend()
#         st.pyplot(fig)

#     plot_moving_averages(data)

#     x = []
#     y = []
#     for i in range(100, data_test_scaled.shape[0]):
#         x.append(data_test_scaled[i-100:i])
#         y.append(data_test_scaled[i, 0])

#     x, y = np.array(x), np.array(y)

#     predicted_prices = model.predict(x)
#     scale = 1 / scaler.scale_[0]
#     predicted_prices = predicted_prices * scale
#     y = y * scale

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(predicted_prices, color='red', label='Predicted Prices')
#     ax.plot(y, color='green', label='Actual Prices')
#     ax.set_title('Original Price vs Predicted Price')
#     ax.set_xlabel('Days')
#     ax.set_ylabel('Price')
#     ax.legend()
#     st.pyplot(fig)

#     st.subheader('📈 Prediction Statistics')
#     st.write(f"Mean Absolute Error (MAE): {np.mean(np.abs(predicted_prices - y)):.2f}")
#     st.write(f"Mean Squared Error (MSE): {np.mean((predicted_prices - y)**2):.2f}")
#     st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(np.mean((predicted_prices - y)**2)):.2f}")

#     st.subheader('📊 Additional Insights')
#     data['Returns'] = data['Close'].pct_change()
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(data.index, data['Returns'], color='purple')
#     ax.set_title(f'{stock_symbol} Daily Returns')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Daily Returns')
#     st.pyplot(fig)

#     vol_window = st.slider('Select Volatility Window (in days)', 10, 120, 30)
#     data['Volatility'] = data['Returns'].rolling(vol_window).std()
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(data.index, data['Volatility'], color='orange')
#     ax.set_title(f'{stock_symbol} {vol_window}-day Volatility')
#     ax.set_xlabel('Date')
#     ax.set_ylabel(f'{vol_window}-day Volatility')
#     st.pyplot(fig)


# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# import plotly.graph_objects as go

# model_path = r'C:\Python\Stock\Stock Predictions Model.keras'
# model = load_model(model_path)

# # Page config with a modern theme and wider layout
# st.set_page_config(page_title='StockInsightAI - Market Predictor', layout='wide', page_icon="📈")

# # Custom CSS for fonts and background
# st.markdown("""
#     <style>
#         body {
#             background-color: #F0F2F6;
#         }
#         .main {
#             font-family: 'Verdana', sans-serif;
#         }
#         .title h1 {
#             color: #2B547E;
#         }
#         .sidebar .sidebar-content {
#             background-color: #D8E3E7;
#         }
#     </style>
#     """, unsafe_allow_html=True)

# st.title('📈 StockInsightAI - Predict Stock Prices with AI')

# # Input for stock symbol
# stock_symbol = st.text_input('Enter Stock Symbol:', 'GOOG')

# start_date = '2012-01-01'
# end_date = '2022-12-31'

# # Fetching stock data
# data = yf.download(stock_symbol, start=start_date, end=end_date)

# if data.empty:
#     st.error("No data found for the stock symbol provided. Please try a different symbol.")
# else:
#     st.subheader('📊 Stock Data')
#     st.dataframe(data)

#     data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
#     data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     last_100_days = data_train.tail(100)
#     data_test = pd.concat([last_100_days, data_test], ignore_index=True)
#     data_test_scaled = scaler.fit_transform(data_test)

#     # Moving Averages using Plotly for interactive charts
#     def plot_moving_averages(data):
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=50).mean(), mode='lines', name='MA 50', line=dict(color='red')))
#         fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=100).mean(), mode='lines', name='MA 100', line=dict(color='blue')))
#         fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=200).mean(), mode='lines', name='MA 200', line=dict(color='green')))
#         fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='black')))
#         fig.update_layout(title=f'Moving Averages for {stock_symbol}', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
#         st.plotly_chart(fig)

#     plot_moving_averages(data)

#     x = []
#     y = []
#     for i in range(100, data_test_scaled.shape[0]):
#         x.append(data_test_scaled[i-100:i])
#         y.append(data_test_scaled[i, 0])

#     x, y = np.array(x), np.array(y)

#     predicted_prices = model.predict(x)
#     scale = 1 / scaler.scale_[0]
#     predicted_prices = predicted_prices * scale
#     y = y * scale

#     # Plot the actual vs predicted prices using Plotly
#     def plot_predictions():
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=np.arange(len(predicted_prices)), y=predicted_prices.flatten(), mode='lines', name='Predicted Prices', line=dict(color='red')))
#         fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y.flatten(), mode='lines', name='Actual Prices', line=dict(color='green')))
#         fig.update_layout(title='Original Price vs Predicted Price', xaxis_title='Days', yaxis_title='Price', template='plotly_white')
#         st.plotly_chart(fig)

#     plot_predictions()

#     st.subheader('📈 Prediction Statistics')
#     st.write(f"Mean Absolute Error (MAE): {np.mean(np.abs(predicted_prices - y)):.2f}")
#     st.write(f"Mean Squared Error (MSE): {np.mean((predicted_prices - y)**2):.2f}")
#     st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(np.mean((predicted_prices - y)**2)):.2f}")

#     # Adding stock daily returns
#     st.subheader('📊 Additional Insights')
#     data['Returns'] = data['Close'].pct_change()
#     fig = go.Figure([go.Scatter(x=data.index, y=data['Returns'], mode='lines', name='Daily Returns', line=dict(color='purple'))])
#     fig.update_layout(title=f'{stock_symbol} Daily Returns', xaxis_title='Date', yaxis_title='Daily Returns', template='plotly_white')
#     st.plotly_chart(fig)

#     # Volatility chart
#     vol_window = st.slider('Select Volatility Window (in days)', 10, 120, 30)
#     data['Volatility'] = data['Returns'].rolling(vol_window).std()
#     fig = go.Figure([go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility', line=dict(color='orange'))])
#     fig.update_layout(title=f'{stock_symbol} {vol_window}-day Volatility', xaxis_title='Date', yaxis_title=f'{vol_window}-day Volatility', template='plotly_white')
#     st.plotly_chart(fig)


import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os  # Import os for file path check

# Use a relative path for model loading
model_path = 'Stock Predictions Model.keras'  # Make sure this file is in the same directory as your app.py

# Check if the model file exists
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Model file not found. Please check the file path.")
    st.stop()  # Stop execution if the model is not found

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
