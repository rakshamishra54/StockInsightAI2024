# StockInsightAI2024 📈

**StockInsightAI2024** is a stock market prediction app that uses **Long Short-Term Memory (LSTM)** neural networks to forecast stock prices. The app provides users with stock price predictions, moving averages, and interactive charts for better decision-making in stock trading.

## 🌟 Inspiration

I was always intrigued by the stock market's unpredictable nature and wanted to create a tool that could help make sense of the chaos. By combining my passion for data science and the financial world, I set out to build an app that offers stock price predictions, hoping to make stock trading more data-driven.

## 🛠️ Features

- **Stock Price Prediction**: Predict future stock prices using LSTM models trained on historical data.
- **Moving Averages**: Gain insights into stock trends with **50-day**, **100-day**, and **200-day** moving averages.
- **Interactive Charts**: Visualize historical and predicted stock prices with dynamic charts.
- **User-Friendly Interface**: Easily input stock symbols and retrieve predictions through an intuitive Streamlit interface.

## 🚀 How It Works

1. **LSTM Model**: The LSTM neural network is trained on historical stock price data fetched using **yFinance**.
2. **Data Visualization**: Predictions and historical prices are plotted using **Matplotlib** for easy comparison.
3. **Moving Averages**: Calculate 50-day, 100-day, and 200-day moving averages to identify stock trends.

## How I built it
Building StockInsightAI2024 felt like assembling a jigsaw puzzle with missing pieces, but hey, challenges make it fun! I leveraged LSTM neural networks to build a robust model that learns from historical stock data. Streamlit provided a clean, interactive interface, while yFinance fetched real-time data. Matplotlib made the charts visually appealing, allowing me to focus on crafting the predictive model.

## Datasets Used
1. **Yahoo Finance Historical Data:** Used for training the LSTM model. [Yahoo Finance API](https://pypi.org/project/yfinance/)
2. **Other Financial Datasets:** [Kaggle Datasets](https://www.kaggle.com/datasets) - You can explore various datasets related to stock prices and financial information.

## 📦 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Keras, TensorFlow
- **Data Fetching**: yFinance
- **Visualization**: Matplotlib
- **Model**: LSTM (Long Short-Term Memory)

## What I learned
- **Data is King:** Clean, well-structured data is crucial for a reliable model.
- **The Power of Visualization:** Engaging visualizations are essential for interpreting model results.
- **Perseverance is Key:** Debugging requires patience and persistence.

## What's next for StockInsightAI2024
- **Feature Expansion:** Integrate additional technical indicators and improve the user interface.
- **Deploy to the Cloud:** Make the app accessible to a wider audience.
- **Continuous Learning:** Dive deeper into machine learning techniques and experiment with different models to enhance prediction accuracy.


## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/rakshamishra54/StockInsightAI2024.git

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/) for providing a fantastic platform for building interactive web apps.
- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for the powerful machine learning capabilities.
- [Matplotlib](https://matplotlib.org/) for the beautiful visualizations.  

