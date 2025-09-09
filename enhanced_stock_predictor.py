# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")

# -----------------------------
# Custom CSS (Modern Dark Theme)
# -----------------------------
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stSidebar, .stTabs [data-baseweb="tab-list"] { background-color: #1a1f2e; }
    .stButton>button { 
        background-color: #4CAF50; color: white; border: none; 
        border-radius: 8px; height: 50px; font-size: 16px;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #2a2f3e; color: white; border: 1px solid #444;
    }
    h1, h2, h3 { color: #a8d8ea !important; }
    .stTabs [data-baseweb="tab"] { color: #c0c0ff; }
    .stTabs [aria-selected='true'] { color: #4CAF50; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Global Ticker Suggestions
# -----------------------------
TICKER_SUGGESTIONS = [
    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "ORCL",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "BP.L", "VOD.L", "SAP.DE", "BMW.DE", "BABA", "700.HK"
]

# -----------------------------
# Fetch Stock Data
# -----------------------------
def fetch_data(ticker, period="2y"):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period)
        info = tk.info
        if df.empty or df['Close'].isna().all():
            return None, None, f"No data for {ticker}"
        return df[['Close', 'Open', 'High', 'Low', 'Volume']], info, None
    except Exception as e:
        return None, None, str(e)

# -----------------------------
# Add Technical Indicators
# -----------------------------
def add_technicals(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# -----------------------------
# Plot Price + Indicators
# -----------------------------
def plot_tech_chart(data, ticker):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} Price", "MACD", "RSI")
    )

    # Candlestick + SMAs
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name="OHLC"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], line=dict(color="orange", width=1), name="SMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color="yellow", width=1), name="SMA 50"), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], line=dict(color="blue"), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], line=dict(color="red"), name="Signal"), row=2, col=1)
    colors = ['green' if val > 0 else 'red' for val in (data['MACD'] - data['MACD_Signal'])]
    fig.add_trace(go.Bar(x=data.index, y=data['MACD'] - data['MACD_Signal'], marker_color=colors, name="Hist"), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color="purple"), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# LSTM Model
# -----------------------------
def build_lstm(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

def prepare_lstm_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(time_step, len(scaled)):
        X.append(scaled[i-time_step:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, time_step, 1)
    y = np.array(y)
    return X, y, scaler

# -----------------------------
# Portfolio Analysis
# -----------------------------
def analyze_portfolio(file):
    try:
        portfolio = pd.read_csv(file)
        if 'Ticker' not in portfolio.columns or 'Shares' not in portfolio.columns:
            st.error("CSV must have 'Ticker' and 'Shares' columns.")
            return

        tickers = portfolio['Ticker'].tolist()
        shares = portfolio['Shares'].tolist()
        prices, values, names = [], [], []

        for t in tickers:
            df, info, err = fetch_data(t, "1d")
            price = df['Close'].iloc[-1] if not err else 0
            name = info.get('longName', t) if not err else "Unknown"
            prices.append(price)
            names.append(name)

        portfolio['Company'] = names
        portfolio['Current Price'] = prices
        portfolio['Market Value'] = portfolio['Shares'] * portfolio['Current Price']
        total = portfolio['Market Value'].sum()
        portfolio['Allocation (%)'] = (portfolio['Market Value'] / total) * 100

        st.dataframe(portfolio)
        st.metric("Total Portfolio Value", f"${total:,.2f}")

        fig = go.Figure(data=[go.Pie(labels=portfolio['Ticker'], values=portfolio['Market Value'])])
        fig.update_layout(title="Portfolio Allocation", template="plotly_dark")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error reading file: {e}")

# -----------------------------
# Tabs UI
# -----------------------------
st.title("🚀 Stock Analyzer Pro")
st.markdown("Predictions • Technicals • Portfolio Analysis")

tab1, tab2, tab3 = st.tabs(["📈 Predict", "📊 Technicals", "💼 Portfolio"])

# ----------------------------- TAB 1: Predict
with tab1:
    st.header("Price Prediction")
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    model_choice = st.selectbox("Model", ["LSTM", "Linear Regression"])
    days = st.slider("Days to Predict", 1, 30, 7)

    if st.button("Run Prediction"):
        data, info, err = fetch_data(ticker)
        if err:
            st.error(err)
        else:
            st.subheader(f"{info.get('longName', ticker)} ({ticker})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Price"))
            fig.update_layout(template="plotly_dark", title="Historical Price")
            st.plotly_chart(fig)

            prices = data['Close'].values
            if model_choice == "LSTM" and len(prices) > 60:
                X, y, scaler = prepare_lstm_data(prices)
                split = int(0.8 * len(X))
                X_train = X[:split]
                model = build_lstm(X_train, y[:split])
                last = scaler.transform(prices[-60:].reshape(-1, 1))
                preds = []
                current_seq = last.copy()
                for _ in range(days):
                    p = model.predict(current_seq.reshape(1, 60, 1), verbose=0)[0,0]
                    preds.append(p)
                    current_seq = np.append(current_seq[1:], [[p]], axis=0)
                preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            else:
                X = np.array(range(len(prices))).reshape(-1, 1)
                model = LinearRegression().fit(X, prices)
                future_X = np.array(range(len(prices), len(prices)+days)).reshape(-1,1)
                preds = model.predict(future_X)

            future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=days)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=future_dates, y=preds, name="Prediction", line=dict(color="lime")))
            fig2.update_layout(template="plotly_dark", title="Forecast")
            st.plotly_chart(fig2)
            st.write(pd.DataFrame({"Date": future_dates, "Predicted Price": preds.round(2)}))

# ----------------------------- TAB 2: Technicals
with tab2:
    st.header("Technical Analysis")
    tech_ticker = st.text_input("Ticker", "AAPL", key="tech").upper()
    if st.button("Analyze"):
        data, _, err = fetch_data(tech_ticker)
        if err:
            st.error(err)
        else:
            data = add_technicals(data)
            plot_tech_chart(data, tech_ticker)

# ----------------------------- TAB 3: Portfolio
with tab3:
    st.header("Portfolio Analyzer")
    uploaded = st.file_uploader("Upload portfolio.csv", type="csv")
    if uploaded:
        analyze_portfolio(uploaded)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("✅ Powered by Yahoo Finance | For education only.")