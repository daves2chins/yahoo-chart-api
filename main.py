from flask import Flask, request, send_file, jsonify
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import io
import datetime
import os
import pandas as pd

app = Flask(__name__)

@app.route('/chart')
def chart():
    try:
        ticker = request.args.get('ticker', default='AAPL').upper()
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1)

        # Fetch 1-minute interval data from Yahoo Finance
        data = yf.download(ticker, start=start, end=end, interval='1m')
        if data.empty:
            return jsonify({"error": f"No data found for ticker {ticker}"}), 404

        # Calculate indicators using proper 1D Series
        close = data['Close']
        high = data['High']
        low = data['Low']

        data['rsi'] = ta.momentum.RSIIndicator(close=close).rsi()
        data['atr'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range()
        data['adx'] = ta.trend.ADXIndicator(high=high, low=low, close=close).adx()

        # Williams Alligator (Jaw: SMMA(13), Teeth: SMMA(8), Lips: SMMA(5))
        data['jaw'] = close.ewm(span=13, adjust=False).mean().shift(8)
        data['teeth'] = close.ewm(span=8, adjust=False).mean().shift(5)
        data['lips'] = close.ewm(span=5, adjust=False).mean().shift(3)

        # Plotting
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        axs[0].plot(data.index, close, label='Close Price', color='blue')
        axs[0].plot(data.index, data['jaw'], label='Jaw (13, 8)', color='navy')
        axs[0].plot(data.index, data['teeth'], label='Teeth (8, 5)', color='red')
        axs[0].plot(data.index, data['lips'], label='Lips (5, 3)', color='green')
        axs[0].set_title(f'{ticker} - Last 24h Price with Williams Alligator')
        axs[0].legend()

        axs[1].plot(data.index, data['rsi'], label='RSI', color='purple')
        axs[1].axhline(70, color='red', linestyle='--', linewidth=0.8)
        axs[1].axhline(30, color='green', linestyle='--', linewidth=0.8)
        axs[1].legend()

        axs[2].plot(data.index, data['atr'], label='ATR', color='orange')
        axs[2].legend()

        axs[3].plot(data.index, data['adx'], label='ADX', color='brown')
        axs[3].legend()

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
