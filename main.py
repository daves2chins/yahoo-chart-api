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

        # Ensure data columns are 1D Series
        close = data['Close'].squeeze()
        high = data['High'].squeeze()
        low = data['Low'].squeeze()
        volume = data['Volume'].squeeze()

        # Indicators
        data['rsi'] = ta.momentum.RSIIndicator(close=close).rsi()
        macd = ta.trend.MACD(close)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()

        # Williams Alligator (overlay on price)
        data['jaw'] = close.ewm(span=13, adjust=False).mean().shift(8)
        data['teeth'] = close.ewm(span=8, adjust=False).mean().shift(5)
        data['lips'] = close.ewm(span=5, adjust=False).mean().shift(3)

        # Plotting
        fig, axs = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

        # 1. Price Only
        axs[0].plot(data.index, close, label='Close Price', color='blue')
        axs[0].set_title(f'{ticker} - Last 24h Close Price')
        axs[0].yaxis.tick_right()
        axs[0].legend(loc='upper left')

        # 2. Williams Alligator
        axs[1].plot(data.index, data['jaw'], label='Jaw (13,8)', color='navy', linewidth=1)
        axs[1].plot(data.index, data['teeth'], label='Teeth (8,5)', color='red', linewidth=1)
        axs[1].plot(data.index, data['lips'], label='Lips (5,3)', color='green', linewidth=1)
        axs[1].set_title('Williams Alligator')
        axs[1].yaxis.tick_right()
        axs[1].legend(loc='upper left')

        # 3. RSI
        axs[2].plot(data.index, data['rsi'], label='RSI', color='purple')
        axs[2].axhline(70, color='red', linestyle='--', linewidth=0.8)
        axs[2].axhline(30, color='green', linestyle='--', linewidth=0.8)
        axs[2].set_title('Relative Strength Index')
        axs[2].yaxis.tick_right()
        axs[2].legend(loc='upper left')

        # 4. MACD with Histogram
        axs[3].plot(data.index, data['macd'], label='MACD', color='black')
        axs[3].plot(data.index, data['macd_signal'], label='Signal', color='orange')
        axs[3].bar(data.index, data['macd_diff'], label='Histogram', color='gray', alpha=0.5)
        axs[3].set_title('MACD')
        axs[3].yaxis.tick_right()
        axs[3].legend(loc='upper left')

        # 5. Volume
        axs[4].bar(data.index, volume, color='slategray')
        axs[4].set_title('Volume')
        axs[4].yaxis.tick_right()

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
