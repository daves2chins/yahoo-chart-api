from flask import Flask, request, send_file, jsonify
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import io
import datetime

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

        # Calculate indicators
        data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        data['adx'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()

        # Plotting
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        axs[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axs[0].set_title(f'{ticker} - Last 24h Price')
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
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
