# import numpy as np
# import pandas as pd
# import yfinance as yf
# import matplotlib.pyplot as plt
# from flask import Flask, render_template, request
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# import io
# import base64

# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/contact', methods=['GET', 'POST'])
# def contact():
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         message = request.form['message']
#         return render_template('contact.html', success="Your message has been sent!")

#     return render_template('contact.html')

# @app.route('/help')
# def help():
#     return render_template('help.html')
    

# @app.route('/feedback', methods=['GET', 'POST'])
# def feedback():
#     if request.method == 'POST':
#         feedback_text = request.form['feedback']
#         return render_template('feedback.html', success="Thank you for your feedback!")

#     return render_template('feedback.html')

# @app.route('/companyinfo')
# def companyinfo():
#     return render_template('companyinfo.html')

# if __name__ == '__main__':
#     app.run(debug=True)


# # Function to fetch stock data
# def fetch_stock_data(start_date, end_date):
   
#     data = yf.download('TATAMOTORS.NS', start=start_date, end=end_date)
#     return data
#     # print(data)

# # Preprocess data
# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data[['Close']])
#     return scaled_data, scaler

# # Create sequences for LSTM model
# def create_sequences(data, time_steps=60):
#     X, Y = [], []
#     for i in range(time_steps, len(data)):
#         X.append(data[i-time_steps:i, 0])
#         Y.append(data[i, 0])
#     return np.array(X), np.array(Y)

# # Train LSTM model
# def train_model(X_train, Y_train):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(25),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)
#     return model

# # Predict stock price
# def predict_stock(model, data, scaler):
#     last_60_days = data[-60:].reshape(1, -1, 1)
#     predicted_price = model.predict(last_60_days)
#     return scaler.inverse_transform(predicted_price)[0][0]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     start_date = request.form['start_date']
#     end_date = request.form['end_date']

#     try:
#         # Fetch data based on user input
#         data = fetch_stock_data(start_date, end_date)

#         if data.empty:
#             return render_template('index.html', error="No data found for selected dates.")

#         # Preprocess & train model
#         scaled_data, scaler = preprocess_data(data)
#         X_train, Y_train = create_sequences(scaled_data)
#         X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

#         model = train_model(X_train, Y_train)

#         # Predict stock price
#         predicted_price = predict_stock(model, scaled_data, scaler)

#         # Plot stock price history
#         img = io.BytesIO()
#         plt.figure(figsize=(10, 5))
#         plt.plot(data['Close'], label='Stock Price')
#         plt.legend()
#         plt.savefig(img, format='png')
#         img.seek(0)
#         graph_url = base64.b64encode(img.getvalue()).decode()

#         return render_template('result.html', price=round(predicted_price, 2), graph=graph_url)

#     except Exception as e:
#         return render_template('index.html', error=f"Error: {str(e)}")

# if __name__ == '__main__':
#     app.run(debug=True)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import io
import base64

app = Flask(__name__)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Contact Page Route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        return render_template('contact.html', success="Your message has been sent!")
    return render_template('contact.html')

# Help Page Route
@app.route('/help')
def help():
    return render_template('help.html')

# Feedback Page Route
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        return render_template('feedback.html', success="Thank you for your feedback!")
    return render_template('feedback.html')

# Company Info Page Route
@app.route('/companyinfo')
def companyinfo():
    return render_template('companyinfo.html')

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(start_date, end_date):
    data = yf.download('TATAMOTORS.NS', start=start_date, end=end_date)
    return data

# Preprocess Data (Scaling)
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    return scaled_data, scaler

# Create Sequences for LSTM Model
def create_sequences(data, time_steps=60):
    X, Y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

# Train LSTM Model
def train_model(X_train, Y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)
    return model

# Predict Stock Price
def predict_stock(model, data, scaler):
    last_60_days = data[-60:].reshape(1, -1, 1)
    predicted_price = model.predict(last_60_days)
    return scaler.inverse_transform(predicted_price)[0][0]

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    try:
        # Fetch Data
        data = fetch_stock_data(start_date, end_date)
        if data.empty:
            return render_template('index.html', error="No data found for selected dates.")

        # Preprocess Data
        scaled_data, scaler = preprocess_data(data)
        X_train, Y_train = create_sequences(scaled_data)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Train Model
        model = train_model(X_train, Y_train)

        # Predict Stock Price
        predicted_price = predict_stock(model, scaled_data, scaler)

        # Plot Stock Price Graph
        img = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.plot(data['Close'], label='Stock Price')
        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        return render_template('result.html', price=round(predicted_price, 2), graph=graph_url)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
