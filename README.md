![](https://github.com/Sarthakkk16/Stock-EDA-Real-Time-Prediction-App/blob/main/Blue%20and%20Green%20Gradient%20Modern%20Stock%20Market%20Presentation.png)
# Stock-EDA-Real-Time-Prediction-App
An interactive web application built with Streamlit for real-time stock analysis and next-day closing price prediction.

# This application allows users to:

Download Real-Time Financial Data for any ticker using yfinance.

Perform Exploratory Data Analysis (EDA): Visualize closing price trends, calculate moving averages (MA20, MA50), and analyze returns distribution.

Feature Engineering: Create time-series features essential for forecasting, including lag features, daily returns, and rolling mean/standard deviation.

Train a Machine Learning Model: Implement a Random Forest Regressor with a TimeSeriesSplit cross-validation strategy to robustly train the model on sequential data.

Predict Next-Day Close: Generate a real-time, next-day price forecast based on the latest available data, demonstrating the model's practical application.

# Features
Data Acquisition: Uses the yfinance library to fetch historical stock data based on user input (ticker and start date).

EDA & Visualization:

Plots historical closing prices.

Calculates and visualizes 20-day and 50-day Moving Averages (MA).

Displays a histogram of daily percentage returns.

Feature Engineering (create_features function):

Creates lagged features (lag1, lag2, lag3) from the 'Close' price.

Calculates daily returns (return_1d).

Generates rolling statistics (roll_mean_3, roll_std_3).

Defines the target variable (target) as the next day's closing price.

Model: Uses a RandomForestRegressor for prediction.

Validation: Employs TimeSeriesSplit (n_splits=3) for proper backtesting and evaluation, reporting the average RMSE.

Real-Time Prediction: Uses the last available data point's engineered features to generate a prediction for the immediate next trading day.

Visualization of Results: Plots the actual vs. predicted values for the final test fold to demonstrate model performance.

# Technologies Used
Frontend/Deployment: streamlit

Data Handling: pandas, numpy

Financial Data: yfinance

Visualization: matplotlib

Machine Learning: sklearn (RandomForestRegressor, TimeSeriesSplit, metrics)
