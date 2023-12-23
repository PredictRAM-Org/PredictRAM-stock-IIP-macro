import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load stock data
stock_data = pd.read_json("stock.json")

# Load macro data
macro_data = pd.read_excel("macro_data.xlsx")

# Load IIP data
iip_data = pd.read_csv("IIP.csv")

# Ensure 'Date' column is in datetime format for proper merging
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
macro_data['Reporting Date'] = pd.to_datetime(macro_data['Reporting Date'])
iip_data['Date'] = pd.to_datetime(iip_data['Date'])

# Combine data based on common date columns
combined_data = pd.merge(stock_data, macro_data, how="inner", left_on="Date", right_on="Reporting Date")
combined_data = pd.merge(combined_data, iip_data, how="inner", on="Date")

# User interface
st.title("Stock Analysis App")

# Allow user to select columns from macro_data.xlsx and IIP.csv
selected_macro_columns = st.multiselect("Select columns from macro_data.xlsx", macro_data.columns)
selected_iip_columns = st.multiselect("Select columns from IIP.csv", iip_data.columns)

# Train models and make predictions
if st.button("Train Models and Predict"):
    # Create features and target variables
    X = combined_data[selected_macro_columns + selected_iip_columns]
    y = combined_data["Total Revenue/Income"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    linear_reg = LinearRegression()
    random_forest_reg = RandomForestRegressor()
    gradient_boosting_reg = GradientBoostingRegressor()

    linear_reg.fit(X_train, y_train)
    random_forest_reg.fit(X_train, y_train)
    gradient_boosting_reg.fit(X_train, y_train)

    # Make predictions
    linear_reg_preds = linear_reg.predict(X_test)
    random_forest_preds = random_forest_reg.predict(X_test)
    gradient_boosting_preds = gradient_boosting_reg.predict(X_test)

    # Evaluate models
    st.subheader("Model Evaluation")
    st.write("Linear Regression MSE:", mean_squared_error(y_test, linear_reg_preds))
    st.write("Random Forest Regression MSE:", mean_squared_error(y_test, random_forest_preds))
    st.write("Gradient Boosting Regression MSE:", mean_squared_error(y_test, gradient_boosting_preds))

    # Plot predictions vs actual values
    st.subheader("Predictions vs Actual Values")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, linear_reg_preds, label="Linear Regression")
    plt.scatter(y_test, random_forest_preds, label="Random Forest Regression")
    plt.scatter(y_test, gradient_boosting_preds, label="Gradient Boosting Regression")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    st.pyplot()

# User input for upcoming values
st.sidebar.title("Input Upcoming Values")
upcoming_macro_data = {}
upcoming_iip_data = {}

for column in selected_macro_columns:
    upcoming_macro_data[column] = st.sidebar.number_input(f"Enter upcoming value for {column}", value=0.0)

for column in selected_iip_columns:
    upcoming_iip_data[column] = st.sidebar.number_input(f"Enter upcoming value for {column}", value=0.0)

# Make predictions for upcoming values
upcoming_data = pd.DataFrame({**upcoming_macro_data, **upcoming_iip_data}, index=[0])
upcoming_predictions = {
    "Linear Regression": linear_reg.predict(upcoming_data),
    "Random Forest Regression": random_forest_reg.predict(upcoming_data),
    "Gradient Boosting Regression": gradient_boosting_reg.predict(upcoming_data)
}

# Display upcoming predictions
st.sidebar.subheader("Upcoming Predictions")
for model, prediction in upcoming_predictions.items():
    st.sidebar.write(f"{model}: {prediction[0]}")
