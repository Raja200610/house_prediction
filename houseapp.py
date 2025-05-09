
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained LightGBM model
model = joblib.load("lightgbm_model.pkl")

# Features used by the model
features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF',
            '1st Flr SF', 'Year Built', 'Full Bath', 'TotRms AbvGrd',
            'Fireplaces', 'Mas Vnr Area']

# Streamlit App Title
st.title("🏠 House Price Predictor")
st.markdown("Enter the house features in the sidebar to predict its price.")

# Sidebar inputs
st.sidebar.header("Input House Features")
input_data = {
    'Overall Qual': st.sidebar.slider('Overall Quality (1-10)', 1, 10, 5),
    'Gr Liv Area': st.sidebar.slider('Ground Living Area (sqft)', 500, 5000, 1500),
    'Garage Cars': st.sidebar.slider('Garage Capacity (cars)', 0, 4, 2),
    'Total Bsmt SF': st.sidebar.slider('Total Basement Area (sqft)', 0, 3000, 800),
    '1st Flr SF': st.sidebar.slider('1st Floor Area (sqft)', 500, 3000, 1200),
    'Year Built': st.sidebar.slider('Year Built', 1900, 2023, 2000),
    'Full Bath': st.sidebar.slider('Full Bathrooms', 0, 4, 2),
    'TotRms AbvGrd': st.sidebar.slider('Total Rooms Above Ground', 2, 12, 6),
    'Fireplaces': st.sidebar.slider('Number of Fireplaces', 0, 3, 1),
    'Mas Vnr Area': st.sidebar.slider('Masonry Veneer Area (sqft)', 0, 1000, 100),
}

# Convert to DataFrame for prediction
df = pd.DataFrame([input_data])

# Predict and display the price
price = model.predict(df)[0]
st.success(f"💰 Predicted House Price: ${price:,.2f}")

# Optional: Load original dataset for market average
@st.cache_data
def load_data():
    return pd.read_csv("AmesHousing.csv")

# Show average market price
try:
    market_data = load_data()
    avg_price = market_data['SalePrice'].mean()
    st.info(f"📊 Average Market Price: ${avg_price:,.2f}")
except:
    st.warning("Could not load dataset to calculate average price.")

# SHAP explainability
st.subheader("🔍 Feature Impact (SHAP Explanation)")
try:
    explainer = shap.Explainer(model)
    shap_values = explainer(df)

    # Plot SHAP values
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
except:
    st.warning("SHAP explanation not available. Ensure model is compatible.")
