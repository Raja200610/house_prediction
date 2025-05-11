import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained LightGBM model and columns
model = joblib.load("lightgbm_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Page layout
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Sidebar - Upload Dataset
st.sidebar.header("📂 Upload Dataset")
user_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

@st.cache_data
def load_data():
    return pd.read_csv("AmesHousing.csv")

# Load dataset
market_data = pd.read_csv(user_file) if user_file else load_data()

# Rename 'Location' to 'Neighborhood' if needed
if "Location" in market_data.columns:
    market_data = market_data.rename(columns={"Location": "Neighborhood"})

# Determine price column
price_column = "Price" if "Price" in market_data.columns else "SalePrice"

# Validate columns
if 'Neighborhood' not in market_data.columns:
    st.error("❌ 'Neighborhood' column is required in the dataset.")
    st.stop()

# Neighborhood list
neighborhoods = sorted(market_data['Neighborhood'].dropna().unique())

# App Title
st.title("🏠 House Price Predictor")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Insights", "🧠 SHAP Explainability"])

with tab1:
    st.header("📝 Enter House Features")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            neighborhood = st.selectbox("Neighborhood", neighborhoods)
            overall_qual = st.slider('Overall Quality (1-10)', 1, 10, 5)
            gr_liv_area = st.slider('Ground Living Area (sqft)', 500, 5000, 1500)
            garage_cars = st.slider('Garage Capacity (cars)', 0, 4, 2)
            total_bsmt_sf = st.slider('Total Basement Area (sqft)', 0, 3000, 800)
            first_flr_sf = st.slider('1st Floor Area (sqft)', 500, 3000, 1200)

        with col2:
            year_built = st.slider('Year Built', 1900, 2023, 2000)
            full_bath = st.slider('Full Bathrooms', 0, 4, 2)
            tot_rms_abv = st.slider('Total Rooms Above Ground', 2, 12, 6)
            fireplaces = st.slider('Number of Fireplaces', 0, 3, 1)
            mas_vnr_area = st.slider('Masonry Veneer Area (sqft)', 0, 1000, 100)

        submitted = st.form_submit_button("📈 Predict Price")

    if submitted:
        # Prepare input
        input_data = {
            'Neighborhood': neighborhood,
            'Overall Qual': overall_qual,
            'Gr Liv Area': gr_liv_area,
            'Garage Cars': garage_cars,
            'Total Bsmt SF': total_bsmt_sf,
            '1st Flr SF': first_flr_sf,
            'Year Built': year_built,
            'Full Bath': full_bath,
            'TotRms AbvGrd': tot_rms_abv,
            'Fireplaces': fireplaces,
            'Mas Vnr Area': mas_vnr_area,
        }

        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df)

        # Align columns with training features
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]  # correct order
        df = df.loc[:, model_columns]  # drop any extra columns if present

        prediction = model.predict(df)[0]
        st.success(f"💰 Predicted House Price: ${prediction:,.2f}")

        # Download
        df['Predicted Price'] = prediction
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Prediction", csv, "prediction.csv", "text/csv")

        # Market average
        if price_column in market_data.columns:
            avg_price = market_data[price_column].mean()
            st.info(f"📊 Average Market Price: ${avg_price:,.2f}")

with tab2:
    st.header("📊 Interactive Visualizations")
    selected_cities = st.multiselect("Filter Neighborhoods", neighborhoods, default=neighborhoods)
    filtered_data = market_data[market_data['Neighborhood'].isin(selected_cities)]

    # Price distribution
    st.subheader("📉 Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_data[price_column], kde=True, ax=ax1)
    st.pyplot(fig1)

    # Area vs Price
    if 'Area' in filtered_data.columns:
        st.subheader("📍 Area vs Price")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='Area', y=price_column, data=filtered_data, hue='Neighborhood', ax=ax2)
        st.pyplot(fig2)

    # Average price by neighborhood
    st.subheader("🏙️ Average Price by Neighborhood")
    avg_price_by_n = filtered_data.groupby('Neighborhood')[price_column].mean().sort_values()
    fig3, ax3 = plt.subplots()
    avg_price_by_n.plot(kind='barh', ax=ax3)
    st.pyplot(fig3)

with tab3:
    st.header("🧠 SHAP Feature Impact")
    try:
        input_data = df if 'df' in locals() else None
        if input_data is not None:
            input_data = input_data[model_columns]  # Ensure SHAP uses correct features
            explainer = shap.Explainer(model)
            shap_values = explainer(input_data)
            fig4, ax4 = plt.subplots()
            shap.plots.bar(shap_values[0], max_display=10, show=False)
            st.pyplot(fig4)
        else:
            st.info("ℹ️ Make a prediction first to view SHAP explanations.")
    except Exception as e:
        st.warning(f"SHAP not available: {e}")
