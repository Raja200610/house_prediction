
# 🏠 House Price Prediction App

This project is a **machine learning-based web application** that predicts house prices using the Ames Housing Dataset and a trained **LightGBM regression model**. It includes an interactive UI for predictions, visual analytics, and SHAP-based model explainability.

---

## 📌 Features

- ✅ Accurate price prediction using LightGBM
- ✅ Interactive form to enter house features
- ✅ SHAP explainability to interpret model predictions
- ✅ Price distribution and neighborhood insights
- ✅ CSV upload and downloadable prediction results

---

## 📁 Project Structure

```
├── AmesHousing.csv              # Raw dataset (optional)
├── train_model.py              # Model training script
├── lightgbm_model.pkl          # Trained model (output)
├── model_columns.pkl           # List of features used in the model (output)
├── houseapp.py                 # Streamlit app
├── requirements.txt            # Project dependencies
```

---

## 🚀 Getting Started

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Train the Model (Optional)**
If not using the provided model:
```bash
python train_model.py
```

### 3. **Run the Streamlit App**
```bash
streamlit run houseapp.py
```

---

## 🧠 SHAP Explainability

- SHAP analysis is shown **after a prediction is made**.
- The app **automatically aligns features** to avoid shape mismatch errors.
- Extra columns are removed, and missing ones are added as zero.

---

## 🧩 Custom Input Features

The app uses the following input features:
- Neighborhood (one-hot encoded)
- Overall Qual
- Gr Liv Area
- Garage Cars
- Total Bsmt SF
- 1st Flr SF
- Year Built
- Full Bath
- TotRms AbvGrd
- Fireplaces
- Mas Vnr Area

---

## 📂 Output Files

| File                | Description                                 |
|---------------------|---------------------------------------------|
| `lightgbm_model.pkl` | Trained model for prediction               |
| `model_columns.pkl` | Feature list required for SHAP & alignment  |

---

## 📊 Dataset

- **Name:** Ames Housing Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
- **Type:** Tabular / Regression  
- **Size:** ~2,900 rows, 80+ features

---

## 🛠 Tech Stack

- Python
- LightGBM
- Streamlit
- SHAP
- scikit-learn
- pandas, seaborn, matplotlib

---

## 👨‍💻 Author

Developed by **Rajayogeswaran,Vishnu,Velan,Vetri**  
For academic, demo, and portfolio use
