import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# 🔹 Prompt for dataset path
print("📂 Enter path to your CSV dataset (press Enter to use 'AmesHousing.csv'):")
dataset_path = input("Dataset path: ").strip()
if not dataset_path:
    dataset_path = "AmesHousing.csv"

# 🔹 Load dataset
if not os.path.exists(dataset_path):
    print(f"❌ File not found: {dataset_path}")
    exit()

try:
    data = pd.read_csv(dataset_path)
    print(f"✅ Loaded dataset: {dataset_path}")
    print("🔎 First 5 rows:\n", data.head())
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit()

# 🔹 Required columns for the model
required_cols = ['Neighborhood', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF',
                 '1st Flr SF', 'Year Built', 'Full Bath', 'TotRms AbvGrd', 'Fireplaces',
                 'Mas Vnr Area', 'SalePrice']

missing = [col for col in required_cols if col not in data.columns]
if missing:
    print(f"❌ Missing required columns: {missing}")
    exit()

# 🔹 Clean data
data = data[required_cols].dropna()
print(f"✅ Data cleaned: {len(data)} rows remaining")

# 🔹 One-hot encode Neighborhood
data_encoded = pd.get_dummies(data, columns=['Neighborhood'])
X = data_encoded.drop('SalePrice', axis=1)
y = data_encoded['SalePrice']

# 🔹 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train model
print("🔄 Training LightGBM model...")
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# 🔹 Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Model trained successfully.")
print(f"📊 RMSE: ${rmse:,.2f}")
print(f"📈 R² Score: {r2:.4f}")

# 🔹 Save model and columns
joblib.dump(model, "lightgbm_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
print("💾 Saved: 'lightgbm_model.pkl' and 'model_columns.pkl'")
