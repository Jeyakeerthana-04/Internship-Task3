import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
# Load dataset
df = pd.read_csv('train.csv')
# ❗ FIXED this line:
df.drop('Loan_ID', axis=1, inplace=True)
# Fill missing values
df.fillna(method='ffill', inplace=True)
# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
# Features and Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Save model
joblib.dump(model, 'loan_model.pkl')
print("✅ Model trained and saved as loan_model.pkl")
