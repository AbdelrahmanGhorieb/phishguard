import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("Phishing_Legitimate_full.csv")

print(df.head())
print(df.shape)

# Check missing values
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

# Features and target
X = df.drop("CLASS_LABEL", axis=1)
y = df["CLASS_LABEL"]

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Balance classes
smote = SMOTE(random_state=42)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save split files
pd.DataFrame(X_train_scaled).to_csv("train.csv", index=False)
pd.DataFrame(X_val_scaled).to_csv("validation.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("test.csv", index=False)

print("Done successfully")