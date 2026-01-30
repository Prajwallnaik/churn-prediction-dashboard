import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

# Load data
df = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\churn\data\customer_subscription_churn_usage_patterns.csv")

# Drop unnecessary columns
df = df.drop(columns=["user_id", "signup_date"], errors="ignore")

# Encode categorical variables
le = LabelEncoder()
df["plan_type"] = le.fit_transform(df["plan_type"])
df["churn"] = le.fit_transform(df["churn"])

# Split features & target
X = df.drop("churn", axis=1)
y = df["churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base model
xgb = XGBClassifier(eval_metric="logloss")

# Grid parameters
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1],
}

# GridSearch
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Train
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test)

# Results
print("Best Parameters:\n", grid_search.best_params_)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
pickle.dump(best_model, open("xgb_churn_model.pkl", "wb"))