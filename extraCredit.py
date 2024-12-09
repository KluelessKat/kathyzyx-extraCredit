import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
# Import additional classifiers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv("train.csv")  # Replace with your dataset path
test_data = pd.read_csv("test.csv")

from sklearn.cluster import KMeans
from datetime import timedelta

# Data preprocessing
def preprocess_data(data):
    # Columns to keep and their operations
    label_cols = ['category', 'gender', 'city', 'state', 'cc_num', 'merchant', 'zip', 'job', 'trans_date']
    numeric_cols = ['amt', 'city_pop', 'dob']
    
    # Encode categorical features
    for col in label_cols:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    
    # Convert dob to age
    current_year = pd.Timestamp.now().year
    data['dob'] = current_year - pd.to_datetime(data['dob'], errors='coerce').dt.year
    data['trans_hour'] = data['trans_time'].str.split(':').str[0].astype(int)
    data['delta_lat'] = data['lat'] - data['merch_lat']
    data['delta_long'] = data['long'] - data['merch_long']
#     data['transactions_last_7d'] = (
#     data.groupby('cc_num')['trans_date']
#     .transform(lambda x: x.rolling('7D', on=x).count())
#     .fillna(0)
# )
     # Additional feature engineering
    # Distance between cardholder and merchant
    data['distance'] = np.sqrt((data['lat'] - data['merch_lat'])**2 + 
                                (data['long'] - data['merch_long'])**2)
    
    # Time-based features
    data['trans_day'] = pd.to_datetime(data['trans_date']).dt.day_name()
    data['is_weekend'] = pd.to_datetime(data['trans_date']).dt.dayofweek.isin([5,6]).astype(int)
    
    # Transaction frequency features
    data['transaction_frequency'] = data.groupby('cc_num')['trans_num'].transform('count')
    
    # Amount anomaly detection
    data['amount_z_score'] = (data['amt'] - data['amt'].mean()) / data['amt'].std()
    
    # Unusual transaction time
    data['is_unusual_hour'] = ((data['trans_hour'] < 6) | (data['trans_hour'] > 22)).astype(int)

    
    # Standardize numerical features
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Drop all other columns
    data = data[label_cols + numeric_cols + ['trans_hour']]
    
    return data



X = preprocess_data(data)
test_data_X = preprocess_data(test_data)
# import pdb; pdb.set_trace()
# Split the data into features and target
y = data['is_fraud']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


# SMOTE for numerical features
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train LightGBM on numerical features
# Train LightGBM on numerical features
# Create a more sophisticated ensemble
def create_advanced_ensemble():
    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)

    # Train XGBoost on numerical features
    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Train RandomForest on numerical features
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        
    # Stacking approach
    final_estimator = LogisticRegression(class_weight='balanced')
        
    # Voting Classifier with Stacking
    ensemble = VotingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft',
        weights=[0.4, 0.3, 0.2, 0.1],
        flatten_transform=True
    )

    return ensemble

# Updated training approach
ensemble_model = create_advanced_ensemble()
ensemble_model.fit(X_train, y_train)

# Predictions
y_pred = ensemble_model.predict(X_test)


# # Predict probabilities for LightGBM, XGBoost, and RandomForest on the test set
# lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
# xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
# rf_probs = rf_model.predict_proba(X_test)[:, 1]

# # Assign weights based on model performance
# lgb_weight = 0.5  # LightGBM weight
# xgb_weight = 0.3  # XGBoost weight
# rf_weight = 0.2   # RandomForest weight

# combined_probs = (lgb_weight * lgb_probs + xgb_weight * xgb_probs + rf_weight * rf_probs)

# # Combine predictions with soft voting (average probabilities)
# y_pred = (combined_probs >= 0.5).astype(int)

# Evaluate the combined model
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

# # Predict on the test data (numerical features only)
# lgb_test_probs = lgb_model.predict_proba(test_data_X)[:, 1]
# xgb_test_probs = xgb_model.predict_proba(test_data_X)[:, 1]
# rf_test_probs = rf_model.predict_proba(test_data_X)[:, 1]

# # Combine predictions with soft voting
# test_combined_probs = (lgb_test_probs + xgb_test_probs + rf_test_probs) / 3
# test_predictions = (test_combined_probs >= 0.5).astype(int)

# Predict on the original test dataset (test_data_X)
test_predictions = ensemble_model.predict(test_data_X)

# Save predictions in a submission file
submission = pd.DataFrame({
    'id': test_data['id'],  # Ensure the 'id' column is present in test_data
    'is_fraud': test_predictions  # Predicted values for the test dataset
})

# Save the DataFrame to a CSV file
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)

print(f"Submission file saved as {submission_file}")