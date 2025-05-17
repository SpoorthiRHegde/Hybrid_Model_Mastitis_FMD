#foot and mouth text model
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset for foot and mouth (replace with actual file paths)
foot_df = pd.read_csv('C:/Users/spoor/OneDrive/Desktop/Major/FMD_Detection/Dataset/foot_fmd.csv')
mouth_df = pd.read_csv('C:/Users/spoor/OneDrive/Desktop/Major/FMD_Detection/Dataset/mouth_fmd.csv')

# Features for foot and mouth models
foot_features = ['temperature', 'milk_production', 'lethargy', 'difficulty_in_walking', 
                 'foot_blister', 'foot_swelling', 'hoof_detachment']

mouth_features = ['temperature', 'milk_production', 'lethargy', 'mouth_ulcers', 
                  'mouth_blister', 'salivation', 'nasal_discharge']

# Separate features and target for foot and mouth datasets
X_foot = foot_df[foot_features]
y_foot = foot_df['fmd_status']

X_mouth = mouth_df[mouth_features]
y_mouth = mouth_df['fmd_status']

# Split data into training and testing sets for both foot and mouth datasets
X_train_foot, X_test_foot, y_train_foot, y_test_foot = train_test_split(X_foot, y_foot, test_size=0.2, random_state=42)
X_train_mouth, X_test_mouth, y_train_mouth, y_test_mouth = train_test_split(X_mouth, y_mouth, test_size=0.2, random_state=42)

# Standardize the features for both datasets
scaler_foot = StandardScaler()
X_train_foot_scaled = scaler_foot.fit_transform(X_train_foot)
X_test_foot_scaled = scaler_foot.transform(X_test_foot)

scaler_mouth = StandardScaler()
X_train_mouth_scaled = scaler_mouth.fit_transform(X_train_mouth)
X_test_mouth_scaled = scaler_mouth.transform(X_test_mouth)

# Convert back to DataFrame to preserve feature names
X_train_foot_scaled = pd.DataFrame(X_train_foot_scaled, columns=foot_features)
X_test_foot_scaled = pd.DataFrame(X_test_foot_scaled, columns=foot_features)

X_train_mouth_scaled = pd.DataFrame(X_train_mouth_scaled, columns=mouth_features)
X_test_mouth_scaled = pd.DataFrame(X_test_mouth_scaled, columns=mouth_features)

# Create individual classifiers for both foot and mouth
rf_foot = RandomForestClassifier(n_estimators=100, random_state=42)
svm_foot = SVC(kernel='rbf', probability=True, random_state=42)

rf_mouth = RandomForestClassifier(n_estimators=100, random_state=42)
svm_mouth = SVC(kernel='rbf', probability=True, random_state=42)

# Create soft voting classifiers for both foot and mouth
voting_clf_foot = VotingClassifier(
    estimators=[('rf', rf_foot), ('svm', svm_foot)],
    voting='soft'
)

voting_clf_mouth = VotingClassifier(
    estimators=[('rf', rf_mouth), ('svm', svm_mouth)],
    voting='soft'
)

# Train the classifiers
voting_clf_foot.fit(X_train_foot_scaled, y_train_foot)
voting_clf_mouth.fit(X_train_mouth_scaled, y_train_mouth)

# Make predictions on the test set for both foot and mouth models
y_pred_foot = voting_clf_foot.predict(X_test_foot_scaled)
y_pred_mouth = voting_clf_mouth.predict(X_test_mouth_scaled)

# Calculate metrics for foot model
foot_accuracy = accuracy_score(y_test_foot, y_pred_foot)
foot_precision = precision_score(y_test_foot, y_pred_foot, average='weighted')
foot_recall = recall_score(y_test_foot, y_pred_foot, average='weighted')
foot_conf_matrix = confusion_matrix(y_test_foot, y_pred_foot)

# Calculate metrics for mouth model
mouth_accuracy = accuracy_score(y_test_mouth, y_pred_mouth)
mouth_precision = precision_score(y_test_mouth, y_pred_mouth, average='weighted')
mouth_recall = recall_score(y_test_mouth, y_pred_mouth, average='weighted')
mouth_conf_matrix = confusion_matrix(y_test_mouth, y_pred_mouth)

# Display metrics for foot model
print("\nFoot Model Performance Metrics:")
print(f"Accuracy: {foot_accuracy:.4f}")
print(f"Precision: {foot_precision:.4f}")
print(f"Recall: {foot_recall:.4f}")
print("\nConfusion Matrix:")
print(foot_conf_matrix)

# Display metrics for mouth model
print("\nMouth Model Performance Metrics:")
print(f"Accuracy: {mouth_accuracy:.4f}")
print(f"Precision: {mouth_precision:.4f}")
print(f"Recall: {mouth_recall:.4f}")
print("\nConfusion Matrix:")
print(mouth_conf_matrix)

# Save the models and scalers
joblib.dump(voting_clf_foot, '../models/foot_text_model.pkl')
joblib.dump(scaler_foot, '../models/foot_text_scaler.pkl')

joblib.dump(voting_clf_mouth, '../models/mouth_text_model.pkl')
joblib.dump(scaler_mouth, '../models/mouth_text_scaler.pkl')

print("\nModels and scalers saved successfully!")
