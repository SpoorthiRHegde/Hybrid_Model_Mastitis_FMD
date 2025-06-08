import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load the dataset
dataframe = pd.read_csv('C:/Users/spoor/OneDrive/Desktop/Major/Mastitis_Detection/Dataset/dataset_95.csv')
data = dataframe.drop_duplicates(keep=False)

# Define features and target
X = data[['Temperature', 'Hardness', 'Pain', 'Milk Yield', 'Milk Color']]
y = data['Mastitis']

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
svc_model = SVC(kernel='rbf', probability=True, random_state=42)
model = VotingClassifier(
    estimators=[
        ('RandomForest', rf_model),
        ('GradientBoosting', gb_model),
        ('SVM', svc_model)
    ],
    voting='soft'
)

# Train the ensemble model
model.fit(X_train, y_train)

# Ensure correct directory for saving models
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
os.makedirs(output_dir, exist_ok=True)

# Save model and scaler
joblib.dump(model, os.path.join(output_dir, 'mastitis_text_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'mastitis_scaler.pkl'))

# Prediction function
def predict_text(features):
    import pandas as pd
    columns = ['Temperature', 'Hardness', 'Pain', 'Milk Yield', 'Milk Color']
    input_df = pd.DataFrame([features], columns=columns)
    features_scaled = scaler.transform(input_df)
    prediction = model.predict(features_scaled)
    return "Mastitis Detected" if prediction[0] == 1 else "No Mastitis"
