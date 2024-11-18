import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class CKDEnsembleModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.svm_model = SVC(probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def preprocess_data(self, df):
        # Create copies of the data
        X = df.drop('classification', axis=1)
        y = df['classification']
        
        # Handle categorical variables
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        
        for col in categorical_cols:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Convert target to binary
        self.label_encoders['classification'] = LabelEncoder()
        y = self.label_encoders['classification'].fit_transform(y.astype(str))
        
        # Handle missing values
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Scale the features
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    
    def train(self, X, y):
        # Train individual models
        self.rf_model.fit(X, y)
        self.gb_model.fit(X, y)
        self.svm_model.fit(X, y)
        
    def predict_proba(self, X):
        # Get probability predictions from each model
        rf_pred = self.rf_model.predict_proba(X)
        gb_pred = self.gb_model.predict_proba(X)
        svm_pred = self.svm_model.predict_proba(X)
        
        # Average the probabilities
        ensemble_pred = (rf_pred + gb_pred + svm_pred) / 3
        return ensemble_pred
    
    def predict(self, X):
        # Get ensemble probabilities
        ensemble_pred = self.predict_proba(X)
        # Return class with highest probability
        return (ensemble_pred[:, 1] >= 0.5).astype(int)

def load_and_prepare_data(file_path):
    """
    Load and prepare the CKD dataset from a CSV file
    """
    # Load the data
    df = pd.read_csv(Kidney_data.csv)
    
    # Remove the 'id' column if it exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    return df

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    print("\nModel Evaluation Metrics:")
    print("-------------------------")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    # Assuming the data is in a CSV file named 'kidney_disease.csv'
    try:
        df = load_and_prepare_data('kidney_disease.csv')
        
        # Create and train the ensemble model
        model = CKDEnsembleModel()
        X, y = model.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        print("Training the ensemble model...")
        model.train(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
        
        return model
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    main()