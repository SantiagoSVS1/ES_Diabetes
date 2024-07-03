from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

'''
This class is used to evaluate the models on the test set and make predictions on new data.

Attributes:
    models (dict): A dictionary containing the trained models.
    X_test (pd.DataFrame): The test set features.
    y_test (pd.Series): The test set labels.
    normalized (bool): A flag indicating if the input data is normalized.
'''

class ModelEvaluator:
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.normalized = ModelEvaluator.verify_normalized_data(X_test)

    def evaluate(self):

        # Evaluate the models on the test set
        print("\nEvaluating models...")
        results = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            cm = confusion_matrix(self.y_test, y_pred)
            cr = classification_report(self.y_test, y_pred)
            results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm,
                "classification_report": cr
            }
        return results
    
    def predict(self, model_name, input_data):

        # Predict the output for the input data
        if model_name in self.models:
            model = self.models[model_name]
            input_data = pd.DataFrame(input_data, columns=self.X_test.columns)
            if self.normalized:
                print("Normalizing input data...")
                input_data = ModelEvaluator.normalize_data(input_data)
            prediction = model.predict(input_data)
            return prediction
        else:
            raise ValueError(f"Model '{model_name}' not found in the trained models.")

    @staticmethod
    def normalize_data(df, epsilon=1e-10):

        # Normalize the input data
        from sklearn.preprocessing import StandardScaler
        features = df
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        scaled_features = (features - scaler.mean_) / (scaler.scale_ + epsilon)
        df = scaled_features
        return df
    
    @staticmethod
    def verify_normalized_data(X_test):

        # Verify if the input data is normalized by using thresholds
        for col in X_test.columns:
            if X_test[col].max() > 4 or X_test[col].min() < -3:
                return False
        return True
    
    def print_results(self, results):

        # Print the evaluation results
        for model_name, metrics in results.items():
            print(f"Results for {model_name}:")
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"Precision: {metrics['precision']}")
            print(f"Recall: {metrics['recall']}")
            print(f"F1 Score: {metrics['f1_score']}")
            #print("Confusion Matrix:")
            #print(metrics['confusion_matrix'])
            #print("Classification Report:")
            #print(metrics['classification_report'])
            print("\n")