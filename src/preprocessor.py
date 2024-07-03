import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
This class is used to preprocess the data before training the models.

Attributes:
    data (pd.DataFrame): The dataset to be preprocessed.
'''

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def split_data(self, test_size=0.2, target_column='Outcome'):
            
        # Split the data into training and testing sets
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def handle_outliers(self):

        # Handle outliers to clean the data
        if self.data is not None:
            numeric_columns = self.data.select_dtypes(include='number').columns.tolist()
            total_outliers = 0
            for col in numeric_columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                total_outliers += outliers_count
                
                if outliers_count > 0:
                    print(f"Column '{col}' has {outliers_count} outliers.")
                    # Eliminar outliers del dataset
                    self.data = self.data[~outliers_mask]
                else:
                    print(f"Column '{col}' has no outliers.")
                
            print(f"{total_outliers} outliers removed successfully.")
        else:
            print("No dataset loaded. Please load the dataset first.")

    def handle_missing_values(self, strategy='mean'):

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy=strategy)
        self.data[self.data.columns] = imputer.fit_transform(self.data)
        print("Missing values handled successfully.")
        return self.data

    def normalize_data(self, target_column, epsilon=1e-10):

        # Normalize the input data using StandardScaler
        features = self.data.drop(columns=[target_column])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        scaled_features = (features - scaler.mean_) / (scaler.scale_ + epsilon)
        self.data[features.columns] = scaled_features
        return self.data

    def encode_categorical_variables(self):

        # Encode the categorical variables using one-hot encoding
        self.data = pd.get_dummies(self.data, drop_first=True)
        print("Categorical variables encoded successfully.")
        return self.data
    
    def remove_highly_correlated_features(self, threshold=0.9):

        # Remove highly correlated features
        corr_matrix = self.data.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.data = self.data.drop(columns=to_drop)
        return self.data
    
    def convert_target_to_categorical(self, target_column):

        # Convert the target column to categorical
        self.data[target_column] = self.data[target_column].astype('category')
        print(f"Target column '{target_column}' converted to categorical.")
        return self.data