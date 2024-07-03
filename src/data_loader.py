import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

''' 
This class is used to load a dataset from a file path and perform some basic operations on it.

Attributes:
    file_path (str): The path to the dataset file.
    data (pd.DataFrame): The dataset loaded from the file path.

'''

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None 

    def load_data(self):

        # Load the dataset from the file path
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Correcly loaded dataset from {self.file_path}")
        except Exception as e:
            print(f"Error loading dataset from {self.file_path}")
            print(e)

    def get_data(self):

        # Return the data
        return self.data

    def print_data(self, n=5):

        # Print the first n rows of the dataset
        print(self.data.head(n))

    def show_basic_info(self):

        # Print basic information about the dataset
        print(f"Shape of the dataset: {self.data.shape}")
        print(f"Columns: {self.data.columns}")
        print(self.data.info())

    def plot_histogram(self, column):

        #Plot an histogram of just one column
        if self.data is not None and column in self.data.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.data[column], kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
        else:
            print(f"Column '{column}' not found in the dataset.")

    def plot_boxplot_all(self, ignore_columns=[]):
        if self.data is not None:

            # Ignore the columns that are not numeric and the ones in the ignore_columns list
            filtered_columns = self.data.select_dtypes(include='number').columns.tolist()
            filtered_columns = [col for col in filtered_columns if col not in ignore_columns]
            
            # Plot boxplot
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=self.data[filtered_columns])
            plt.title('Boxplots of all columns')
            plt.xticks(rotation=45)
            plt.xlabel('Variables')
            plt.ylabel('Value')
            plt.ylim(0, 360)
            plt.tight_layout()
            plt.show()
        else:
            print("No dataset loaded. Please load the dataset first.")

    
