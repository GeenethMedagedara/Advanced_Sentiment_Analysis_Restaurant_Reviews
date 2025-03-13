import seaborn as sns
import matplotlib.pyplot as plt
import re

class DataCleaningStrategy:
    def execute(self, df):
        pass

class PreviewDataValues(DataCleaningStrategy):
    def execute(self, df):
        """
        Display the head() info() and describe() the dataset
        """
        return df.head(), df.info(), df.describe()

class DuplicateValues(DataCleaningStrategy):
    def execute(self, df):
        """
        Display the count of duplicate rows in the dataset
        """
        return df.duplicated().sum()

class DropDuplicateValues(DataCleaningStrategy):
    def execute(self, df):
        """
        Drop duplicate rows from the dataset.
        """
        return df.drop_duplicates()

class CountMissingValues(DataCleaningStrategy):
    def execute(self, df):
        """
        Count the number of missing values in each column of the DataFrame.
        """
        return df.isnull().sum()

class VisualizeMissingValues(DataCleaningStrategy):
    def execute(self, df):
        """
        Visualize the missing values using a heatmap.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Heatmap")
        plt.show()

class DropMissingValues(DataCleaningStrategy):
    def execute(self, df):
        """
        Drop rows with missing values.
        """
        return df.dropna()
    
class TextCleaning(DataCleaningStrategy):
    def execute(self, df):
        """
        Clean the text in a manner so that it is ready for tokenization
        """
        df['text'] = df['text'].apply(clean_text)
        return df

def clean_text(text):
    text = re.sub(r'<span class=" raw__09f24__T4Ezm" lang="en">', '', text)
    text = re.sub(r'</span>', '', text)
    text = re.sub(r'<br>', '', text)
    text = re.sub(r'\\', '', text)
    return text

class TextRatingCleaning(DataCleaningStrategy):
    def execute(self, df):
        """
        Clean the text in rating in a manner so that it is ready for visualization
        """
        df['rating'] = df['rating'].astype(str).str.replace(r'star rating', '', regex=True)
        return df
    
class WrongTextInDataframeCleaning(DataCleaningStrategy):
    def execute(self, df):
        """
        Clean the text in the dataset that is wrong a manner so that it is ready for tokenization
        """
        df = df[df['location'] != 'Location']
        df = df[df['location'] != 'location']
        df = df[df['text'] != '<span class=" raw__09f24__T4Ezm">299 Sussex St']
        return df