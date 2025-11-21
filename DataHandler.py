import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pandas.api.types import is_numeric_dtype
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import OneHotEncoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
creating a class to import the data
and tranfer it to a dictionary of data frames 
we have only one csv file in our data set so we will work directly at him as data frame
"""

class ImportData:
    
    #import the data from kaggle to a directory
    @staticmethod
    def import_data(dir_path = "customers_data", data_path = "ulrikthygepedersen/online-retail-dataset"):
        path = pathlib.Path(dir_path)
        if path.exists():
            logging.warning("directory alreadt exist")
            return
        else:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset=data_path, path=dir_path, unzip=True)
            logging.info("successfully imported the data to the directory")

    #transfer all the data csv file to data frame
    @staticmethod
    def to_df(dir_path = "customers_data")->pd.DataFrame:
        data_path = pathlib.Path(dir_path)
        for file in data_path.iterdir():
            try:
                df = pd.read_csv(file)
                return df
            except Exception as e:
                logging.error(f"error has occured {e} while transfering the file")

"""
create a class that analyz the data and help us
to understand it
"""

class DataAnalysis:
    
    #create constructor for the class
    def __init__(self, df:pd.DataFrame):
        self.df = df

    #get information about the data frame
    def get_info(self):
        self.df.info()

    #get the top and bbuttom info of the data frame
    def get_edges_info(self):
        logging.info(self.df.head(10))
        logging.info(self.df.tail(10))

    #count how much missing vals we have in each column
    def get_miss_vals_amount(self):
        for col in self.df:
            miss_val = self.df[col].isnull().sum()
            logging.info(f"in coloumn {col} there are {miss_val} missing values")
    
    #get statistical information about the numerical data set
    def get_numeric_statistical_info(self):
        logging.info(self.df.describe())

    #get info about the categorical data
    def get_categoric_unique_info(self):
        for col in self.df:
            unique = self.df[col].nunique()
            logging.info(f"in col {col} there are {unique} unique items")

    #create histogram of the data
    def features_hist(self, feature):
        plt.hist(self.df[feature], bins=30, edgecolor='black')
        plt.xlabel(feature)
        plt.ylabel("frequency")
        plt.title(feature + "histogram")
        plt.show()

    #look at the top 10 countries of the dataset
    def top_ten_countries_graph(self):
        sorted_df = self.df["Country"].sort_values(ascending=False) 
        countries_df = sorted_df.head(10)
        countries_df.plot(kind="bar",edgecolor='black')
        plt.title("top 10 countries")
        plt.xlabel("countries")
        plt.ylabel("orders")
        plt.show()

    #create heat map to check the correlation between all the features
    def featrues_correlation(self):
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        heatmap = sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='2f',
            cmap="coolwarm",
            linewidths=.5,
            cbar=True
        )
        plt.show()

    #distribution Detection by creating a box plot
    def distribution_detection(self, feature1, feature2):
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x=feature1,
            y=feature2,
            data=self.df,
            palette="coolwarm"
        )
        plt.title(f"distribution of {feature1} by {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    #outlier Detection by creating a box plot
    def outlier_detection(self, feature):
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            y=feature,
            data=self.df,
            palette="coolwarm"
        )
        plt.title(f"show the outlier of {feature}")
        plt.xlabel(feature)
        plt.show()

"""
create class that handle with all the data cleaning functions 
"""

class DataCleaning:

    #create constructor to the class that will contain the data frame
    def __init__(self, df:pd.DataFrame):
        self.df = df

    #handling with missing data. fill with the median
    def handle_missing_vals(self):
        for col in self.df:
            if is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)            
            else:
                self.df[col] = self.df[col].fillna("Unknown")
        return self.df
    
    #outlier capping, minimize the effect of outliers on the data
    def cap_outliers(self, factor=1.5):
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (factor * IQR)
            upper_bound = Q3 + (factor * IQR)
            self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)

        logging.info(f"Outliers capped successfully using factor {factor}")
        return self.df 
    
    #make sure each col have the right dtype
    def correct_data_types(self):
        if 'InvoiceDate' in self.df.columns:
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce')

        for col in self.df.select_dtypes(include='object').columns:
            cleaned_col = self.df[col].astype(str).str.replace('$', '').str.replace(',', '')
            self.df[col] = pd.to_numeric(cleaned_col, errors='ignore')

        logging.info("Data types corrected successfully")
        return self.df
    
    #unit same categorical values written in different ways
    def clean_categorical_text(self):
        categorical_cols = self.df.select_dtypes(include='object').columns

        for col in categorical_cols:
            self.df[col] = self.df[col].astype(str).str.strip().str.lower()

        if 'Country' in self.df.columns:
            country_corrections = {
                'u.k.': 'united kingdom',
                'uk': 'united kingdom',
                'usa': 'united states',
                'us': 'united states'
            }
            self.df['Country'] = self.df['Country'].replace(country_corrections)

        logging.info("Categorical text cleaned successfully")
        return self.df
    
    #remove duplicated rows
    def drop_duplicates(self):
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(keep='first')
        dropped_count = initial_rows - len(self.df)
        
        if dropped_count > 0:
            logging.info(f"Removed {dropped_count} duplicate rows")
        else:
            logging.info("No duplicate rows found")

        return self.df
    
    #create pipeline for data cleaning class
    def run_pipeline(self):
        self.drop_duplicates()
        self.correct_data_types()
        self.clean_categorical_text()
        self.handle_missing_vals()
        self.cap_outliers()
        
        logging.info("Pipeline Completed Successfully")
        return self.df
    

class DataPreProcessing:

    #create constructor to the class
    def __init__(self, df:pd.DataFrame):
        self.df = df

    #create new coloumns and add them to our data frame
    def create_rfm_features(self):
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']
        max_date = self.df['InvoiceDate'].max()
        reference_date = max_date + pd.Timedelta(days=1)

        rfm = self.df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days, 
            'InvoiceNo': 'nunique',                                   
            'TotalPrice': 'sum'                                       
        })

        rfm.rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'TotalPrice': 'Monetary'
        }, inplace=True)

        logging.info(f"RFM features created successfully. Shape: {rfm.shape}")

        return rfm

    """
    Handling positive skewness is robust in numerical features (especially RFM features). 
    The goal of the transformation is to create a more normal-like distribution, 
    which improves the performance of K-Means distance-based clustering algorithms.
    """
    def apply_log_transformation(self, features_list):
        for feature in features_list:
            if (self.df[feature] < 0).any():
                logging.warning(f"Column {feature} contains negative values. Log transform might fail.")
            
            self.df['log_' + feature] = np.log1p(self.df[feature])

        logging.info(f"Log transformation applied to: {features_list}")
        return self.df

    """
    Convert nominal categorical features (Nominal - without internal order, 
    such as gender, country) to binary format (0 or 1) using One-Hot Encoding.
    Clustering algorithms cannot work directly with text
    """
    def encode_nominal_features(self):
        categorical_cols = ["Country"]