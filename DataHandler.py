import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from kaggle.api.kaggle_api_extended import KaggleApi



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
    pass