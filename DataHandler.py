import pathlib
import numpy as np
import pandas as pd
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
    def import_data(dir_path = "customers_data", data_path = "datasets/ulrikthygepedersen"):
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
        for file in data_path.iterdir:
            try:
                df = pd.read_csv(file)
                return df
            except Exception as e:
                logging.error(f"error has occured {e} while transfering the file")




