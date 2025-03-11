"""
Finding Survival Markers in Lower Grade Glioma (LGG) and Glioblastoma Multiforme (GBM)seed
"""
# Import libraries
import flexynesis as flx
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path

# Set the seed for reproducibility
os.environ["OMP_NUM_THREADS"] = "1"

# Define the folders
DATA_FOLDER = "data/"
DATA_FL_UNZIP = "lgggbm_tcga_pub_processed/"
FIGURES_FOLDER = "figures/"


def pipeline():
    """
    """

    def create_folders(data_folder: Path, data_unzip_fl: Path, figures_fl: Path):
        """
        Create the folders
        """
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        if not os.path.exists(data_unzip_fl):
            os.mkdir(data_unzip_fl)
        if not os.path.exists(figures_fl):
            os.mkdir(figures_fl)

    def downloda_data():
        """
        Download the data
        """
        os.system("bash download_data.sh")
    
    def data_importer():
        """
        Import the data
        """
        # Define the data importer
        data_importer = flx.DataImporter(path ='lgggbm_tcga_pub_processed', 
                                        data_types = ['mut', 'cna'], log_transform=False, 
                                        concatenate=False, top_percentile=10, 
                                        min_features=1000, correlation_threshold=0.8, 
                                       variance_threshold=0.5)
        
        # Define the train and the test data
        train_data, test_data = data_importer.import_data()

        return train_data, test_data
    
    

    # Create the folders
    create_folders(data_folder=DATA_FOLDER, data_unzip_fl=DATA_FL_UNZIP, figures_fl=FIGURES_FOLDER)
    # Download the data
    downloda_data()
    # Import the data and define the test and train data
    train_data, test_data = data_importer()

    
if __name__=="__main__":
    pipeline()
