"""
Pipeline
"""
import os
from pandas import DataFrame


def pipeline():
    """
    Download data
    Analysis of the data
    """

    def download_data()-> DataFrame:
        """
        Download the data
        """
        # Create folder if not exist
        if not os.path.exists("data/"):
            os.mkdir("data/")
        # Check the if the folder is empty
        if len(os.listdir("data/")) == 0:
            # Download the data
            os.system("bash download_data.sh")
        else:
            # Print that data exist
            print("Data exist")
        
        if not os.path.exists("analysis_fl/"):
            os.mkdir("analysis_fl/")

    
    def analysis():
        """
        """
        # Preprocess file names
        if not os.path.isfile("ccle_vs_gdsc/train/mut.csv") and os.path.isfile("ccle_vs_gdsc/test/mut.csv"):
            # Change files names 
            os.system("bash preprocessing_files.sh")
        else:
            print("Files are ready for analysis")
            # Apply supervised training
            os.system("bash analysis.sh")
    


    download_data()
    analysis()


if __name__=="__main__":
    pipeline()