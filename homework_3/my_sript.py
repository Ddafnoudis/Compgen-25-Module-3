"""
Pipeline
"""
import os
import time
from pandas import DataFrame

# Define the analysis folders
DIRECTEDPRED = "analysis_fl/DirectPred/"
SUPERVISED_VAE = "analysis_fl/supervised_vae/"


def pipeline():
    """
    1) Download data
    2) Analysis of the data
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
        # Output direcotry
        if not os.path.exists("analysis_fl/"):
            os.mkdir("analysis_fl/")

    
    def analysis(dir_directpred=DIRECTEDPRED, dir_supervised_vae=SUPERVISED_VAE):
        """
        """
        # Check if folders exist
        if not os.path.exists(dir_directpred) or not os.path.exists(dir_supervised_vae):
            # Preprocess file names
            if not os.path.isfile("ccle_vs_gdsc/train/mut.csv") and os.path.isfile("ccle_vs_gdsc/test/mut.csv"):
                # Change files names 
                os.system("bash preprocessing_files.sh")
            else:
                print("Files are ready for analysis")
                # Start time 
                start_time = time.time()
                # Apply supervised training
                os.system("bash analysis.sh")
                # End time
                end_time = time.time()
                # Total time
                print("Time of execution: ", end_time - start_time)
        else:
            print("Folders exist")
    


    download_data()
    analysis()


if __name__=="__main__":
    pipeline()