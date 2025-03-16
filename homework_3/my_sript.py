"""
Pipeline
"""
import os
import time
import torch
import pandas as pd
import flexynesis as flx
from pathlib import Path
from pandas import DataFrame
from typing import Dict, List
import matplotlib.pyplot as plt


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

    def preprocessing_results(dir_directpred: Path, dir_supervised_vae: Path)-> Dict[str, List[float]]:
        """
        Preprocess the results of the analysis by adding them in dictionaries
        """
        # Create dictionaries
        direct_pred: Dict[str, List] = {
            "mutcnv_early": [],
            "mutcnv_intermediate": [],
            "mutrna_early": [],
            "mutrna_intermediate": [],
            "mut_early": [],
            "mut_intermediate": [],
        }
        supervised_vae: Dict[str, List] = {
            "mutcnv_early": [],
            "mutcnv_intermediate": [],
            "mutrna_early": [],
            "mutrna_intermediate": [],
            "mut_early": [],
            "mut_intermediate": [],
        }

        # Define empty lists
        direct_pred_list = []
        supervised_vae_list = []

        # Read the files with stats and append pearson corr values based on file_names
        for file_name in os.listdir(dir_directpred):
            # print(file_name)
            if file_name.endswith("stats.csv"):
                # Read the file
                df = pd.read_csv(os.path.join(dir_directpred, file_name))
                # Define the value of pearson in df
                pears_cor_value = float(df.loc[df["metric"] == "pearson_corr", "value"].values[0])
                # Add pearson corr values to the list
                direct_pred_list.append(pears_cor_value)
            
        # Ensure number of values matches dictionary keys
        if len(direct_pred_list) == len(direct_pred.keys()):
            # Assign values correctly
            for key, value in zip(direct_pred.keys(), direct_pred_list):
                direct_pred[key].append(value)
        else:
            print("Error: Mismatch between number of values and dictionary keys")


        for file_name_2 in os.listdir(dir_supervised_vae):
            if file_name_2.endswith("stats.csv"):
                # Read the file
                df = pd.read_csv(os.path.join(dir_supervised_vae, file_name_2))
                # Define the value of pearson in df
                pears_cor_value = float(df.loc[df["metric"] == "pearson_corr", "value"].values[0])
                # Add pearson corr values to the list
                supervised_vae_list.append(pears_cor_value)
        # Ensure number of values matches dictionary keys
        if len(supervised_vae_list) == len(supervised_vae.keys()):
            # Assign values correctly
            for key, value in zip(supervised_vae.keys(), supervised_vae_list):
                supervised_vae[key].append(value)
        else:
            print("Error: Mismatch between number of values and dictionary keys")
    
        # Print the dictionaries
        print(f"DiretctPred dictionary:\n {direct_pred}\n\n")
        print(f"Supervised vae dictionary:\n {supervised_vae}\n\n")

        # Return dictionaries
        return direct_pred, supervised_vae
    

    def plots(direct_pred: Dict[str, List[float]], supervised_vae: Dict[str, List[float]]):
        """
        """
        # Convert dictionaries into a DataFrame
        df_direct = pd.DataFrame(direct_pred).T.rename(columns={0: 'DirectPred'})
        df_vae = pd.DataFrame(supervised_vae).T.rename(columns={0: 'SupervisedVAE'})

        # Merge data into one DataFrame
        df = df_direct.merge(df_vae, left_index=True, right_index=True)

        # Sort based on best performance in either method
        df["Best Score"] = df.max(axis=1)
        df_sorted = df.sort_values(by="Best Score", ascending=False).drop(columns=["Best Score"])

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        df_sorted.plot(kind="bar", figsize=(10, 6), width=0.7, colormap="coolwarm", edgecolor="black")

        # Title and labels
        plt.title("Ranking of Experiments Based on Pearson Correlation", fontsize=14, fontweight="bold")
        plt.ylabel("Pearson Correlation", fontsize=12)
        plt.xlabel("Experiment (data types)", fontsize=12)
        plt.xticks(rotation=45, ha="right")

        # Add grid and legend
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(title="Method", fontsize=11)

        # Show plot
        plt.tight_layout()
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig("plots/ranking_of_experiments.png", dpi=300)
        plt.show()


    def emb():
        """
        """
        # Load embeddings
        train_embeddings = pd.read_csv("analysis_fl/DirectPred/DirectPred_mutrna_early.embeddings_train.csv", sep=",")
        test_embeddings = pd.read_csv("analysis_fl/DirectPred/DirectPred_mutrna_early.embeddings_test.csv", sep=",")

        clinical_data = pd.read_csv("ccle_vs_gdsc/test/clin.csv", sep=",")
        print(clinical_data["tissueid"].unique())
        # # Load the model
        final_model = torch.load("analysis_fl/DirectPred/DirectPred_mutrna_early.final_model.pth", weights_only=False)

        # Define the data from test-embendings
        data_tr = train_embeddings.iloc[:, 1:].values  
        data_te = test_embeddings.iloc[:, 1:].values
        # Extracting the labels 
        labels_tr = train_embeddings.iloc[:, 0].values
        labels_te = test_embeddings.iloc[:, 0].values

        # Applying PCA and plotting with flexynesis plot
        fig_1 = flx.plot_dim_reduced(data_tr, labels_tr, color_type="categorical")
        fig_2 = flx.plot_dim_reduced(data_te, labels_te, color_type="numerical")

        fig_1.show()
        fig_2.show()


        featur_impor_dataset = pd.read_csv("analysis_fl/DirectPred/DirectPred_mutrna_early.feature_importance.IntegratedGradients.csv", sep=",")

        # print(featur_impor_dataset)
        importance_col = featur_impor_dataset["importance"]

        # Sort the DataFrame by the 'importance' column in descending order
        sorted_data = featur_impor_dataset.sort_values('importance', ascending=False)

        # print(sorted_data)
        sorted_data_10 = sorted_data[:10]
        print(sorted_data_10)

        # Print the 10 first names of genes based on sorted importance column
        print(sorted_data_10["name"])


    download_data()
    analysis()
    direct_pred, supervised_vae = preprocessing_results(dir_directpred=DIRECTEDPRED, dir_supervised_vae=SUPERVISED_VAE)
    plots(direct_pred=direct_pred, supervised_vae=supervised_vae)
    emb()



if __name__=="__main__":
    pipeline()
