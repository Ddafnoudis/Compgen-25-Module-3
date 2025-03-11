"""
Demonstration of flexynesis package:
BRCA subtype analysis using multi-modal data
"""
# Import libraries
import os
import flexynesis as flx
from pathlib import Path
import torch
torch.set_num_threads(4)

# Set the seed for reproducibility
os.environ["OMP_NUM_THREADS"] = "1"

# Define the number of HPO iterations
HPO_ITER = 10

# Define the folders
DATA_FOLDER = "data/"
DATA_FL_UNZIP = "brca_metabric_processed/"
RESULTS_FOLDER = "b-params_metrics/"
FIGURES_FOLDER = "figures/"


def pipeline():

    def create_folders(data_folder: Path, data_fl_unzip: Path, results_folder:Path, figures_folder: Path)-> None:
        """
        Create the folders if not exist
        """
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        if not os.path.exists(data_fl_unzip):
            os.mkdir(data_fl_unzip)
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        if not os.path.exists(figures_folder):
            os.mkdir(figures_folder)


    def data_download():
        """
        Download the data
        """
        os.system("bash data_download.sh")


    def data_importer():
        """
        Import the data
        """
        # Import the data
        data_importer = flx.DataImporter(path="brca_metabric_processed/",
                                         data_types = ['gex', 'cna'],
                                         concatenate=False,
                                         top_percentile=10,
                                         min_features=1000,
                                         variance_threshold=0.8,)
        
        # Define the test and train data
        train_data, test_data = data_importer.import_data()

        print(f"Data Matrices\n{train_data.dat}\n")

        print(f"{train_data.dat['gex'].shape}\n, {train_data.dat['cna'].shape}\n")

        print(f"Data annotation\n{train_data.ann}\n")

        print(f"Mapping of sample labels for categorical variables\n{train_data.label_mappings}\n")

        print(f"Samples 1:10:\n{train_data.samples[1:10]}\n\n Features\n{train_data.features}\n")

        print(f"Summary of the samples\n{flx.print_summary_stats(train_data)}\n")

        # Perform hyperparameter tuning
        tuner = flx.HyperparameterTuning(dataset=train_data,
                                         model_class=flx.DirectPred,
                                         target_variables=["CLAUDIN_SUBTYPE"],
                                         config_name="DirectPred",
                                         n_iter=1,
                                         plot_losses=False,
                                         early_stop_patience=10)
        
        # Define the model and the best parameters
        model, best_params = tuner.perform_tuning()

        print(f"Best parameters: {best_params}\n")
        print(f"Model: {model}\n")

        return model, best_params, test_data

    def prediction(model, best_params, test_data, result_folder=RESULTS_FOLDER, figures=FIGURES_FOLDER):
        """
        Prediction task of the model
        """
        # Predict the values
        y_pred_dict = model.predict(test_data)
        print(f"Predicted values: {y_pred_dict}\n")

        metrics_df = flx.evaluate_wrapper(method = 'DirectPred',
                                          y_pred_dict=y_pred_dict, dataset=test_data)
        
        print(f"Metrics: {metrics_df}\n")

        # Save the metrics
        metrics_df.to_csv(result_folder + "metrics.csv")
        # Open a file and save the best parameters
        with open (result_folder + "best_params.csv", "w") as f:
            f.write(str(best_params))
        
        # Rename test data
        ds = test_data
        # Transform the data
        E = model.transform(ds)
        # Print the type of model transforming data
        print(f"Type of model transforming data: {type(E)}\n")
        
        print(f"Head of model transformed data: \n{E.head()}\n")

        # Visualize the embeddings in reduced dimensions
        f = 'CLAUDIN_SUBTYPE'
        # Map the sample labels from numeric vector to initial labels. 
        labels = [ds.label_mappings[f][x] for x in ds.ann[f].numpy()] 

        # Define the PCA figure of reduced plots
        fig = flx.plot_dim_reduced(E, labels, color_type = 'categorical', method='pca')
        # save the figure
        fig.save(figures + "PCA.png")
        # Plot the figure
        fig.show()
        
        # Define the UMAP figure of reduced plots
        fig2 = flx.plot_dim_reduced(E, labels, color_type = 'categorical', method='umap')
        fig2.save(figures + "UMAP.png")
        # FIgure background color white
        fig2.show()

    # Create the folders
    create_folders(data_folder=DATA_FOLDER, data_fl_unzip=DATA_FL_UNZIP, results_folder=RESULTS_FOLDER, figures_folder=FIGURES_FOLDER)
    # Download the data
    data = data_download()
    # Model, Best parameters and test data
    model, best_params, test_data = data_importer()
    # Prediction task
    prediction(model=model, best_params=best_params, test_data=test_data)


if __name__=="__main__":
    pipeline()
