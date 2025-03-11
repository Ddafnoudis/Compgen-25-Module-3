"""
Demonstration of flexynesis package:
BRCA subtype analysis using multi-modal data
"""
# Import libraries
import os
import flexynesis as flx
import torch
import numpy as np

# Set the seed for reproducibility
os.environ["OMP_NUM_THREADS"] = "1"

# Define the number of HPO iterations
HPO_ITER = 10


def pipeline():

    def data_download():
        """
        Download the data
        """
        if not os.path.exists("data/brca_metabric_processed"):
            os.mkdir("data/")
             # Execute the bash script 
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
        tuner = flx.HyperparameterTuning(dataset = train_data,
                                         model_class=flx.DirectPred,
                                         target_variables=["CLAUDIN_SUBTYPE"],
                                         config_name="DirectPred",
                                         n_iter=1,
                                         plot_losses=True,
                                         early_stop_patience=10)
        
        # Define the model and the best parameters
        model, best_params = tuner.perform_tuning()

        print(f"Best parameters: {best_params}\n")
        print(f"Model: {model}\n")

        return model, best_params, test_data

    # Download the data
    data = data_download()
    # Model, Best parameters and test data
    model, best_params, test_data = data_importer()

    def prediction(model, best_params, test_data):
        """
        Prediction task of the model
        """
        # Predict the values
        y_pred_dict = model.predict(test_data)
        print(f"Predicted values: {y_pred_dict}\n")

        metrics_df = flx.evaluate_wrapper(method = 'DirectPred',
                                          y_pred_dict=y_pred_dict, dataset=test_data)
        
        print(f"Metrics: {metrics_df}\n")

        # Save the metrics & best parameters
        metrics_df.to_csv("metrics.csv")
        with open ("best_params.csv", "w") as f:
            f.write(str(best_params))
        
        ds = test_data
        E = model.transform(ds)
        print(type(E))
        E.head()
        # Visualize the embeddings in reduced dimensions
        f = 'CLAUDIN_SUBTYPE'
        #map the sample labels from numeric vector to initial labels. 
        labels = [ds.label_mappings[f][x] for x in ds.ann[f].numpy()] 

        fig = flx.plot_dim_reduced(E, labels, color_type = 'categorical', method='pca')
        # Plot the figure
        fig.show()
        # UMAP Visualization
        fig2 = flx.plot_dim_reduced(E, labels, color_type = 'categorical', method='umap')
        # FIgure background color white
        fig2.show()

    # Prediction task
    prediction(model=model, best_params=best_params, test_data=test_data)


if __name__=="__main__":
    pipeline()
