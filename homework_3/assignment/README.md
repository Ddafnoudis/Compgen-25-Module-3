Description
-----------------

- With this homework, we will practice building multiple models on the command-line. 
- We will use multi-omics data from human cancer cell lines from the CCLE and GDSC databases. 
- We will build the models on the CCLE dataset and evaluate the models on the GDSC dataset. 
- We will do a benchmarking of different deep learning architectures and different combinations of omic data modalities used as input.
- Finally, we will explore the best performing models in a jupyter notebook. 

Steps for the homework:

  1. Download and unpack the data: https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/ccle_vs_gdsc.tgz
    
  2. Familiarize yourself with the command-line options for Flexynesis:
    See tutorial here: https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/getting_started/

  3. Use flexynesis on the command-line to predict drug responses for “Erlotinib”.
    Write a bash script to run the following experiments: 
      Try a combination of:
     
      a) **different architectures**: e.g. DirectPred, Supervised VAE, GNN (Test at least 2 of these). 
     
      b) **data type combinations** (e.g. mutation, mutation + rna, mutation + cnv) (Test at least 2 of these) 
     
      c) **fusion methods**: early, intermediate (applies only to tools other than GNN)
      
      So, in total, you will run maximally 3 x 3 x 2 = 18 different flexynesis runs (and minimally 2 x 2 x 2 = 8 different runs).

      **Note**: GNNs actually only support "early" fusion, so you can skip "intermediate" fusion for GNNs, but you can try different graph convolution options for GNNs.
        For GNNs, try "GC" and "SAGE" as different options in your experiment (See --gnn_conv_type argument). 

      **Hint 1**: Restrict your analysis to 5-10% of the features (use a combination of variance and laplacian score filtering).
     
      **Hint 2**: It is okay to use few HPO iterations for this exercise (e.g. 15 iterations or so) (considering the resourse/time limits etc). 
       The point of this exercise is not to find the perfect model, but to get an insight on the idea of benchmarking different setups. 
     
  5. Open a jupyter notebook and do the following:
  
        a) Import the results of the experiments from step 3, and rank the experiments based on performance (pearson_corr)
        Which combination yields the best results?
      
        b) Explore the train/test embeddings from the best model (from 4a).
     
        c) Import the feature importance scores from the best model (from 4a). 
           Get top 10 markers. Do literature search. Are any of the top markers associated to “Erlotinib”?  


  6. Submit the jupyter notebook from step 4 as your assignment on the google classroom. 
  
