#!/bin/bash

# Define the paths and variables
DATA_PATH="ccle_vs_gdsc"
OUTDIR="analysis_fl"
TARGET_VARIABLE="Erlotinib"
HPO_ITER=15
FEATURES_TOP_PERCENTILE=10

# Define the model architectures
MODELS=("DirectPred" "supervised_vae" "GNN")

# Define the data type combinations
DATA_TYPES=("mut" "mut,rna" "mut,cnv")

# Define the fusion types (GNN only supports early fusion)
FUSION_TYPES=("early" "intermediate")

# GNN-specific convolution types
GNN_CONV_TYPES=("GC" "SAGE")

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    # Define the output directory based on the model name
    MODEL_OUTDIR="$OUTDIR/$MODEL"
    # Create a subdirectory for each model
    mkdir -p "$MODEL_OUTDIR"
    # Check if model is GNN
    if [[ "$MODEL" == "GNN" ]]; then
        for GNN_CONV in "${GNN_CONV_TYPES[@]}"; do
            for DATA in "${DATA_TYPES[@]}"; do
                echo "Running $MODEL with $GNN_CONV and $DATA"
                flexynesis --data_path "$DATA_PATH" \
                           --model_class "$MODEL" \
                           --gnn_conv_type "$GNN_CONV" \
                           --target_variables "$TARGET_VARIABLE" \
                           --data_types "$DATA" \
                           --fusion_type "early" \
                           --hpo_iter "$HPO_ITER" \
                           --features_top_percentile "$FEATURES_TOP_PERCENTILE" \
                           --outdir "$MODEL_OUTDIR" \
                           --prefix "${MODEL}_${GNN_CONV}_${DATA//,/}_early"
            done
        done
    else
        for DATA in "${DATA_TYPES[@]}"; do
            for FUSION in "${FUSION_TYPES[@]}"; do
                echo "Running $MODEL with $FUSION fusion and $DATA"
                flexynesis --data_path "$DATA_PATH" \
                           --model_class "$MODEL" \
                           --target_variables "$TARGET_VARIABLE" \
                           --data_types "$DATA" \
                           --fusion_type "$FUSION" \
                           --hpo_iter "$HPO_ITER" \
                           --features_top_percentile "$FEATURES_TOP_PERCENTILE" \
                           --outdir "$MODEL_OUTDIR" \
                           --prefix "${MODEL}_${DATA//,/}_${FUSION}"
            done
        done
    fi
done
