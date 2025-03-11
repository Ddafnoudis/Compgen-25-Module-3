#!/bin/bash

# Download data
curl -L "https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/lgggbm_tcga_pub_processed.tgz" -o data/lgggbm_tcga_pub_processed.tgz && tar -xzvf data/lgggbm_tcga_pub_processed.tgz