#!/bin/bash

# Download the BRCA-METABRIC dataset
curl -L "https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/brca_metabric_processed.tgz" -o data/brca_metabric.tgz && tar -xvzf data/brca_metabric.tgz
