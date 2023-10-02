#!/bin/bash

# Script that fetches checkpoints and other necessary data for inference

# Download utility files and other constants
wget https://keeper.mpdl.mpg.de/f/6f2e2258558f46ceb269/?dl=1 --max-redirect=2 --trust-server-names  && tar -xvf Release_Checkpoint.tar.gz --directory data && rm -r Release_Checkpoint.tar.gz

# Downloading existing checkpoint
mkdir checkpoints
wget https://keeper.mpdl.mpg.de/f/6f2e2258558f46ceb269/?dl=1 --max-redirect=2 --trust-server-names  && tar -xvf Release_Checkpoint.tar.gz --directory checkpoints && rm -r Release_Checkpoint.tar.gz

# Downloading other checkpoint
wget https://keeper.mpdl.mpg.de/f/9cb970221b1e45d185b8/?dl=1 --max-redirect=2 --trust-server-names  && tar -xvf Other_Checkpoints.tar.gz --directory checkpoints && rm -r Other_Checkpoints.tar.gz

# Downloading training datasets
mkdir datasets
wget https://keeper.mpdl.mpg.de/f/81c3ec9997dd440b8db3/?dl=1 --max-redirect=2 --trust-server-names  && tar -xvf Release_Datasets.tar.gz --directory datasets && rm -r Release_Datasets.tar.gz