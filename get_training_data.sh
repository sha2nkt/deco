#!/bin/bash

# Script that fetches all necessary data for training and eval

# Download dataset npzs
cd data
mkdir Datasets
wget https://keeper.mpdl.mpg.de/d/aa565394a09b4b0880a1/ --max-redirect=2 --trust-server-names