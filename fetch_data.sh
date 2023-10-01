#!/bin/bash

# Script that fetches checkpoints and other necessary data for inference

# Downloading existing checkpoint
cd checkpoints
wget https://keeper.mpdl.mpg.de/d/9cc35bc939c14330acc1/ --max-redirect=2 --trust-server-names