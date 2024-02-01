# Use the official Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the conda environment YAML file
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Activate the conda environment
RUN echo "source activate deco_p39" > ~/.bashrc
ENV PATH /opt/conda/envs/deco_p39/bin:$PATH
