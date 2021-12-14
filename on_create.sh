#!/bin/bash

set -e

# OVERVIEW
# This script installs a custom, persistent installation of conda on the Notebook Instance's EBS volume, and ensures
# that these custom environments are available as kernels in Jupyter.
#
# The on-create script downloads and installs a custom conda installation to the EBS volume via Miniconda. Any relevant
# packages can be installed here.
#   1. ipykernel is installed to ensure that the custom environment can be used as a Jupyter kernel
#   2. Ensure the Notebook Instance has internet connectivity to download the Miniconda installer


sudo -u ec2-user -i <<'EOF'
unset SUDO_UID

conda update -n base -c defaults conda

# Install a separate conda installation via Miniconda
WORKING_DIR=/home/ec2-user/SageMaker/custom-miniconda
mkdir -p "$WORKING_DIR"
wget https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O "$WORKING_DIR/miniconda.sh"
bash "$WORKING_DIR/miniconda.sh" -b -u -p "$WORKING_DIR/miniconda"
rm -rf "$WORKING_DIR/miniconda.sh"

# Create a custom conda environment
source "$WORKING_DIR/miniconda/bin/activate"
KERNEL_NAME="mailout"
PYTHON="3.7"
conda create --yes --name "$KERNEL_NAME" python="$PYTHON"
conda activate "$KERNEL_NAME"
pip install --quiet ipykernel

# Customize these lines as necessary to install the required packages

#pip3 install -r requirements.txt
#conda install --yes --file requirements.txt
#while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements2.txt

conda install --yes pandas==1.3.4
conda install --yes numpy==1.21.2
conda install --yes scikit-learn==1.0.1
conda install -c conda-forge --yes imbalanced-learn==0.8.0
conda install --yes xgboost==1.3.3
conda install --yes seaborn==0.11.2
conda install --yes matplotlib==3.5.0
conda install --yes missingno==0.4.2
conda install --yes openpyxl==3.0.9
conda install --yes boto3==1.18.21
conda install -c conda-forge sagemaker-python-sdk
conda install --yes jupyter==1.0.0
conda install --yes python-graphviz

pip3 install --quiet sagemaker==2.68.0