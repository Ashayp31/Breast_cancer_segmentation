#!/bin/bash
#Author: Adam Jaamour
# ----------------------------
echo "Setting Dissertation remote Jupyter environment"
source /cs/scratch/agj6/tf2/venv/bin/activate
cd  ~/Projects/Breast-Cancer-Detection-Code
jupyter notebook --no-browser --
port=8888
