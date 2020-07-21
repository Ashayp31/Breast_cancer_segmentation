"""
Script predominantly generated by Shuen-jen for visualisation of classification results
Script added and ammended for evaluation and visualisation of segmentation tasks by me.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import pydicom
import random
from sklearn.metrics import jaccard_score
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_io as tfio
from data_operations.dataset_feed import gaussian_blur, tf_equalize_histogram

import config



def plot_comparison_chart_own_results() -> None:
    """
    Plot comparison bar chart.
    :return: None.
    """
    title = "Accuracy Comparison of models"
    
    names = ["Model 1", "Model 2", "Model 3", "Model 4"]
    accuracy = [0.65, 0.63, 0.65, 0.67]

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=names, y=accuracy)

    # Add number at the top of the bar.
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01, height, ha='center')

    # Set title.
    plt.title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.ylim(bottom=0, top=1)
    plt.savefig("../output/Accuracy_Comparison_of_Classification_models.png", bbox_inches='tight')
    plt.show()
    
def plot_comparison_chart() -> None:
    with open('data_visualisation/other_paper_results.json') as config_file:  # load other papers' result from json file
        data = json.load(config_file)
    df = pd.DataFrame.from_records(data["CBIS-DDSM"]["B-M"],
                                   columns=['paper', 'accuracy'])  # Filter data by dataset and classification type.
    new_row = pd.DataFrame({'paper': 'Dissertation Result', 'accuracy': 0.67},
                           index=[0])  # Add model result into dataframe to compare.
    df = pd.concat([new_row, df]).reset_index(drop=True)
    df['accuracy'] = pd.to_numeric(df['accuracy'])  # Digitize the accuracy column.
    
    title = "Accuracy Comparison"

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x='paper', y='accuracy', data=df)

    # Add number at the top of the bar.
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01, height, ha='center')

    # Set title.
    plt.title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.ylim(bottom=0, top=1)
    plt.savefig("../output/comparison_of_best_to_other_papers.png".format(config.dataset, config.model, config.imagesize, title), bbox_inches='tight')
    
plot_comparison_chart_own_results()
plot_comparison_chart()