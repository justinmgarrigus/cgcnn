import matplotlib.pyplot as plt
import glob
import re
import os
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator
from statistics import mean

def extract_final_MAE(filename):
    mae_values = []
    completed = False
    with open(filename, 'r') as file:
        for line in file:
            if '* MAE' in line:
                mae = float(line.split('* MAE')[1].strip())
                mae_values.append(mae)
                # afterloss = line.split('Loss')[1]
                # loss = float(afterloss.split()[0].strip())
                # losses.append(loss)
            if 'NO TEST' in line:
                completed = True
    if not completed:
        return -1.0
    else:
        return mae_values[len(mae_values) - 1]

def extract_mae_from_file(filename):
    mae_values = []
    with open(filename, 'r') as file:
        for line in file:
            if '* MAE' in line:
                mae = float(line.split('* MAE')[1].strip())
                mae_values.append(mae)
    return mae_values

def plot_mae_over_functions(variables, typemodel):

    #if the loss exists in the file, average over all k folds
    #make sure the optimizer is different
    #save to separate files 
    #cry 
    
    plt.figure(figsize =(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    markers = ['o', 's', 'D', '^', 'v', '*']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    marker_size = 11  # Increase marker size

    for (title, directory_list), marker, linestyle in zip(variables.items(), marker, linestyle):
        all_mae_values = []
        
        #gather all files that have the name 'finetune', 'from scratch', etc, create a 2d array of size X type
        for j in range(len(directory_list[1]) + 1): #iterating over the different sizes of the thing we are running
            mae_per_size = []
            for l in range(6): #iterating over the six different types of model used
                if j == len(directory_list): #default value
                    root = directory_list[0] + '/' + str(directory_list[2])
                else:
                    root = directory_list[0] + '/' +  str(directory_list[1][j])
                files = glob.glob(typemodel[l], root_dir = root)
                
                avgmaelist = []
                for file in files:
                    mae = extract_final_MAE(file)
                    if mae != -1.0:
                        avgmaelist.append(mae)
                
                maefinal = mean(avgmaelist)
                mae_per_size.append(maefinal)
            all_mae_values.append(mae_per_size)
        
            




                
            

        path = directory_list[0] + '/' + str(directory_list[1])
        files.append(list(os.scandir(path)))
        #files.extend(os.scandir(path))
        print("wow")


    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.title('Loss over Epochs for Variations in ' + title, fontsize=26)
    plt.legend(fontsize=18, loc='best')

    # Set y-axis limits and ticks
    ax = plt.gca()
    ax.set_ylim(0, 8)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    # Increase tick label sizes for x and y axis numbers
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

    return

def plot_mae_for_patterns(patterns, output_filename):
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    markers = ['o', 's', 'D', '^', 'v', '*']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    marker_size = 11  # Increase marker size

    for (title, file_pattern), marker, linestyle in zip(patterns.items(), markers, linestyles):
        files = glob.glob(file_pattern)
        all_mae_values = []

        for file in files:
            mae_values = extract_mae_from_file(file)
            all_mae_values.append(mae_values)

        # Convert to numpy array for easy mean and std computation
        all_mae_values = np.array(all_mae_values)

        if all_mae_values.size > 0:
            mean_mae = np.mean(all_mae_values, axis=0)
            std_mae = np.std(all_mae_values, axis=0)

            epochs = range(1, len(mean_mae) + 1)
            plt.plot(epochs, mean_mae, label=title, linewidth=3, marker=marker, linestyle=linestyle, markersize=marker_size)
            plt.fill_between(epochs, mean_mae - std_mae, mean_mae + std_mae, alpha=0.2)

    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('MAE', fontsize=24)
    plt.title('MAE over Epochs for Different Training Methods', fontsize=26)
    plt.legend(fontsize=18, loc='best')

    # Set y-axis limits and ticks
    ax = plt.gca()
    ax.set_ylim(0, 8)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    # Increase tick label sizes for x and y axis numbers
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

variables = {
    'Batch Size': ['tests/batchsize', [64, 128, 384, 512], 256],
    'Learning Rate': ['tests/learningrate', [0.1, 0.001, 0.0001, '1e-05'], 0.01],
    'Momentum': ['tests/momentum', [0.5, 0.75, 0.8, 0.99], 0.9],
    'Optimizer Type': ['tests/optimizer', [], 'Adam'],
    'Training Size': ['tests/trainsize', [100, 200, 300, 400, 500, 600], 'full'], 
    'Weight Decay': ['tests/weightdecay', [0.1, 0.01, 0.001, 0.0001], 0]
}

typemodel = [
    'FromScratch_k_fold_*.txt',
    'FreezeConv_k_fold_*.txt',
    'FreezeConvEmbed_kfold_*.txt',
    'FineTune_kfold_*.txt',
    'FreezeConv-FineTune_kfold_*.txt',
    'FreezeConvEmbed-FineTune_kfold_*.txt'
]

# Define the patterns for the sets of files
patterns = {
    'Finetuning': 'finetuning_kfold_*.txt',
    'From Scratch': 'fromscratch_kfold_*.txt',
    'Transfer': 'transfer_kfold_*.txt',
    'Transfer Embedding Finetuning': 'transferembeddingfinetuning_kfold_*.txt',
    'Transfer Embedding': 'transferembedding_kfold_*.txt',
    'Finetuning Adam': 'finetuningadamfc_kfold_*.txt',
}

# Plot and save the single figure with multiple lines
plot_mae_for_patterns(patterns, 'mae_over_epochs_combined.png')

print("Plot saved successfully.")