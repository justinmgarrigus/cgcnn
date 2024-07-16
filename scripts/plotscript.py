import matplotlib.pyplot as plt
import glob
import re
import os
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator
from statistics import mean

def extract_final_loss(filename):
    losses = []
    completed = False
    with open(filename, 'r') as file:
        for line in file:
            if 'Loss' in line:
                #mae = float(line.split('* MAE')[1].strip())
                #mae_values.append(mae)
                afterloss = line.split('Loss')[1]
                loss = float(afterloss.split()[0].strip())
                losses.append(loss)
            if 'NO TEST' in line:
                completed = True
    if not completed:
        return -1.0
    else:
        return losses[len(losses) - 1]

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

    #each title is one plot
    for (title, directory_list) in variables.items():
        if title == 'Optimizer Type':
            continue
        all_mae_values = []
        print('--------------' + title + '--------------------')
        
        #gather all files that have the name 'finetune', 'from scratch', etc, create a 2d array of size X type
        #want one thing of each of these tyeps, and they all go on the asme graph
        for j in range(len(directory_list[1]) + 1): #iterating over the different sizes of the thing we are running
            #meant to be a mae per each run of the thing
            mae_per_size = []

            for l in range(6): #iterating over the six different types of model used
                files = []
                if title != 'Optimizer Type':
                    if j == len(directory_list[1]): #default value
                        root = 'tests/default'
                    else:
                        root = directory_list[0] + '/' +  str(directory_list[1][j])
                    files = [(root + '/' + x) for x in glob.glob(typemodel[l], root_dir = os.getcwd() + '/' + root + '/')]
                else:
                    if j == 1:
                        root = 'tests/default'
                        files = [(root + '/' + x) for x in glob.glob(typemodel[l], root_dir = os.getcwd() + '/' + root + '/')]
                    else:
                        root = 'tests/optimizer'
                        adamtype = ['Adam_k_fold_*.txt', 'Adam-FreezeConv_kfold_*.txt', 'Adam-FreezeConvEmbed_kfold_*.txt', 'Adam-FineTune-kfold_*.txt', 'Adam-FreezeConv-FineTune_kfold_*.txt', 'Adam-FreezeConvEmbed-FineTune_kfold_*.txt']

                print(os.getcwd())
                print(type(os.getcwd()))
                
                print(os.getcwd() + '/' + root + '/')

                avgmaelist = []

                print('type of model: ' + typemodel[l])
                print(files)

                

                #compiles an average mae for that group of files
                for file in files:
                    mae = extract_final_MAE(file)
                    if mae != -1.0:
                        avgmaelist.append(mae)
                
                print('LIST OVER THE 10 FILES') 
                print(avgmaelist)
                maefinal = mean(avgmaelist)
                print(maefinal)

                
                mae_per_size.append(maefinal)
            if (j < len(directory_list[1])):
                print('MAE PER '+ str(directory_list[1][j]))
            else:
                print('MAE PER DEFAULT')
            print(mae_per_size)
            if j == len(directory_list[1]):
                all_mae_values.insert(directory_list[3], mae_per_size)
            else:
                all_mae_values.append(mae_per_size)

        #all_mae_values is now a 2d array with the x axis being the type of model and the y axis being the different values of the thing, e.g., batchsize
        #we need to flip it
        print('FIRST ALL MAE VALUES')
        print(all_mae_values)
        all_mae_values = np.array(all_mae_values)
        all_mae_values = all_mae_values.T

        print('ALL MAE VALUES ') 
        print(all_mae_values)

        for typeofrun in range(6):
            #at this point, mae_per_size is now [mae for size 1, mae for size 2, ....]      
            xaxis = directory_list[1].copy()
            xaxis.insert(directory_list[3], directory_list[2])
            print('X AXIS ') 
            print(xaxis)
            plt.plot(xaxis, all_mae_values[typeofrun, :], label = title, linewidth = 3, marker = markers[typeofrun], linestyle = linestyles[typeofrun], markersize = marker_size)
        plt.xlabel(title, fontsize = 24)
        plt.ylabel('MAE', fontsize = 24)
        plt.title('MAE over Epochs for Variations in ' + title, fontsize = 26)
        #plt.savefig('MAE over Epochs for Variations in ' + title, fontsize = 26)
        plt.legend(['From Scratch', 'FreezeConv', 'FreezConvEmbed', 'FineTune', 'FreezeConv FineTune', 'FreezeConvEmbed Finetune'], fontsize = 18, loc = 'best')

         # Set y-axis limits and ticks
        ax = plt.gca()
        ax.set_ylim(0, 1.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        # Increase tick label sizes for x and y axis numbers
        ax.tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()
        plt.savefig('mae for varying ' + title + '.png', dpi=300)
        plt.show()
        plt.clf()

def plot_mae_for_patterns(patterns, output_filename):
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    markers = ['o', 's', 'D', '^', 'v', '*']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    marker_size = 11  # Increase marker size

    for (title, file_pattern), marker, linestyle in zip(patterns.items(), markers, linestyles):
        #one thing per file pattern, and they all go on one graph
        files = glob.glob(file_pattern)
        print(files)
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
    'Batch Size': ['tests/batchsize', [64, 128, 384, 512], 256, 2],
    'Learning Rate': ['tests/learningrate', [0.1, 0.001, 0.0001, '1e-05'], 0.01, 1],
    'Momentum': ['tests/momentum', [0.5, 0.75, 0.8, 0.99], 0.9, 3],
    'Optimizer Type': ['tests/optimizer', ['Adam'], 'SGD', 0],
    'Training Size': ['tests/trainsize', [100, 200, 300, 400, 500, 600], 'full', 6], 
    'Weight Decay': ['tests/weightdecay', [0.1, 0.01, 0.001, '0.0001'], 0, 0]
}

typemodel = [
    'FromScratch_kfold_*.txt',
    'FreezeConv_kfold_*.txt',
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
plot_mae_over_functions(variables, typemodel)

print("Plot saved successfully.")