import subprocess
import shlex

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

## FIRST WE HAVE THE REGULAR SETTINGS
# default 
for i in range(10): 
    f = open("optimizer/FromScratch_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-fc 0 --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# freeze convolution
for i in range(10): 
    f = open("optimizer/FreezeConv_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# freezing convolution + embedding
for i in range(10): 
    f = open("optimizer/FreezeConvEmbed_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-embedding --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# fine tuning
for i in range(10): 
    f = open("optimizer/FineTune_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# freezing convolution + fine tuning
for i in range(10): 
    f = open("optimizer/FreezeConv-FineTune_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# freezing convolution + embedding + finetuning
for i in range(10): 
    f = open("optimizer/FreezeConvEmbed-FineTune_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-embedding --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

## NOW WE HAVE ALL OF THE SAME FLAGS BUT JUST WITH THE ADAM OPTIMIZER INSTEAD OF SGD

# default 
for i in range(10): 
    f = open("optimizer/Adam_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim Adam --freeze-fc 0 --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# freeze convolution
for i in range(10): 
    f = open("optimizer/Adam-FreezeConv_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim Adam --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# freezing convolution + embedding
for i in range(10): 
    f = open("optimizer/Adam-FreezeConvEmbed_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim Adam --freeze-embedding --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# fine tuning
for i in range(10): 
    f = open("optimizer/Adam-FineTune_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim Adam --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# freezing convolution + fine tuning
for i in range(10): 
    f = open("optimizer/Adam-FreezeConv-FineTune_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim Adam --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()

# freezing convolution + embedding + finetuning
for i in range(10): 
    f = open("optimizer/Adam-FreezeConvEmbed-FineTune_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim Adam --freeze-embedding --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 –n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.Popen(shlex.split(stringstring), stdout = f)
    f.close()