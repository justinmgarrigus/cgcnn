import subprocess
import shlex

#varying the momentum + whether the conv/embedding layers are frozen + whether we have fine tuning

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

#default is 0.9
# varying momentum size from 0.5 to 0.99, stored in the "momentum folder" 
for b in range(4):
    if (b == 0):
        momentum = 0.5
    if (b == 1):
        momentum = 0.75
    if (b == 2):
        momentum = 0.8
    if (b == 3):
        momentum = 0.99
    for i in range(10): 
        f = open("tests/momentum/" + str(momentum) + "/FromScratch_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --momentum " + str(momentum) + " --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freeze convolution
    for i in range(10): 
        f = open("tests/momentum/" + str(momentum) + "/FreezeConv_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --momentum " + str(momentum) + " --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding
    for i in range(10): 
        f = open("tests/momentum/" + str(momentum) + "/FreezeConvEmbed_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --momentum " + str(momentum) + " --freeze-embedding --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # fine tuning
    for i in range(10): 
        f = open("tests/momentum/" + str(momentum) + "/FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --momentum " + str(momentum) + " --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + fine tuning
    for i in range(10): 
        f = open("tests/momentum/" + str(momentum) + "/FreezeConv-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --momentum " + str(momentum) + " --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding + finetuning
    for i in range(10): 
        f = open("tests/momentum/" + str(momentum) + "/FreezeConvEmbed-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --momentum " + str(momentum) + " --freeze-embedding --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()
