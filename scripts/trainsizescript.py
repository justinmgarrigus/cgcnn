import subprocess
import shlex

#varying the training size + whether the conv/embedding layers are frozen + whether we have fine tuning

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

# varying training size from 1 to 600, stored in the "trainsize folder" 
for b in range(1, 8):
    trainsize = b * 100
    for i in range(10): 
        f = open("tests/trainsize/" + str(trainsize) + "/FromScratch_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --train-size " + str(trainsize) + " --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freeze convolution
    for i in range(10): 
        f = open("tests/trainsize/" + str(trainsize) + "/FreezeConv_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --train-size " + str(trainsize) + " --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding
    for i in range(10): 
        f = open("tests/trainsize/" + str(trainsize) + "/FreezeConvEmbed_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --train-size " + str(trainsize) + " --freeze-embedding --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # fine tuning
    for i in range(10): 
        f = open("tests/trainsize/" + str(trainsize) + "/FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --train-size " + str(trainsize) + " --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + fine tuning
    for i in range(10): 
        f = open("tests/trainsize/" + str(trainsize) + "/FreezeConv-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --train-size " + str(trainsize) + " --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding + finetuning
    for i in range(10): 
        f = open("tests/trainsize/" + str(trainsize) + "/FreezeConvEmbed-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --train-size " + str(trainsize) + " --freeze-embedding --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()
