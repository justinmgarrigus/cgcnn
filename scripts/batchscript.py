import subprocess
import shlex

#varying the batch size + whether the conv/embedding layers are frozen + whether we have fine tuning

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

#default is 256
# varying batchsize from 64-512, stored in the "batchsize folder" 
for b in range(4):
    batchsize = 0
    if (b == 0):
        batchsize = 64
    if (b == 1):
        batchsize = 128
    if (b == 2):
        batchsize = 384
    if (b == 3):
        batchsize = 512
    for i in range(10): 
        f = open("tests/batchsize/" + str(batchsize) + "/FromScratch_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --batch-size " + str(batchsize) + " --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freeze convolution
    for i in range(10): 
        f = open("tests/batchsize/" + str(batchsize) + "/FreezeConv_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --batch-size " + str(batchsize) + " --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding
    for i in range(10): 
        f = open("tests/batchsize/" + str(batchsize) + "/FreezeConvEmbed_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --batch-size " + str(batchsize) + " --freeze-embedding --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # fine tuning
    for i in range(10): 
        f = open("tests/batchsize/" + str(batchsize) + "/FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --batch-size " + str(batchsize) + " --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + fine tuning
    for i in range(10): 
        f = open("tests/batchsize/" + str(batchsize) + "/FreezeConv-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --batch-size " + str(batchsize) + " --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding + finetuning
    for i in range(10): 
        f = open("tests/batchsize/" + str(batchsize) + "/FreezeConvEmbed-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --batch-size " + str(batchsize) + " --freeze-embedding --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()
