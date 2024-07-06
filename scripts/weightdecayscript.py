import subprocess
import shlex

#varying the weight decay + whether the conv/embedding layers are frozen + whether we have fine tuning

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

#default is 0
# varying weight decay from 0.1 to 0.0001, stored in the "momentum folder" 
for b in range(4):
    weightdecay = 0
    if (b == 0):
        weightdecay = 0.1
    if (b == 1):
        weightdecay = 0.01
    if (b == 2):
        weightdecay = 0.001
    if (b == 3):
        weightdecay = 0.0001
    for i in range(10): 
        f = open("tests/weightdecay/" + str(weightdecay) + "/FromScratch_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --weight-decay " + str(weightdecay) + " --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # freeze convolution
    for i in range(10): 
        f = open("tests/weightdecay/" + str(weightdecay) + "/FreezeConv_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --weight-decay " + str(weightdecay) + " --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding
    for i in range(10): 
        f = open("tests/weightdecay/" + str(weightdecay) + "/FreezeConvEmbed_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --weight-decay " + str(weightdecay) + " --freeze-embedding --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # fine tuning
    for i in range(10): 
        f = open("tests/weightdecay/" + str(weightdecay) + "/FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --weight-decay " + str(weightdecay) + " --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + fine tuning
    for i in range(10): 
        f = open("tests/weightdecay/" + str(weightdecay) + "/FreezeConv-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --weight-decay " + str(weightdecay) + " --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding + finetuning
    for i in range(10): 
        f = open("tests/weightdecay/" + str(weightdecay) + "/FreezeConvEmbed-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --weight-decay " + str(weightdecay) + " --freeze-embedding --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()
