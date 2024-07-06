import subprocess
import shlex

#varying the learning rate + whether the conv/embedding layers are frozen + whether we have fine tuning

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

#default is 0.01
# varying learning rate size from 0.1 to 0.00001, stored in the "learningrate folder" 
for b in range(4):
    learningrate = 0
    if (b == 0):
        learningrate = 0.1
    if (b == 1):
        learningrate = 0.001
    if (b == 2):
        learningrate = 0.0001
    if (b == 3):
        learningrate = 0.00001
    for i in range(10): 
        f = open("tests/learningrate/" + str(learningrate) + "/FromScratch_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --learning-rate " + str(learningrate) + " --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # freeze convolution
    for i in range(10): 
        f = open("tests/learningrate/" + str(learningrate) + "/FreezeConv_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --learning-rate " + str(learningrate) + " --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding
    for i in range(10): 
        f = open("tests/learningrate/" + str(learningrate) + "/FreezeConvEmbed_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --learning-rate " + str(learningrate) + " --freeze-embedding --freeze-conv --freeze-fc 0 --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # fine tuning
    for i in range(10): 
        f = open("tests/learningrate/" + str(learningrate) + "/FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --learning-rate " + str(learningrate) + " --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + fine tuning
    for i in range(10): 
        f = open("tests/learningrate/" + str(learningrate) + "/FreezeConv-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --learning-rate " + str(learningrate) + " --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()

    # freezing convolution + embedding + finetuning
    for i in range(10): 
        f = open("tests/learningrate/" + str(learningrate) + "/FreezeConvEmbed-FineTune_kfold_" + str(i) + ".txt", "w")
        stringstring = "python3 main.py --optim SGD --learning-rate " + str(learningrate) + " --freeze-embedding --freeze-conv --freeze-fc 0 --fine-tune --h-fea-len 32 --n-conv 4 --n-h 1 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
        subprocess.Popen(shlex.split(stringstring), stdout = f)
        f.close()
