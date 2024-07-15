import subprocess
import shlex

# PART 1: cross validation test for a single type of model

# k-fold cross validation
for i in range(10): 
    f = open("tests/regular/fromscratch_kfold_" + str(i) + ".txt", "w")
    subprocess.run(["python3", "main.py", "root_dir", "--optim", "SGD", "--freeze-fc", "0", "--n-h", "1", "--h-fea-len", "32", "--n-conv", "4", "--cross-validation", "k-fold-cross-validation", "--cross-param","10", "--counter", str(i), "--seed", "42"], stdout = f)
    f.close()

# bootstrapping
for i in range(100):
    f = open("tests/regular/fromscratch_bootstrapping_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-fc 0 --n-h 1 --h-fea-len 32 --n-conv 4 --cross-validation bootstrapping --cross-param 100 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# leave 2 out
for i in range(600*500):
    f = open("tests/regular/fromscratch_leave2out_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-fc 0 --n-h 1 --h-fea-len 32 --n-conv 4 --cross-validation leave-p-out --cross-param 2 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()  

# leave 1 out
for i in range(600):
    f = open("tests/regular/fromscratch_leave1out_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-fc 0 --n-h 1 --h-fea-len 32 --n-conv 4 --cross-validation leave-one-out --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# monte carlo
for i in range(100):
    f = open("tests/regular/fromscratch_montecarlo_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-fc 0 --n-h 1 --h-fea-len 32 --n-conv 4 --cross-validation monte-carlo --cross-param 100 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# PART 2: each model with a single cross validation type (k-fold)

# file for pretraining
fileName = "pre-trained/formation-energy-per-atom.pth.tar"

# transfer learning
for i in range(10): 
    f = open("tests/regular/transfer_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-embedding --freeze-conv --freeze-fc 0 --n-h 1 --h-fea-len 32 --n-conv 4 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# transfer + 1 CNN
for i in range(10): 
    f = open("tests/regular/transfercnn_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-embedding --freeze-conv --freeze-fc 0 --n-h 1 --n-conv 5 --h-fea-len 32 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# transfer + 1 frozen fc + 1 added fc
for i in range(10): 
    f = open("tests/regular/transferfrozenadditional_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-embedding --freeze-conv --freeze-fc 1 --n-h 2 --n-conv 4 --h-fea-len 32 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# transfer + 1 free fc + 1 added fc + cnn
for i in range(10): 
    f = open("tests/regular/transferfreeadditionalcnn_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-embedding --freeze-conv --freeze-fc 0 --n-h 2 --n-conv 5 --h-fea-len 32 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# transfer + embedding free + fine-tuning
for i in range(10): 
    f = open("tests/regular/transferembeddingfinetuning_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-conv --freeze-fc 0 --n-h 1 --fine-tune --h-fea-len 32 --n-conv 4 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# transfer + embedding free
for i in range(10): 
    f = open("tests/regular/transferembedding_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-conv --freeze-fc 0 --n-h 1 --h-fea-len 32 --n-conv 4 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# transfer + embedding free + 1 add fc
for i in range(10): 
    f = open("tests/regular/transferembeddingfc_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-conv --freeze-fc 0 --n-h 2 --h-fea-len 32 --n-conv 4 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# fine tuning + adam + 1 fc
for i in range(10): 
    f = open("tests/regular/finetuningadamfc_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim Adam --freeze-fc 0 --n-h 2 --fine-tune --h-fea-len 32 --n-conv 4 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# fine tuning + adam + 1 CNN
for i in range(10): 
    f = open("tests/regular/finetuningadamcnn_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim Adam --freeze-fc 0 --n-h 1 --fine-tune --h-fea-len 32 --n-conv 5 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()


# fine tuning all free
for i in range(10): 
    f = open("tests/regular/finetuning_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-fc 0 --n-h 1 --fine-tune --h-fea-len 32 --n-conv 4 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

# fine tuning + 1 add fc
for i in range(10): 
    f = open("tests/regular/finetuningfc_kfold_" + str(i) + ".txt", "w")
    stringstring = "python3 main.py --optim SGD --freeze-fc 0 --n-h 2 --fine-tune --h-fea-len 32 --n-conv 4 --pretrain " + fileName + " --cross-validation k-fold-cross-validation --cross-param 10 --counter " + str(i) + " --seed 42 root_dir"
    subprocess.run(shlex.split(stringstring), stdout = f)
    f.close()

