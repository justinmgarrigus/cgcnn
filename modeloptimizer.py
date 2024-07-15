import argparse
import random
import subprocess
import shlex
import sys

parser = argparse.ArgumentParser(description='optimizing model parameters')
parser.add_argument('--type', choices=['random', 'genetic', 'anneal'],
                    default='regression', help='gives the type of optimizer, can be random, genetic, or anneal (simulated annealing)')
args = parser.parse_args()

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

epochs = [15, 30, 50, 100, 150, 200, 250]
batchsize = [64, 128, 256, 384, 512]
learningrate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
lrmilestone = [25, 50, 100]
momentum = [0.5, 0.75, 0.8, 0.9, 0.99]
weightdecay = [0, 0.1, 0.01, 0.0001, 0.00001]
trainsize = [100, 200, 300, 400, 500, 600]
optimizer = ['SGD', 'Adam']
atomfealen = [16, 32, 64, 128]
hfealen = [64, 128, 256]
nconv = [3, 4, 5]
nh = [1, 2, 3]
pretrain = [True, False]

low = sys.float_info.max

if args.type == 'random':
    for x in range(100):

        r1 = random.random()*len(epochs)
        r2 = random.random()*len(batchsize)
        r3 = random.random()*len(learningrate)
        r4 = random.random()*len(lrmilestone)
        r5 = random.random()*len(momentum)
        r6 = random.random()*len(weightdecay)
        r7 = random.random()*len(trainsize)
        r8 = random.random()*len(optimizer)
        r9 = random.random()*len(atomfealen)
        r10 = random.random()*len(hfealen)
        r11 = random.random()*len(nconv)
        r12 = random.random()*len(nh)
        r13 = random.random()*len(pretrain)

        f = open("optimizing_run.txt", "w")
        stringstring = "python3 main.py --epochs " + str(epochs[r1]) + " --batch-size " + str(batchsize[r2]) + " --learning-rate " + str(learningrate[r3]) + " --lr-milestones " + str(lrmilestone[r4]) + " --momentum " + str(momentum[r5]) + " --weight-decay " + str(weightdecay[r6]) + " --train-size " + str(trainsize[r7]) + "--optim " + str(optimizer[r8]) + " --athom-fea-len " + str(atomfealen[r9]) + " --h-fea-len " + str(hfealen[r10]) + " --n-conv " + str(nconv[r11]) + " --n-h " + str(nh[r12]) + " --freeze-fc 0 --test-ratio 0.2 --seed 42 root_dir"
        subprocess.run(shlex.split(stringstring), stdout = f)
        f.close()
        losses = []
        with open('optimizing_run.txt') as file:
            for line in file:
                if 'Loss' in line:
                    afterloss = line.split('Loss')[1]
                    loss = float(afterloss.split()[1].strip())
                    losses.append(loss)
        if losses[len(losses)-1] < low:
            f = open("optimized_python_command.txt", "w")
            f.write(stringstring)
            f.close()
            
