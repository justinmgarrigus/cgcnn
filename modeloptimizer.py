import argparse
import random
import subprocess
import shlex
import sys
import math

parser = argparse.ArgumentParser(description='optimizing model parameters')
parser.add_argument('--type', choices=['random', 'genetic', 'anneal'],
                    default='random', help='gives the type of optimizer, can be random, genetic, or anneal (simulated annealing)')
args = parser.parse_args()

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

epochs = [15, 30, 50, 100, 150, 200, 250]
batchsize = [64, 128, 256, 384, 512]
learningrate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
lrmilestone = [25, 50, 100]
momentum = [0.5, 0.75, 0.8, 0.9, 0.99]
weightdecay = [0, 0.1, 0.01, 0.0001, 0.00001]
trainsize = [100, 200, 300, 400, 500, 600]
#optimizer = ['SGD', 'Adam']
atomfealen = [16, 32, 64, 128]
hfealen = [64, 128, 256]
nconv = [3, 4, 5]
nh = [1, 2, 3]
pretrain = [True, False]

low = sys.float_info.max

if args.type == 'random':
    for x in range(100):

        r1 = int(random.random()*len(epochs))
        r2 = int(random.random()*len(batchsize))
        r3 = int(random.random()*len(learningrate))
        r4 = int(random.random()*len(lrmilestone))
        r5 = int(random.random()*len(momentum))
        r6 = int(random.random()*len(weightdecay))
        r7 = int(random.random()*len(trainsize))
        #r8 = random.random()*len(optimizer)
        r9 = int(random.random()*len(atomfealen))
        r10 = int(random.random()*len(hfealen))
        r11 = int(random.random()*len(nconv))
        r12 = int(random.random()*len(nh))
        r13 = int(random.random()*len(pretrain))

        #f = open("optimizing_run.txt", "w")
        stringstring = "python3 main.py --epochs " + str(epochs[r1]) + " --batch-size " + str(batchsize[r2]) + " --learning-rate " + str(learningrate[r3]) + " --lr-milestones " + str(lrmilestone[r4]) + " --momentum " + str(momentum[r5]) + " --weight-decay " + str(weightdecay[r6]) + " --train-size " + str(trainsize[r7]) + " --optim SGD --atom-fea-len " + str(atomfealen[r9]) + " --h-fea-len " + str(hfealen[r10]) + " --n-conv " + str(nconv[r11]) + " --n-h " + str(nh[r12]) + " --freeze-fc 0 --test-ratio 0.2 --seed 42 root_dir"
        f = subprocess.run(shlex.split(stringstring), encoding = 'utf-8', stdout = subprocess.PIPE)
        losses = []
        #with open('optimizing_run.txt', 'r') as file:
        for line in f.stdout.split('\n'):
            if 'Loss' in line:
                afterloss = line.split('Loss')[1]
                loss = float(afterloss.split()[0].strip())
                losses.append(loss)
        print(losses)
        print(losses[len(losses)-1])
        if losses[len(losses)-1] < low:
            g = open("optimized_python_command.txt", "w")
            g.write(stringstring + "\n" + str(losses[len(losses)-1]))
            g.close()
            low = losses[len(losses)-1]
        print('ITERATION ' + str(x) + ' DONE')

def momentumchoice(currmomentum):
    if currmomentum == 0.5:
        currmomentum = currmomentum + random.random()/4
    else:
        x = random.random()
        if x < 0.5:
            distancetohalf = currmomentum -0.5
            currmomentum = currmomentum - 0.5*random.random() * distancetohalf
        else:
            distanceto1 = 1 - currmomentum
            currmomentum = currmomentum + 0.25*random.random()*distanceto1
    
    return currmomentum

def weightdecaychoice(currwd):
    if currwd == 0:
        x = random.random()
        if x < 0.5:
            currwd = 0.01
        else:
            currwd = 0.1
    else:
        x = random.random()
        if x < 0.3:
            currwd = currwd/10
        elif x < 0.7:
            currwd = currwd
        else:
            multto1 = 1/currwd
            currwd = currwd * random.random() * 0.5 * multto1

def trainsizechoice(curr):
    x = random.random()
    if curr == 700:
        if x < 0.5:
            return curr
        else:
            return 600
    elif curr == 100:
        if x < 0.5:
            return curr
        else:
            return 200
    else:
        if x < 0.3:
            return curr - 100
        elif x < 0.7:
            return curr
        else:
            return curr + 100  

def fealenchoice(curr):
    x = random.random()
    if curr == 8:
        if x < 0.5:
            return curr
        else:
            return 16
    else:
        if x < 0.3:
            return curr/2
        elif x < 0.7:
            return curr
        else:
            return curr*2

def convchoice(curr):
    x = random.random()
    if curr == 3:
        if x < 0.5:
            return curr
        else:
            return curr + 1
    else:
        distto3 = curr-3
        if x < 0.3:
            return curr - int(random.random()*(distto3+1))
        elif x < 0.7:
            return curr
        else:
            return curr + 1 + int(random.random()*2)
def hiddenchoice(curr):
    x = random.random()
    if curr == 1:
        if x < 0.5:
            return curr
        else:
            return curr + 1
    else:
        distto1 = curr-1
        if x < 0.3:
            return curr - int(random.random()*(distto1+1))
        elif x < 0.7:
            return curr
        else:
            return curr + 1 + int(random.random()*2)

if args.type == 'anneal':
    temp = 1
    count = 0

    curr = [30, 256, 0.01, 50, 0.9, 0, 600, 64, 128, 4, 1]
    #curr = [random.random()*len(epochs), random.random()*len(batchsize), random.random()*len(learningrate), random.random()*len(lrmilestone), random.random()*len(momentum),\
    #         random.random()*len(weightdecay), random.random()*len(trainsize),random.random()*len(atomfealen), random.random*len(hfealen), random.random()*len(nconv),\
    #            random.random()*len(nh), random.random()*len(pretrain)]
    curr = [int(i) for i in curr]

    stringstring = "python3 main.py --epochs " + str(curr[0]) + " --batch-size " + str(curr[1]) + " --learning-rate " + str(curr[2]) + " --lr-milestones " + str(curr[3]) + " --momentum " + str(curr[4])\
            + " --weight-decay " + str(curr[5]) + " --train-size " + str(curr[6]) + " --optim SGD --atom-fea-len " + str(curr[7]) + " --h-fea-len " + str(curr[8]) + " --n-conv " + str(curr[9]) + " --n-h " + str(curr[10]) + " --freeze-fc 0 --test-ratio 0.2 --seed 42 root_dir"
    f = subprocess.run(shlex.split(stringstring), encoding = 'utf-8', stdout = subprocess.PIPE)
    losses = []
    #with open('optimizing_run.txt', 'r') as file:
    for line in f.stdout.split('\n'):
        print('wOOOOOWOOWOOWOWWO')
        if 'Loss' in line:
            afterloss = line.split('Loss')[1]
            loss = float(afterloss.split()[0].strip())
            losses.append(loss)
    currloss = losses[len(losses-1)]

    while temp > 0.0001:

        new = [curr[0] + int((random.random() - 0.5)*10), curr[1] + int((random.random() - 0.5)*48), curr[2]/int(random.random() * 100 + 1), curr[3] + int((random.random() - 0.5)*48), momentumchoice(curr[4]),\
               weightdecaychoice(curr[5]), trainsizechoice(curr[6]), fealenchoice(curr[7]), fealenchoice(curr[8]), convchoice(curr[9]), hiddenchoice(curr[6])]
        


        #f = open("optimizing_run.txt", "w")
        stringstring = "python3 main.py --epochs " + str(new[0]) + " --batch-size " + str(new[1]) + " --learning-rate " + str(new[2]) + " --lr-milestones " + str(new[3]) + " --momentum " + str(new[4])\
            + " --weight-decay " + str(new[5]) + " --train-size " + str(new[6]) + " --optim SGD --atom-fea-len " + str(new[7]) + " --h-fea-len " + str(new[8]) + " --n-conv " + str(new[9]) + " --n-h " + str(new[10]) + " --freeze-fc 0 --test-ratio 0.2 --seed 42 root_dir"
        f = subprocess.run(shlex.split(stringstring), encoding = 'utf-8', stdout = subprocess.PIPE)
        losses = []
        #with open('optimizing_run.txt', 'r') as file:
        for line in f.stdout.split('\n'):
            print('wOOOOOWOOWOOWOWWO')
            if 'Loss' in line:
                afterloss = line.split('Loss')[1]
                loss = float(afterloss.split()[0].strip())
                losses.append(loss)
        print(losses)
        print(losses[len(losses)-1])

        newloss = losses[len(losses-1)]
        accept = math.exp((currloss - newloss)/temp)
        
        if accept > 1:
            g = open("optimized_python_command.txt", "w")
            g.write(stringstring + "\n" + str(losses[len(losses)-1]))
            g.close()

        else:
            x = random.random()
            if x < accept:
                g = open("optimized_python_command.txt", "w")
                g.write(stringstring + "\n" + str(losses[len(losses)-1]))
                g.close()
        
        
        temp = 1/(count + 1)
        count = count+1

print('DONE')
