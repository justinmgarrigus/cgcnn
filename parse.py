# Given one or more files representing the output of an execution of the 
# training sessions, obtain the list of MSE's and display a line graph of them. 

import sys
import re 
import matplotlib.pyplot as plt 

# pattern = r'Epoch.+Loss (\d\.\d+).*'
# pattern = r'Epoch.+MAE (\d\.\d+).*'
pattern = r' \* MAE (\d\.\d+).*' 

fig, ax = plt.subplots()

for fname in sys.argv[1:]:
    f = open(fname, 'r')
    log = [
        float(match[1])
        for line in f
        if (match := re.search(pattern, line)) is not None 
    ]
    f.close() 
    ax.plot(log, label=fname)
    print(fname) 
    print(log)
    print() 

ax.legend() 
plt.show() 
