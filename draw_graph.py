# coding: utf-8
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import six

file_names = [
    'log1019'
]

logfile = {}

for name in file_names:
    f = open("result/" + name)
    logfile[name] = json.load(f)
    f.close()

data = {
#    'simple/levenstein_distance': "*",
#    'synth/levenstein_distance': "o",
#    'simple/avg_loss': "s",
#    'synth/avg_loss': "+",
#    'myval/0/loss': "h",
#    'myval/3/loss': "H",
#    'myval/6/loss': "H",
#    'myval/8/loss': "8",
#    'myval/0/accuracy': "h",
#    'myval/3/accuracy': "H",
#    'myval/6/accuracy': "H",
#    'myval/8/accuracy': "8",
    '0/loss': "2",
    '1/loss': "3",
    '2/loss': "x",
    '4/loss': "p",
    '5/loss': "v",
    '10/loss': ">",
    '19/loss': "1"
}

for name in logfile:
    for key, value in six.iteritems(data):
        plt.plot(
        list(map(lambda x:x["epoch"], logfile[name])), 
        list(map(lambda x:x[key], logfile[name])),
        markevery=100, label=key )
        #marker=value, markevery=100, label=key )

plt.xlabel("epochs")
plt.ylabel("loss or distance")
#plt.legend(loc="lower right")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.subplots_adjust(right=0.7)
plt.savefig('result/graph.png', bbox_inches='tight')
