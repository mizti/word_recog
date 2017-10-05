#! /usr/bin/env python
# coding: utf-8
# coding=utf-8
# -*- coding: utf-8 -*-
# vim: fileencoding=utf-8
import sys
import random
import h5py
import numpy as np
import string
import chainer
import matplotlib.pyplot as plt
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from utils import *

def flatten_list(txt):
    def spl(t):
        return t.split("\n")
    flatten = lambda l: [item for sublist in l for item in sublist]
    txt = map(spl, txt)
    txt = flatten(txt)
    txt = filter(lambda t: t!="", txt)
    txt = map(lambda t: t.strip(), txt)
    return txt

class SynthTextDataset(chainer.dataset.DatasetMixin):

    def __init__(self, datanum=sys.maxint, max_length=8, flatten=False, validation=False ,device=-1):
        self.datanum = datanum
        self.flatten = flatten
        self.device = device
        self.max_length = max_length
        self.validation = validation

        if validation:
            self.db = h5py.File('data/SynthTextVal.h5', 'r') # DB for validation
        else:
            self.db = h5py.File('data/SynthText.h5', 'r') # DB for training
            
        self.dsets = sorted(self.db['data'].keys()) # list of filename
        self.word_list = []
        self.bookmark = [None] * len(self.dsets)

        for i, filename in enumerate(self.dsets):
            words_in_file = flatten_list(self.db['data'][filename].attrs['txt'])
            self.word_list += words_in_file
            if i > 0:
                self.bookmark[i] = self.bookmark[i-1]+len(words_in_file)
            elif i == 0:
                self.bookmark[i] = len(words_in_file)

    def __len__(self):
        return min(len(self.word_list),self.datanum)
        #return min(self.datanum, len(self.word_list))
        #return len(self.word_list)

    def get_example(self, i):
        image_array, label = self.get_image_and_label(i)
        return image_array, label

    def get_image_and_label(self, i):
        index = 0
        while self.bookmark[index] <= i:
            index += 1

        db = self.db
        if index != 0:
            loc_in_file = i - self.bookmark[index-1]
        elif index == 0:
            loc_in_file = i
        else:
            raise ValueError
        label = self.word_list[i]
        image_file = db['data'][self.dsets[index]][...]
        wordBB = db['data'][self.dsets[index]].attrs['wordBB']
        bb = wordBB[:,:,loc_in_file]
        bb = np.c_[bb,bb[:,0]]

        im = np.asarray(image_file)

        #cut off bottom
        im = im[:int(max(bb[1,2],bb[1,3]))]
        #cut off top
        im = im[int(min(bb[1,0],bb[1,1])):]
        #cut off right
        im = im[:,:int(max(bb[0,1],bb[0,2]))]
        #cut off left
        im = im[:,int(min(bb[0,0],bb[0,3])):]

        im = self.adjust_image(im)
        return im, text_to_label(label.upper(), length=self.max_length)

    def show_image(self, image_array, mode=""):
        print("--")
        print(image_array.shape)
        plt.close(1)
        plt.figure(1)
        if mode == 'grey':
            plt.imshow(image_array, cmap='gray')
        else:
            plt.imshow(image_array)
        plt.hold(True)
        ##plt.gca().set_xlim([max(bb[0,0],bb[0,3]), min(bb[0,1],bb[0,2])])
        ##plt.gca().set_ylim([max(bb[1,2],bb[1,3]), min(bb[1,0],bb[1,1])])
        plt.show(block=False)
        raw_input("wait")

    # adjust size and format of image array
    def adjust_image(self, image_array):

        #im = Image.fromarray(image_array, 'L')
        im = Image.fromarray(image_array, 'RGB')
        im = im.convert('L')
        
        im = im.resize((100, 32))
    
        im.save('result/image_train' + str(random.randint(0, 100)) + '.png')
        image_array = np.asarray(im)
        
        if self.flatten:
            image_array = image_array.flatten()
        #self.show_image(image_array, mode='grey')
        image_array = image_array.astype('float32') # This will reverse hue for plt
        image_array /= 255
        
        if im.mode == "RGB":
            image_array = image_array.transpose(2, 1, 0) #HWC to CHW
        elif im.mode == "L":
            image_array = image_array[np.newaxis,:]
        else:
            raise ValueError

        # TODO: subtract mean

        if self.device >= 0:
            image_array = chainer.cuda.to_gpu(image_array)
        return image_array

if __name__ == '__main__':        
    train_data = SynthTextDataset(10) 
    train_data.get_example(random.randint(0,30))
