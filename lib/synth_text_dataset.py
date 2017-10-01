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

    def __init__(self, datanum=10, normalize=True, flatten=False, validation=False ,device=-1):
        self.normalize = normalize
        self.flatten = flatten
        self.device = device
        self.max_length = 8
        self.validation = validation

        if validation:
            self.db = h5py.File(DB_NAME, 'r') # DB for validation
        else:
            self.db = h5py.File('SynthText.h5', 'r') # DB for training
            
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
        return len(self._pairs)

    def get_example(self, i):
        image_array, label = self.get_image_and_label(i)
        return image_array, label

    def get_image_and_label(self, i):
        index = 0
        #print(self.bookmark)
        #print("i")
        #print(i)
        while self.bookmark[index] <= i:
            index += 1

        #print(str(i) + " found in " + str(index))
        #print(self.word_list[i]) #label text

        db = self.db
        #print(self.bookmark[index])
        if index != 0:
            loc_in_file = i - self.bookmark[index-1]
        elif index == 0:
            loc_in_file = i
        else:
            raise ValueError
        #print("loc_in_file")
        #print(loc_in_file)

        #print(self.dsets[index])
        label = self.word_list[i]
        image_file = db['data'][self.dsets[index]][...]
        wordBB = db['data'][self.dsets[index]].attrs['wordBB']
        #print(wordBB[:,:,loc_in_file])
        bb = wordBB[:,:,loc_in_file]
        bb = np.c_[bb,bb[:,0]]
        print(bb.shape)

        print("---")
        print(label)
        im = np.asarray(image_file)

        #cut off bottom
        im = im[:int(max(bb[1,2],bb[1,3]))] #ok
        #cut off top
        im = im[int(min(bb[1,0],bb[1,1])):] #ok
        #cut off right
        im = im[:,:int(max(bb[0,1],bb[0,2]))]
        #cut off left
        im = im[:,int(min(bb[0,0],bb[0,3])):]
        
        #plt.close(1)
        #plt.figure(1)
        #im = im.tolist()
        #plt.imshow(im)
        #plt.hold(True)
        ##plt.gca().set_xlim([max(bb[0,0],bb[0,3]), min(bb[0,1],bb[0,2])])
        ##plt.gca().set_ylim([max(bb[1,2],bb[1,3]), min(bb[1,0],bb[1,1])])
        #plt.show(block=False)
        #raw_input("wait")
        return im, label

    def image_adjust(self, text):
        fonts = [
            'Arial Rounded Bold.ttf',
            'Avenir.ttc',
            'Bodoni 72.ttc',
            'GillSans.ttc',
            'HelveticaNeueDeskInterface.ttc',
            'Times New Roman.ttf'
        ]
        fontFile = fonts[random.randint(0,len(fonts)-1)]
        font = ImageFont.truetype('data/'+fontFile, 30)

        #size   
        text_w, text_h = font.getsize(text)
        text_x, text_y = 0, 0
        w, h = text_w, 32 #fixed
        
        
        #im = Image.new('RGB', (w, h), (255,255,255)) # RGB
        im = Image.new('L', (w, h), 30) #Greyscale
        draw = ImageDraw.Draw(im)
        #draw.text((text_x, text_y), text, fill=(0,100,80), font=font) # RGB
        draw.text((text_x, text_y), text, fill=(230), font=font)

        #im = im.resize((32*6, 32))
        im = im.resize((100, 32))
    
        #if self._train:
        #   im.save('result/image_train' + str(random.randint(0, 100)) + '.png')
        #else:
        #   im.save('result/image_test' + str(random.randint(0, 100)) + '.png')

        image_array = np.asarray(im)
        
        if self._normalize:
            image_array = image_array / np.max(image_array)
        
        if self._flatten:
            image_array = image_array.flatten()
        image_array = image_array.astype('float32')
        
        if im.mode == "RGB":
            image_array = image_array.transpose(2, 1, 0) #HWC to CHW
        elif im.mode == "L":
            image_array = image_array[np.newaxis,:]
        else:
            raise ValueError

        if self._device >= 0:
            image_array = chainer.cuda.to_gpu(image_array)
        return image_array
        
train_data = SynthTextDataset(10) 

train_data.get_example(random.randint(0,30))
