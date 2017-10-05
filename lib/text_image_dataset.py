#! /usr/bin/env python
# coding: utf-8
# coding=utf-8
# -*- coding: utf-8 -*-
# vim: fileencoding=utf-8
import sys
import random
import numpy as np
import string
import chainer
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

class TextImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, datanum=10, max_length=6, normalize=True, flatten=False, train=True, device=-1):
        self._normalize = normalize
        self._flatten = flatten
        self._train = train
        self._device = device
        self._max_length = max_length
        pairs = []
        for _ in range(datanum):
            image_array, label = self.generate_data()
            pairs.append([image_array, label])
        self._pairs = pairs

    def __len__(self):
        return len(self._pairs)

    def generate_data(self):
        # generate text
        #length = 6
        length = max(random.randrange(2,self._max_length), random.randrange(2,self._max_length))
        text = self.generate_random_string(length)
        #print(text)

        # text to image
        image_array = self.text_to_image(text)

        # text to label
        label = text_to_label(text, length=self._max_length)
        return image_array, label

    def get_example(self, i):
        image_array, label = self._pairs[i][0], self._pairs[i][1]
        return image_array, label

    def generate_random_string(self, size=6, chars=string.ascii_uppercase + string.digits + ' '):
        return ''.join(random.choice(chars) for _ in range(size))

    #def text_to_label(self, text):
    #    label = [] #np.zeros((len(text), 37)) # 37character type number
    #    for index, c in enumerate(text):
    #        label.append(self.char_to_int(c))
    #    label = np.asarray(label).astype('int32')
    #    if self._device >= 0:
    #        label = chainer.cuda.to_gpu(label)
    #    return label

    ## returns 0~9: number / 10: space / 11~36: upper case
    ## currently don't accept small case alphabets
    #def char_to_int(self, c):
    #    ascii_code = ord(c)
    #    if (ascii_code >=48 and ascii_code <= 57):
    #        # print("number")
    #        ascii_code = ascii_code - 48
    #    elif (ascii_code >=65 and ascii_code <= 90):
    #        # print("upper case")
    #        ascii_code = ascii_code - 54
    #    elif (ascii_code == 32):
    #        # print("space")
    #        ascii_code = 10
    #    else:
    #        raise ValueError("not a alphanumeric character")
    #    return ascii_code

    def text_to_image(self, text):
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

if __name__ == '__main__':        
    train_data = TextImageDataset(10, max_length=8, train=True) 
