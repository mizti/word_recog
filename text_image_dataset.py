#! /usr/bin/env python
# coding: utf-8
# coding=utf-8
# -*- coding: utf-8 -*-
# vim: fileencoding=utf-8
import sys
import random
import numpy as np
import string
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class TextImageDataset(chainer.dataset.DatasetMixin):
	def __init__(self, datanum=10, normalize=True, flatten=True, train=True):
		self._normalize = normalize
		self._flatten = flatten
		self._train = train
		pairs = []
		for _ in range(datanum):
			image_array, label = self.generate_data()
			pairs.append([image_array, label])
		self._pairs = pairs

	def __len__(self):
		return len(self._pairs)

	def generate_data(self):
		# generate text
		length = 6
		text = self.generate_random_string(length)
		#print(text)

		# text to image
		image_array = self.text_to_image(text)

		# text to label
		label = self.text_to_array(text)
		#return image_array, label
		return image_array, label[0]

	def get_example(self, i):
		image_array, label = self._pairs[i][0], self._pairs[i][1]
		return image_array, label

	def generate_random_string(self, size=6, chars=string.ascii_uppercase + string.digits + ' '):
		return ''.join(random.choice(chars) for _ in range(size))

	def text_to_array(self, text):
		label = np.zeros((len(text), 37)) # 37character type number
		for index, c in enumerate(text):
			ascii_code = ord(c)
			# 0~9: number / 10: space / 11~36: upper case
			if (ascii_code == 32):
				# print("space"
				ascii_code = 10
			elif (ascii_code >=48 and ascii_code <= 57):
				# print("number")
				ascii_code = ascii_code - 48
			elif (ascii_code >=65 and ascii_code <= 90):
				# print("upper case")
				ascii_code = ascii_code - 54
			label[index, ascii_code] = 1
		return np.int32(label)

	def text_to_image(self, text):
		# text to image
		fonts = [
			'HelveticaNeueDeskInterface.ttc',
			'Bodoni 72.ttc'
		]
		fontFile = fonts[random.randint(0,len(fonts)-1)]
		font = ImageFont.truetype(fontFile, 60)
		
		w, h = 64 * len(text), 64
		text_w, text_h = font.getsize(text)
		#text_x, text_y = (w - text_w) * random.random(), (h - text_h) * random.random()
		text_x, text_y = (w - text_w) * 0, (h - text_h) * random.random()
		
		im = Image.new('L', (w, h), 255)
		draw = ImageDraw.Draw(im)
		draw.text((text_x, text_y), text, fill=(0), font=font)
	
		if self._train:
			im.save('temp/image_train' + str(random.randint(0, 100)) + '.png')
		else:
			im.save('temp/image_test' + str(random.randint(0, 100)) + '.png')

		image_array = np.asarray(im)
		
		if self._normalize:
		    image_array = image_array / np.max(image_array)
		
		if self._flatten:
			image_array = image_array.flatten()
		image_array = image_array.astype('float32')

		return image_array
		
train_data = TextImageDataset(10, train=True) 
