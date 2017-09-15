import sys
import numpy as np
import chainer
import argparse
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from lib.text_image_dataset import *
from lib.word_recog_updater import *

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            #conv1 = L.Convolution2D(in_channels=3, out_channels=64, ksize=5, stride=1, pad=2), 
            conv1 = L.Convolution2D(in_channels=3, out_channels=64, ksize=5, stride=1, pad=2), 
            norm1 = L.BatchNormalization(64),
            conv2 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, stride=1, pad=1), 
            norm2 = L.BatchNormalization(128),
            conv3 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=1, pad=1), 
            norm3 = L.BatchNormalization(256),
            l1 = L.Linear(24576, 4096)
        )

    def predict(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm2(self.conv2(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm3(self.conv3(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.l1(h))
        return y

    def __call__(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm2(self.conv2(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm3(self.conv3(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.l1(h))
        return h


class Classifier(Chain):
    def __init__(self):
        super(Classifier, self).__init__(
            linear = L.Linear(4096,37) 
        )

    def predict(self, x):
        y = self.linear(x)
        return y

    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
      parser.add_argument('--model_snapshot', '-m', default=None, help='Filename of model snapshot')
      parser.add_argument('--output', '-o', default='result', help='Sampling iteration for each test data')
      #parser.add_argument('--data_dir', '-d', default='data', help='directory of pretrain models and image data')
      #parser.add_argument('--net', '-n', default='GoogLeNet', help='Choose network to use for prediction')
      #parser.add_argument('--iteration', '-t', type=int, default=1, help='Sampling iteration for each test data')
      args = parser.parse_args()
  
train_data = TextImageDataset(1000, train=True, device=args.gpu)
test_data = TextImageDataset(1000, train=False, device=args.gpu)
train_iter = iterators.SerialIterator(train_data, batch_size=20, shuffle=True)
test_iter = iterators.SerialIterator(test_data, batch_size=20, repeat=False, shuffle=False)

base_cnn = CNN()
model1 = Classifier()
model2 = Classifier()

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    base_cnn.to_gpu()
    model1.to_gpu()
    model2.to_gpu()

#base_cnn_optimizer = optimizers.SGD()
base_cnn_optimizer = optimizers.Adam()
base_cnn_optimizer.setup(base_cnn)
#model1_optimizer = optimizers.SGD()
model1_optimizer = optimizers.Adam()
model1_optimizer.setup(model1)
#model2_optimizer = optimizers.SGD()
model2_optimizer = optimizers.Adam()
model2_optimizer.setup(model2)

while True:
	iterator = train_iter.next()
	#print(iterator.__class__) #list 
	#print(iterator) #list of tuples, each tuple contains one data and label set
	from chainer.dataset import convert
	in_arrays = convert.concat_examples(iterator, args.gpu)
	#print(in_arrays.__class__) #<class 'tuple'>
	#print(in_arrays) #tuple of lists, one list contains all data, and the other all labels
	xp = np if int(args.gpu) == -1 else cuda.cupy
	x_batch = xp.array(in_arrays[0])
	t_batch = xp.array(in_arrays[1])
	print(t_batch.data.shape)
	print(t_batch)

	#t_batch1 = t_batch[:][0]
	#t_batch2 = t_batch[:][1]
	t_batch1 = t_batch[:,0]
	t_batch2 = t_batch[:,1]

	print(t_batch1.__class__)
	print(t_batch1.shape)
	print(t_batch1)
	print(t_batch2.__class__)
	print(t_batch2.__class__)
	print(t_batch2)
	
	y = base_cnn(x_batch)
	#print(y.data.shape)

	loss1 = model1(y,t_batch1)
	loss2 = model2(y,t_batch2)
	print("loss1="+str(loss1.data))
	print("loss2="+str(loss2.data))
	#print(loss1.__class__) #<class 'chainer.variable.Variable'>
	#print(loss1.shape) #()
	
	base_cnn.cleargrads()
	model1.cleargrads()
	model2.cleargrads()
	
	loss1.backward()
	loss2.backward()
	
	model1_optimizer.update()
	model2_optimizer.update()
	base_cnn_optimizer.update()
	
