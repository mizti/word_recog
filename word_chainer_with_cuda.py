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
from text_image_dataset import *

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
    def __init__(self, base_cnn):
        super(Classifier, self).__init__(
            base_cnn = base_cnn,
            linear = L.Linear(4096,37) 
        )

    def predict(self, x):
        h = self.base_cnn(x)
        y = self.linear(h)
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
  
train_data = TextImageDataset(10000, train=True, device=args.gpu)
test_data = TextImageDataset(1000, train=False, device=args.gpu)
train_iter = iterators.SerialIterator(train_data, batch_size=50, shuffle=True)
test_iter = iterators.SerialIterator(test_data, batch_size=50, repeat=False, shuffle=False)

base_cnn = CNN()
model = Classifier(base_cnn)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model = CNN().to_gpu()
optimizer = optimizers.SGD()

optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
#updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (80, 'epoch'), out=args.output)


print("start running")
trainer.extend(extensions.Evaluator(test_iter, model))
#trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
print("end running")
