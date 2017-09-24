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
from lib.word_recog_evaluator import *
from lib.sample_result import *

#OUTPUT_NUM = 6
OUTPUT_NUM = 8

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            #conv1 = L.Convolution2D(in_channels=3, out_channels=64, ksize=5, stride=1, pad=2), 
            conv1 = L.Convolution2D(in_channels=3, out_channels=64, ksize=5, stride=1, pad=2), 
            norm1 = L.BatchNormalization(64),
            conv2 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, stride=1, pad=1), 
            norm2 = L.BatchNormalization(128),
            conv3_1 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=1, pad=1), 
            conv3_2 = L.Convolution2D(in_channels=256, out_channels=512, ksize=3, stride=1, pad=1), 
            norm3 = L.BatchNormalization(256),

            conv4 = L.Convolution2D(in_channels=256, out_channels=512, ksize=3, stride=1, pad=1), 
            #conv4 = L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1, pad=1), 

            l1 = L.Linear(26624, 4096)
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        #h = F.relu(self.conv3_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4(h))
        h = F.relu(self.l1(h))
        return h


class Classifier(Chain):
    def __init__(self):
        super(Classifier, self).__init__(
            linear1 = L.Linear(4096,4096),
            linear2 = L.Linear(4096,38) 
        )

    def predict(self, x):
        #y = self.linear1(x)
        y = self.linear2(x)
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
      parser.add_argument('--debug', '-de', action="store_true", help='debug mode')
      args = parser.parse_args()

if args.debug:
    print("debug mode")
    train_data = TextImageDataset(10, max_length=OUTPUT_NUM, train=True, device=args.gpu)
    test_data = TextImageDataset(10, max_length=OUTPUT_NUM, train=False, device=args.gpu)
    train_iter = iterators.SerialIterator(train_data, batch_size=5, shuffle=True)
    test_iter = iterators.SerialIterator(test_data, batch_size=5, repeat=False, shuffle=False)
else:
    train_data = TextImageDataset(1000000, max_length=OUTPUT_NUM, train=True, device=args.gpu)
    test_data = TextImageDataset(10000, max_length=OUTPUT_NUM, train=False, device=args.gpu)
    train_iter = iterators.SerialIterator(train_data, batch_size=50, shuffle=True)
    test_iter = iterators.SerialIterator(test_data, batch_size=50, repeat=False, shuffle=False)


base_cnn = CNN()
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    base_cnn.to_gpu()
base_cnn_optimizer = optimizers.SGD()
base_cnn_optimizer.setup(base_cnn)

classifiers = []
cl_optimizers = []
for i in range(0, OUTPUT_NUM):
    cl = Classifier()
    if args.gpu >= 0:
        cl.to_gpu()
    cl_optimizer = optimizers.SGD()
    cl_optimizer.setup(cl)
    classifiers.append(cl)
    cl_optimizers.append(cl_optimizer)

updater = WordRecogUpdater(train_iter, base_cnn, classifiers, base_cnn_optimizer, cl_optimizers, converter=convert.concat_examples, device=args.gpu)
trainer = training.Trainer(updater, (80, 'epoch'), out=args.output)

print("start running")
trainer.extend(WordRecogEvaluator(test_iter, base_cnn, classifiers, converter=convert.concat_examples, device=args.gpu))
#trainer.extend(sample_recog(trainer, test_data))
trainer.extend(sample_result(TextImageDataset(1, max_length=OUTPUT_NUM, train=False, device=args.gpu)))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'validation/0/loss', 'validation/3/loss', '3/loss', 'validation/5/loss']))
trainer.extend(extensions.ProgressBar())
trainer.run()
print("end running")
