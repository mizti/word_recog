import sys
import datetime
import numpy as np
import chainer
import argparse
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from lib.simple_text_dataset import *
from lib.synth_text_dataset import *
from lib.word_recog_updater import *
from lib.word_recog_evaluator import *
from lib.test_evaluator import *
from lib.sample_result import *
from lib.decay_lr import * 
import lib.utils

#OUTPUT_NUM = 6
#OUTPUT_NUM = 8
OUTPUT_NUM = 20
DROP_OUT_RATIO = 0.1

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(in_channels=3, out_channels=64, ksize=5, stride=1, pad=2), # for RGB
            #conv1 = L.Convolution2D(in_channels=1, out_channels=64, ksize=5, stride=1, pad=2),  # for L
            norm1 = L.BatchNormalization(64),
            conv2 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, stride=1, pad=1), 
            norm2 = L.BatchNormalization(128),
            conv3_1 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=1, pad=1), 
            conv3_2 = L.Convolution2D(in_channels=256, out_channels=512, ksize=3, stride=1, pad=1), 
            norm3 = L.BatchNormalization(256),
            #conv4 = L.Convolution2D(in_channels=256, out_channels=512, ksize=3, stride=1, pad=1), # for CHAR
            conv4 = L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1, pad=1),  # for CHAR +2
            l1 = L.Linear(26624, 4096)
        )

    def __call__(self, x):
        #print("x")
        #print(x.__class__)
        #print(x.shape)
        #print(x)
        #print("current train status:")
        #print(chainer.config.train)
        h = F.dropout(F.relu(self.conv1(x)), ratio=DROP_OUT_RATIO)
        print_debug(h, "@base_cnn0")
        h = F.max_pooling_2d(h, 2)

        h = F.dropout(F.relu(self.conv2(h)), ratio=DROP_OUT_RATIO)
        h = F.max_pooling_2d(h, 2)
        print_debug(h, "@base_cnn1")

        h = F.dropout(F.relu(self.conv3_1(h)), ratio=DROP_OUT_RATIO)
        h = F.relu(self.conv3_2(h)) # for CHAR +2
        h = F.max_pooling_2d(h, 2)
        print_debug(h, "@base_cnn4")

        h = F.dropout(F.relu(self.conv4(h)), ratio=DROP_OUT_RATIO)
        print_debug(h, "@base_cnn6")
        #h = F.dropout(F.relu(self.l1(h)), ratio=DROP_OUT_RATIO)
        h = self.l1(h)
        print_debug(h, "@base_cnn7")
        #print_debug(self.l1.W, "@L1.W")
        #print_debug(self.l1.b, "@L1.b")
        h = F.dropout(F.relu(h), ratio=DROP_OUT_RATIO)
        print_debug(h, "@base_cnn8")
        return h


class Classifier(Chain):
    def __init__(self):
        super(Classifier, self).__init__(
            linear1 = L.Linear(4096,4096),
            linear2 = L.Linear(4096,len(lib.utils.CHARS)+1) #len(CHARS) + 1
        )

    def predict(self, x):
        y = F.dropout(F.relu(self.linear1(x)), ratio=DROP_OUT_RATIO) # fro CHAR +2
        y = self.linear2(x)
        return y

    def __call__(self, x, t):
        #print("t")
        #print(t.__class__)
        #print(t)
        y = self.predict(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_snapshot', '-m', type=int, default=None, help='Filename of model snapshot')
    parser.add_argument('--output', '-o', default='result', help='Sampling iteration for each test data')
    parser.add_argument('--debug', '-de', action="store_true", help='debug mode')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    args = parser.parse_args()


if args.debug:
    print("debug mode")
    with open('result/debug.txt', 'w') as f:
        f.write('%s'%datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        f.write('\n')

    train_data = SimpleTextDataset(8, max_length=OUTPUT_NUM, train=True, device=args.gpu)
    #test_data = SimpleTextDataset(8, max_length=OUTPUT_NUM, train=False, device=args.gpu)
    #train_data = SynthTextDataset(datanum=50, max_length=OUTPUT_NUM, validation=False, device=args.gpu)
    #test_data = SynthTextDataset(datanum=5, max_length=OUTPUT_NUM, validation=True, device=args.gpu)
    test_data = SimpleTextDataset(80, max_length=OUTPUT_NUM, train=False, device=args.gpu)
    test_data2 = SimpleTextDataset(80, max_length=OUTPUT_NUM, train=False, device=args.gpu)

    train_iter = iterators.SerialIterator(train_data, batch_size=2, shuffle=True)
    test_iter = iterators.SerialIterator(test_data, batch_size=20, repeat=False, shuffle=False)
    test_iter2 = iterators.SerialIterator(test_data2, batch_size=20, repeat=False, shuffle=False)

else:
    #train_data = SimpleTextDataset(1000000, max_length=OUTPUT_NUM, train=True, device=args.gpu)
    #test_data = SimpleTextDataset(10000, max_length=OUTPUT_NUM, train=False, device=args.gpu)
    train_data = SynthTextDataset(validation=False, max_length=OUTPUT_NUM, device=args.gpu)
    test_data = SynthTextDataset(validation=True, max_length=OUTPUT_NUM, device=args.gpu)
    test_data2 = SimpleTextDataset(10000, max_length=OUTPUT_NUM, train=False, device=args.gpu)

    train_iter = iterators.SerialIterator(train_data, batch_size=50, shuffle=True)
    test_iter = iterators.SerialIterator(test_data, batch_size=50, repeat=False, shuffle=False)
    test_iter2 = iterators.SerialIterator(test_data2, batch_size=50, repeat=False, shuffle=False)


base_cnn = CNN()
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    base_cnn.to_gpu()
else:
    base_cnn.to_cpu()
#base_cnn_optimizer = optimizers.SGD()
base_cnn_optimizer = optimizers.SGD()
base_cnn_optimizer.setup(base_cnn)

classifiers = []
cl_optimizers = []
for i in range(0, OUTPUT_NUM):
    cl = Classifier()
    if args.gpu >= 0:
        cl.to_gpu()
    else:
        cl.to_cpu()
    #cl_optimizer = optimizers.SGD()
    cl_optimizer = optimizers.SGD()
    cl_optimizer.setup(cl)
    classifiers.append(cl)
    cl_optimizers.append(cl_optimizer)

updater = WordRecogUpdater(train_iter, base_cnn, classifiers, base_cnn_optimizer, cl_optimizers, converter=convert.concat_examples, device=args.gpu)
trainer = training.Trainer(updater, (80, 'epoch'), out=args.output)
#trainer.extend(WordRecogEvaluator([test_iter, test_iter2], base_cnn, classifiers, converter=convert.concat_examples, device=args.gpu))
trainer.extend(WordRecogEvaluator([test_iter], base_cnn, classifiers, converter=convert.concat_examples, device=args.gpu))

trainer.extend(decay_lr(decay_rate=0.98))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'synth/avg_loss', 'synth/levenstein_distance', 'simple/avg_loss', 'simple/levenstein_distance', 'validation/5/accuracy']))
trainer.extend(extensions.ProgressBar())
if args.model_snapshot is not None:
    trainer.extend(extensions.snapshot_object(base_cnn, 'base_cnn_epoch_{.updater.epoch}.npz'), trigger=(args.model_snapshot, 'epoch'))
    for i, cl in enumerate(classifiers):
        trainer.extend(extensions.snapshot_object(cl, 'classifier' + str(i) + '_epoch_{.updater.epoch}.npz'), trigger=(args.model_snapshot, 'epoch'))

if args.resume:
    # Resume from a snapshot
    print("resume from " + args.resume)
    chainer.serializers.load_npz(args.resume + "/base_cnn_epoch_24.npz", base_cnn)
    for i in range(0, OUTPUT_NUM):
        chainer.serializers.load_npz(args.resume + "/classifier%s_epoch_24.npz"%str(i), classifiers[i])

print("start running")
trainer.run()
print("end running")
