import numpy as np
import chainer
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
            l1 = L.Linear(98304, 4096),
            l2 = L.Linear(4096, 37) 
        )

    def predict(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm2(self.conv2(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm3(self.conv3(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.l1(h))
        y = self.l2(h)
        return y

    def __call__(self, x, t):
        print(t)
        print(t.shape)
        print(t.__class__)
        y = self.predict(x)
        print(h)
        print(h.shape)
        print(h.__class__)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

train_data = TextImageDataset(100, train=True)
test_data = TextImageDataset(100, train=False)
train_iter = iterators.SerialIterator(train_data, batch_size=5, shuffle=True)
test_iter = iterators.SerialIterator(test_data, batch_size=5, repeat=False, shuffle=False)

#chainer.cuda.get_device(0).use()
#model = CNN().to_gpu()
model = CNN().to_cpu()
optimizer = optimizers.SGD()

optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
#updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (500, 'epoch'), out='result')


print("start running")
trainer.extend(extensions.Evaluator(test_iter, model))
#trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
print("end running")
