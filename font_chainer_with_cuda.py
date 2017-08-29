import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from font_image_dataset import *

train_data = FontImageDataset(1000, train=True) # returns 64 x 64 ndarray, 
test_data = FontImageDataset(1000, train=False)
train_iter = iterators.SerialIterator(train_data, batch_size=20, shuffle=True)
test_iter = iterators.SerialIterator(test_data, batch_size=20, repeat=False, shuffle=False)


class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            l1 = L.Linear(None, n_units),
            l2 = L.Linear(None, n_units),
            l3 = L.Linear(None, n_out)
        ) 
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(x))
        y = self.l3(h2)
        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

chainer.cuda.get_device(0).use()
model = L.Classifier(MLP(100, 12)).to_gpu()
optimizer = optimizers.SGD()

optimizer.setup(model)

#updater = training.StandardUpdater(train_iter, optimizer)
updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (500, 'epoch'), out='result')


print("start running")
#trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
print("end running")
