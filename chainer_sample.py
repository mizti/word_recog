import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.cuda import cupy
l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)

def my_foraward(x):
	h = l1(x)
	return l2(h)

class MyChain(Chain):
	def __init__(self):
		super(MyChain, self).__init__(
			l1 = L.Linear(4, 3),
			l2 = L.Linear(3, 2),
		)
	def __call__(self, x):
		h = self.l1(x)
		h = self.l2(h)
		return h

class MyChain2(ChainList):
	def __init__(self):
		super(MyChain2, self).__init__(
			L.Linear(4, 3),
			L.Linear(3, 2),
		)

	def __call__(self, x):
		h = self[0](x)
		return self[1](h)

model = MyChain()
optimizer = optimizers.SGD()
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

x = np.random.uniform(-1, 1, (6, 4)).astype(np.float32)
model.cleargrads()
loss = F.sum(model(chainer.Variable(x)))
print(loss)
