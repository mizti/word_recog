import six
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, training, reporter
from chainer.datasets import get_mnist
from chainer.training import trainer, extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.datasets import get_mnist
from chainer import optimizer as optimizer_module

class WordRecogUpdater(training.StandardUpdater):
	def __init__(self, iterator, base_cnn, model1, model2, base_cnn_optimizer, model1_optimizer, model2_optimizer, converter=convert.concat_examples, device=None):
		if isinstance(iterator, iterator_module.Iterator):
			iterator = {'main':iterator}
		self._iterators = iterator
		self.base_cnn = base_cnn
		self.model1 = model1
		self.model2 = model2

		if isinstance(base_cnn_optimizer, optimizer_module.Optimizer) and isinstance(model1_optimizer, optimizer_module.Optimizer) and isinstance(model2_optimizer, optimizer_module.Optimizer):
			optimizer = {
				'base_cnn_opt': base_cnn_optimizer, 
				'model1_opt': model1_optimizer,
				'model2_opt': model2_optimizer
			}
		self._optimizers = optimizer
		self.converter = convert.concat_examples
		self.device = device
		self.iteration = 0

	def update_core(self):
		iterator = self._iterators['main'].next()
		in_arrays = self.converter(iterator, self.device)

		xp = np if int(self.device) == -1 else cuda.cupy
		x_batch = xp.array(in_arrays[0])
		t_batch = xp.array(in_arrays[1])
		t_batch1 = t_batch[:,0]
		t_batch2 = t_batch[:,1]
		y = self.base_cnn(x_batch)

		loss1 = self.model1(y,t_batch1)
		loss2 = self.model2(y,t_batch2)

		reporter.report({'loss1':loss1, 'loss2':loss2})
		loss_dic = {'loss1':loss1, 'loss2':loss2}
		print("loss1="+str(loss1.data))
		print("loss2="+str(loss2.data))
	
		for name, optimizer in six.iteritems(self._optimizers):
			optimizer.target.cleargrads()

		for name, loss in six.iteritems(loss_dic):	
			loss.backward()

		for name, optimizer in six.iteritems(self._optimizers):
			optimizer.update()
