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

class WordRecogUpdater(training.StandartUpdater):
	def __init__(self, iterator, batch_size, base_cnn, model1, model2, base_cnn_optimizer, model1_optimizer, model2_optimizer, converter=convert.concat_examples, device=None):
		if isinstance(iterator, iterator_module.Iterator):
			iterator = {'main':iterator}
		self.iterators = iterator
		self.batch_size = batch_size
		self.base_cnn = base_cnn
		self.model1 = model1
		self.model2 = model2

		if isinstance(base_cnn_optimizer, optimizer_module.Optimizer) and isinstance(model1_optimizer, optimizer_module.Optimizer) and isinstance(model2_optimizer, optimizer_module.Optimizer):
			optimizer = {
				'base_cnn_opt': base_cnn_optimizer, 
				'model1_opt': model1_optimizer,
				'model2_opt': model2_optimizer
			}
		self.optimizers = optimizer
		self.converter = convert.concat_examples
		self.device = device

	def update_core(self):
		iterator = self.iterators['main'].next()
		in_arrays = self.converter(batch, self.device)

		y = self.base_cnn(x_batch)
		loss1 = self.model1(y)
		loss2 = self.model2(y)
		reporter.report({'loss1':loss1, 'loss2':loss2})
		loss_dic = {'loss1':loss1, 'loss2':loss2}

		# cleargrads all of models

		# backward loss1 and loss2

		# update each models and base_cnn
