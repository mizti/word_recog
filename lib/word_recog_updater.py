import six
import numpy as np
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
    def __init__(self, iterator, base_cnn, classifiers, base_cnn_optimizer, cl_optimizers, converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self.base_cnn = base_cnn
        self.classifiers = classifiers

        self._optimizers = {}
        self._optimizers['base_cnn_opt'] = base_cnn_optimizer
        for i in range(0, len(cl_optimizers)):
            self._optimizers[str(i)] = cl_optimizers[i]

        self.converter = convert.concat_examples
        self.device = device
        self.iteration = 0

    def update_core(self):
        chainer.using_config('train', True)
        iterator = self._iterators['main'].next()
        in_arrays = self.converter(iterator, self.device)

        xp = np if int(self.device) == -1 else cuda.cupy
        x_batch = xp.array(in_arrays[0])
        t_batch = xp.array(in_arrays[1])
        y = self.base_cnn(x_batch)

        loss_dic = {}
        for i, classifier in enumerate(self.classifiers):
            loss = classifier(y, t_batch[:,i])
            #print(str(i) + " " +str(loss.data))
            loss_dic[str(i)] = loss
            #reporter.report({'loss'+str(i):loss})
        #print("\n")

        #loss_dic = {'loss1':loss1, 'loss2':loss2}
        #reporter.report({'loss1':loss1, 'loss2':loss2})

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.target.cleargrads()

        for name, loss in six.iteritems(loss_dic):
            loss.backward()

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.update()
