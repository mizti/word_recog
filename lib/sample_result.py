import copy
import six
import random
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extensions
from chainer import training
from lib.utils import *

def sample_result(dataset):
    @training.make_extension(trigger=(1, 'iteration'))
    def _sample_result(trainer):
        print("hoge")
        base_cnn = trainer.updater.base_cnn
        classifiers = trainer.updater.classifiers
        print(dataset.__class__)
        print(dataset[0])
        label = dataset[0][1]
        data = dataset[0][0][np.newaxis, :]
        #label = dataset[0][1]
        #print(data.__class__)
        print(data.shape)
        h = base_cnn(data)
        print(label.__class__)
        print(label_to_text(label))
        #print(trainer.__class__)
    return _sample_result
