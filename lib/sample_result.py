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
    @training.make_extension(trigger=(5, 'epoch'))
    def _sample_result(trainer):
        base_cnn = trainer.updater.base_cnn
        classifiers = trainer.updater.classifiers
        label = dataset[0][1]
        data = dataset[0][0][np.newaxis, :]
        h = base_cnn(data)
        recoged_word = []
        for i, cl in enumerate(classifiers):
            recoged_word.append(cl.predict(h).data[0].argmax())
        print("recoged: " + label_to_text(recoged_word))
        print("label:   " + label_to_text(label))
    return _sample_result
