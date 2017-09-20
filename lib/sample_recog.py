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

def sample_recog():
    #@training.make_extension(trigger=(1, 'epoch'))
    a = "fuga"
    @training.make_extension(trigger=(1, 'iteration'))
    def _sample_recog(trainer):
        print(a) # can see a
        print("hoge")
        #base_cnn = trainer.updater.base_cnn
        #classifiers = trainer.updater.classifiers
        #data = test_data#TextImageDatase(1, train=False, device=trainer.updater.device)
    return _sample_recog
