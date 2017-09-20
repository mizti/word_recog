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

#def sample_recog(message):
#    @training.make_extension(trigger=(3, 'iteration'))
#    def _sample_recog(trainer):
#        print(message)
#        #print(trainer.__class__)
#    return _sample_recog

#@training.make_extension(trigger=(3, 'iteration'))
#def sample_recog(trainer):
#    print("hoge")


def sample_recog():
    def _sample_recog(trainer):
        print("hoge")
    _sample_recog.trigger=(3, 'iteration')
    return _sample_recog
