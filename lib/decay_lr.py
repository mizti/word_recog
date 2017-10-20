import sys
import copy
import six
import random
import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extensions
from chainer import training
from lib.utils import *

def decay_lr(decay_rate=0.98):
    @training.make_extension(trigger=(1, 'epoch'))
    def _decay_lr(trainer):
        for name, opt in six.iteritems(trainer.updater._optimizers):
            opt.lr *= decay_rate
    return _decay_lr
