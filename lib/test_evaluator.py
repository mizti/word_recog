import copy
import six
import random
import distance
import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import link
from chainer import reporter as reporter_module
#from chainer.training import extension
from chainer.training import extensions
from lib.utils import *

class TestEvaluator(extensions.Evaluator):
    default_name='validation'
    def __init__(self, iterator, base_cnn, classifiers, converter=convert.concat_examples, device=None, eval_hook=None, eval_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.base_cnn = base_cnn
        self._targets = {}
        for i, cl in enumerate(classifiers):
            self._targets[str(i)] = cl

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func

    def evaluate(self):
        chainer.using_config('train', False)
        iterator = self._iterators['main']
        targets = self._targets

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            # make a shallow copy of iterator
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        #for i, batch in enumerate(it):
        #    observation = {}
        #    with reporter_module.report_scope(observation):
        #        in_arrays = self.converter(batch, self.device)
        #        with function.no_backprop_mode():
        #            loss = i
        #            chainer.reporter.report({'hoge': loss})
        #    summary.add(observation)
        #    print(summary)
        observation = {}
        with reporter_module.report_scope(observation):
            chainer.reporter.report({'fuga': 4.0})
            print(observation)
        summary.add(observation)
        print(summary._summaries) # chainer.reporter.DictSummary

        observation = {}
        with reporter_module.report_scope(observation):
            chainer.reporter.report({'fuga': 5.0})
            print(observation)
        summary.add(observation)
        print(summary._summaries) # chainer.reporter.DictSummary
        print("mean")
        print(summary.compute_mean())
        print(summary.compute_mean().__class__) #dict

        #return summary.compute_mean()
        return {"piyo": 8.9}
