import copy
import six
import random
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import link
from chainer import reporter as reporter_module
#from chainer.training import extension
from chainer.training import extensions
from lib.utils import *

class WordRecogEvaluator(extensions.Evaluator):
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

        print_now = False
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    h = self.base_cnn(in_arrays[0])
                    recoged_word = []
                    for name, cl in six.iteritems(targets):
                        loss = cl(h, in_arrays[1][:,int(name)])
                        recoged_word.append(cl.predict(h).data[0].argmax())
                    labeled_word = in_arrays[1][0]
                    #print("label: " + label_to_text(labeled_word))
                    #print("recog: " + label_to_text(recoged_word))
            summary.add(observation)
        return summary.compute_mean()
