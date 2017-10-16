import copy
import six
import random
import distance
import chainer
from chainer import cuda
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
    default_name='myval'
    def __init__(self, iterator_list, base_cnn, classifiers, converter=convert.concat_examples, device=None, eval_hook=None, eval_func=None):
        iterators = {'synth': iterator_list[0], 'simple': iterator_list[1]}
        self._iterators = iterators
        self.base_cnn = base_cnn
        self._targets = {} 
        for i, cl in enumerate(classifiers):
            self._targets[str(i)] = cl

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func
        self.xp = np if int(self.device) == -1 else cuda.cupy

    def evaluate(self):
        chainer.using_config('train', False) # not need: already set in __call__

        summary = reporter_module.DictSummary()
        targets = self._targets

        for itr_name, iterator in six.iteritems(self._iterators):
            if hasattr(iterator, 'reset'):
                iterator.reset()
                it = iterator
            if self.eval_hook:
                self.eval_hook(self)

            else:
                # make a shallow copy of iterator
                it = copy.copy(iterator)

            print("itr_name")
            print(itr_name)
            #print("it")
            #print(it)
            for i, iterator in enumerate(it):
                #print("iterator.__class__") 
                #print(iterator.__class__)#<type 'list'>
                observation = {}
                with reporter_module.report_scope(observation):
                    in_arrays = self.converter(iterator, self.device)

                    with function.no_backprop_mode():
                        np.set_printoptions(threshold=np.inf)
                        x_batch = self.xp.array(in_arrays[0])
                        t_batch = self.xp.array(in_arrays[1])
                        #print("x_batch" + str(i))
                        #print(x_batch)
                        y = self.base_cnn(x_batch)
                        predicted_words = []
                        l_distance = 0
                        loss = []
                        for name, cl in six.iteritems(targets):
                            #print(y) # included many nan
                            cl_loss = cl(y, t_batch[:,int(name)])
                            # print(cl_loss) # variable(nan)
                            loss.append(cl_loss)
                            predicted_words.append(cl.predict(y))
                        # print(sum(loss)) # variable(nan)
                        avg_loss = sum(loss) / len(loss)
                        summary.add({itr_name + '/avg_loss': avg_loss})
                        for i in range(len(in_arrays[0])):
                            recoged_word = []
                            recoged = label_to_text(map(lambda x:x.data[i].argmax(), predicted_words))
                            label = label_to_text(in_arrays[1][i])
                            #print("recog: " + recoged)
                            #print("label: " + label)
                            l_distance = distance.levenshtein(recoged, label)
                            summary.add({itr_name + '/levenstein_distance': float(l_distance)})

                summary.add(observation)
        return summary.compute_mean()
