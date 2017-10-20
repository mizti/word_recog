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

def sample_result(dataset, output_dir="result"):
    with open(output_dir + '/sample_result.txt', mode='w') as f:
        f.write("")
    @training.make_extension(trigger=(1, 'epoch'))
    def _sample_result(trainer):
        chainer.using_config('train', False)
        base_cnn = trainer.updater.base_cnn
        classifiers = trainer.updater.classifiers
        label = dataset[0][1]
        data = dataset[0][0][np.newaxis, :]
        h = base_cnn(data)
        recoged_word = []
        for i, cl in enumerate(classifiers):
            content = cl.predict(h).data[0].argmax()
            if 'cupy' in sys.modules:
                if isinstance(content, cupy.ndarray):
                    content = cupy.asnumpy(content)
            recoged_word.append(content)

        t = "label  : " + label_to_text(label) + "  " + "recoged: " + label_to_text(recoged_word) + "\n"
        with open(output_dir + '/sample_result.txt', mode='a') as f:
            f.write(t)

    return _sample_result
