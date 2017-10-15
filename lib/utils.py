import sys
import numpy as np
try:
    import cupy
except ImportError:
    pass
from chainer import Variable

# returns 0~9: number / 10: space / 11~36: upper case /37: empty
#CHARS = "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" #36 chars
EMPTY_CODE=len(CHARS)

def text_to_label(text, length=0 ,device=-1):
    #print(text)
    if length > 0 and len(text)>length:
        raise ValueError("text length exceeds expected max length!")
    label = []
    for index, c in enumerate(text):
        label.append(char_to_int(c))
    if isinstance(length, int) and length>0:
        while len(label) < length:
            label.append(char_to_int(""))

    label = np.asarray(label).astype('int32')
    if device >= 0:
        label = chainer.cuda.to_gpu(label)
    return label

def char_to_int(c):
    if c is "":
        ret = EMPTY_CODE
    else:
        ret = CHARS.index(c)
    return ret


def label_to_text(label):
    if isinstance(label, Variable):
        label = label.data
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if 'cupy' in sys.modules:
        if isinstance(label, cupy.ndarray):
            label = label.tolist()

    char_list = list(CHARS)
    ret = ""
    for i in label:
        if isinstance(i, cupy.core.core.ndarray):
            i = int(i)
        if i >= len(CHARS): #if i exceeds the lenghth of CHARS, it is ""
            ret = ret + "" # same to do nothing
        else:
            ret = ret + char_list[i]
            #print(ret)
    return ret
