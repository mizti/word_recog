import numpy as np
import cupy
from chainer import Variable

def text_to_label(text, device=-1):
    label = []
    for index, c in enumerate(text):
        label.append(char_to_int(c))
    label = np.asarray(label).astype('int32')
    if device >= 0:
        label = chainer.cuda.to_gpu(label)
    return label

# returns 0~9: number / 10: space / 11~36: upper case
# currently don't accept small case alphabets
def char_to_int(c):
    ascii_code = ord(c)
    if (ascii_code >=48 and ascii_code <= 57):
        # print("number")
        ascii_code = ascii_code - 48
    elif (ascii_code >=65 and ascii_code <= 90):
        # print("upper case")
        ascii_code = ascii_code - 54
    elif (ascii_code == 32):
        # print("space")
        ascii_code = 10
    else:
        raise ValueError("not a alphanumeric character")
    return ascii_code


def label_to_text(label):
    if isinstance(label, Variable):
        label = label.data
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if isinstance(label, cupy.ndarray):
        label = label.tolist()

    chars = "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_list = list(chars)
    ret = ""
    for i in label:
       ret = ret + char_list[i]
    return ret

li = [9,20,23,26]
ndarray = np.asarray(li)
var = Variable(ndarray)

print(label_to_text(li))
print(label_to_text(ndarray))
print(label_to_text(var))

st = "9JMP"
print(text_to_label(st))
