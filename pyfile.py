
#helper file for evaluating or comparing diagrams.
#code adapted from https://github.com/y-richie-y/qpl-represent-matrix/blob/main/gauss.ipynb


from discopy import tensor, Tensor, Dim
import numpy as np

from discopy.quantum.zx import Functor
import tensornetwork as tn

import sys

np.set_printoptions(threshold=sys.maxsize)

dim = 2

def set_dim(new_d):
    global dim
    dim = new_d

def f_ob(ob):
    return Dim(dim) ** len(ob)

def f_ar(box):
    return tensor.Box(box.name, f_ob(box.dom), f_ob(box.cod), box.array)


def eval(diagram, round=True):
    
    d = Functor(ob=f_ob, ar=f_ar, ar_factory=tensor.Diagram)(diagram)
    t = d.eval(contractor=tn.contractors.auto)
    
    n, m = len(diagram.dom), len(diagram.cod)
    #print(n, m, "\n")
    
    if round:
        #return np.round(t.array, 5).astype(complex).reshape(2**n, 2**m).transpose()
        return np.round(t.array, 4).reshape(dim**n, dim**m).transpose()
    else:
        return t.array.reshape(dim**n, dim**m).transpose()
    
    
def eq(a, b, close=True, ish=False):
    a = eval(a, round=False)
    b = eval(b, round=False)
    assert a.shape == b.shape, "wrong shapes"

    if ish: # check for equal up to a number
        if len(a.shape) == 1:
            a /= a[0]
            b /= b[0]
        else:
            a /= a[0][0]
            b /= b[0][0]
    return np.allclose(a, b)
    

