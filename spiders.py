
#defines qubit spiders
#some code adapted from https://github.com/y-richie-y/qpl-represent-matrix/blob/main/gauss.ipynb


from discopy.quantum.zx import Spider, Id, Box
from discopy import tensor, Tensor, Dim
from discopy.rigid import PRO
import numpy as np
import sys

from pyfile import eval

np.set_printoptions(threshold=sys.maxsize)


#Defines n,m Z circle spider
class Z(Spider):
    """ Z spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='Z')
        self.color = "green"
        


    @property
    def array(self):
        n, m = len(self.dom), len(self.cod)
        array = np.zeros(1 << (n + m), dtype=complex)
        array[0] = 1.0
        array[-1] = np.exp(1j * self.phase)
        return Tensor(Dim(2) ** n, Dim(2) ** m, array)
    
    
    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), phase=-self.phase)
    
    def conjugate(self):
        return type(self)(len(self.dom), len(self.cod), phase=-self.phase)
    

#Defines the n,m Z box spider. Same as Z but phase is not exponentiated.
class ZBox(Spider):
    """ Green box. """
    def __init__(self, n_legs_in, n_legs_out, phase=1):
        super().__init__(n_legs_in, n_legs_out, phase, name='ZBox')
        self.color = "green"
        self.shape = 'rectangle'
        

    @property
    def array(self):
        n, m = len(self.dom), len(self.cod)
        array = np.zeros(1 << (n + m), dtype=complex)
        array[0] = 1
        array[-1] = self.phase
        return Tensor(Dim(2) ** n, Dim(2) ** m, array)
    
    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), phase=np.conj(self.phase))
    
    def conjugate(self):
        return type(self)(len(self.dom), len(self.cod), phase=np.conj(self.phase))
    
    
#Given list of phases, gives tensor product of corresponding Z box spiders.  
def boxes(ps, ins=0, outs=1):
    if len(ps) == 1:
        return ZBox(ins, outs, ps[0])
    return ZBox(ins, outs, ps[0]) @ boxes(ps[1:], ins, outs)
   
#Gives binary expansion of x
def bitstring(x):
    return [int(b) for b in "{0:b}".format(x)]

#Defines n, m X spider (only implemeneted for phase of 0, pi)
class X(Spider):
    """ X spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, float(phase), name='X')
        self.color = "red"

    @property
    def array(self):
        
        assert self.phase in (0, 1.0)
        n, m = len(self.dom), len(self.cod)
        array = np.zeros(1 << (n + m), dtype=complex)
        bit = 1 if self.phase == 1.0 else 0
        for i in range(len(array)):
            parity = (bitstring(i).count(1) + bit) % 2
            array[i] = 1 - parity
        return Tensor(Dim(2) ** n, Dim(2) ** m, array)
    
    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), phase=self.phase)
    
    def conjugate(self):
        assert self.phase in (0.0, 1.0) #cos of weird design wheres theres only integer phases
        return type(self)(len(self.dom), len(self.cod), phase=self.phase)
       
    
#returns |10..0> + |01...> + ... |00...1> for m outputs (used to construct statevector for W spider).
def one_hots(n):
    zeros = '0'*n
    strings = []
    for i in range(n):
        strings.append('0'*i + '1' + '0'*(n-i-1))
    return strings

def w_mat(m, n):
    mat = np.zeros((2**m, 2**n)) # rows, columns
    for i in range(2**n - 1):
        bi = format(i, '0' + str(n) + 'b')
        if bi.count('0') == 1:
            mat[0][i] = 1.0

    for j in range(m): # all powers of 2 in final column
        mat[2**j][-1] = 1.0

    return mat

#Defines n, m W spider (or coW)
class W(Spider):
    def __init__(self, n=1, m=2, down=False, norm=False):
        if norm: #optional normalisation factor
            self.norm_factor = max(1, np.sqrt(n))/np.sqrt(m) 
        else:
            self.norm_factor = 1.0
        
        
        
        super().__init__(n, m, 0, name='W')
        self.color = "black"
        self.shape = "triangle_down" if down else "triangle_up"
        
        self.down = down
        self.n = n
        self.m = m
        
    @property
    def array(self):
        if not self.down:
           
            mat = w_mat(self.m, self.n) * self.norm_factor
            return Tensor(Dim(2)**self.n, Dim(2)**self.m, mat.transpose())
        
        else: # flip (and then tranpose later)
            mat = w_mat(self.n, self.m) * self.norm_factor
            return Tensor(Dim(2)**self.n, Dim(2)**self.m, mat)
        
        
           
    def dagger(self):
        return type(self)(n=self.m, m=self.n, down=not self.down)
    
    def conjugate(self):
        return self
    
    

H = Box('H', PRO(1), PRO(1))
H.dagger = lambda: H
H.draw_as_spider = True
H.drawing_name, H.tikzstyle_name, = '', 'H'
H.color, H.shape = "yellow", "rectangle"
H.array = Tensor(Dim(2), Dim(2), 1/np.sqrt(2) * np.array([1.0, 1, 1, -1]))

CX = Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1)
CZ = Z(1, 2) @ Id(1) >> Id(1) @ H @ Id(1) >> Id(1) @ Z(2, 1)


Swap = Id(2).swap(1, 1)

FSwap = Box('O', PRO(2), PRO(2), data=eval(Swap >> CZ).flatten(), draw_as_spider=True)
FSwap.array = FSwap.data
FSwap.color = 'white'
