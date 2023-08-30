#file for qudit spiders
#generalises classes from spiders.py


from discopy.quantum.zx import Spider, Id, Box
from discopy import tensor, Tensor, Dim
from discopy.rigid import PRO
import numpy as np
import sys

import pyfile

d = 2

#helper functions

#updates global variable d (should be constant throughout file when used)
def set_d(new_d):
    global d
    d = new_d
    pyfile.set_dim(new_d)
    print("set d to", d)
    
#return basis state |i> (or <i| when bra=True)
def basis(i, bra=False):
    assert 0 <= i < d, "i out of range"
    k = np.zeros((d, 1))
    k[i][0] = 1
    if bra:
        return k.transpose()
    else:
        return k
    
#Returns |x>^n (x is vector)  
def T(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    return np.kron(x, T(x, n-1))

#Returns x^n (x is diagram)
def T_diag(x, n):
    if n == 0:
        return Id(0)
    if n == 1:
        return x
    return x @ T_diag(x, n-1)
    
    
#Defines n,m Z circle spider
class Z(Spider):
    """ Z spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='Z')
        self.color = "green"
    
    #Gives the state vector: sum_j e^ia_j |j><j|
    @property
    def array(self):
        if self.phase == 0:
            phase = np.zeros(d)
        else:
            assert len(self.phase) == d and self.phase[0] == 0, "bad phase"
            phase = self.phase
            
        n, m = len(self.dom), len(self.cod)
        z = np.zeros((d**m,d**n), dtype=complex)

        for i in range(d):
            z += np.exp(1j * phase[i]) * T(basis(i), m) * T(basis(i, bra=True), n)
    
        return Tensor(Dim(d) ** n, Dim(d) ** m, z.transpose())

#Defines the n,m Z box spider. Same as Z but phase is not exponentiated.
class ZBox(Spider):
    """ Z spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='ZBox')
        self.color = "green"
        self.shape = 'rectangle'
        
    @property
    def array(self):
        if self.phase == 0:
            phase = np.ones(d)
        else:
            assert len(self.phase) == d and self.phase[0] == 1, "phases should be length d with first parameter 1"
            phase = self.phase
            
        n, m = len(self.dom), len(self.cod)
        z = np.zeros((d**m,d**n))

        for i in range(d):
            z += phase[i] * T(basis(i), m) * T(basis(i, bra=True), n)
    
        return Tensor(Dim(d) ** n, Dim(d) ** m, z.transpose())
    
    
#Given list of phases, gives tensor product of corresponding Z box spiders.  
def boxes(ps, ins=0, outs=1):
    if len(ps) == 1:
        return ZBox(ins, outs, ps[0])
    return ZBox(ins, outs, ps[0]) @ boxes(ps[1:], ins, outs)


#returns |i0..0> + |0i...> + ... |00...i> for m outputs (used to construct statevector for W spider).
def one_hots(m, i):
   
    x = np.zeros((d**m, 1))
    for j in range(m):
        x += np.kron(np.kron(T(basis(0), j),  basis(i)), T(basis(0), m-j-1))

    return x

#Builds the matrix for 
def w_mat(m):
    mat = T(basis(0), m) * basis(0, bra=True)
    for i in range(1, d):
        mat += one_hots(m, i) * basis(i, bra=True)
    return mat
    
#Defines 1, m W spider
class W(Spider):
    def __init__(self, n=1, m=2, down=False, norm=False):
        if norm: # optional normalisation factor
            self.norm_factor = max(1, np.sqrt(n))/np.sqrt(m) 
        else:
            self.norm_factor = 1.0
        
        #assert not down, "havent worked out tranpose yet"
        
        assert n == 1 or (down and m==1), "multiple inputs not implemented yet"
        
        super().__init__(n, m, 0, name='W')
        self.color = "black"
        self.shape = "triangle_down" if down else "triangle_up"
        
        self.down = down
        
    @property
    def array(self):
        if not self.down:
            n, m = 1, len(self.cod)
            mat = w_mat(m)
        
            return Tensor(Dim(d)**n, Dim(d)**m, mat.transpose())
        
        else: # flip (and then tranpose later)
            n, m = len(self.dom), len(self.cod)
            
            mat = w_mat(n)
            return Tensor(Dim(d)**n, Dim(d)**m, mat)
        
        
    #flips inputs and ouputs and turns triangle upside down
    def dagger(self):
        return type(self)(n=self.m, m=self.n, down=not self.down)
    
    

class H(Spider):
    def __init__(self):
        super().__init__(1, 1, 0, name='H')
        self.shape = 'rectangle'
        self.color = 'yellow'
        
        
    @property
    def array(self):
        #omega = np.exp(2j * np.pi/d)
        p = [1] + [0]*(d-1) + [-1]
        omega = np.roots(p)[0]
        
        hmat = np.zeros((d, d), dtype=complex)
        for j in range(d):
            for k in range(d):
                hmat += omega**(j * k) * basis(j) * basis(k, bra=True)
        
        return hmat/np.sqrt(d)
        
def H_dagger():
    Hd = Box('†', PRO(1), PRO(1), data=pyfile.eval(H() >> H() >> H()).flatten(), draw_as_spider=True)
    Hd.array = Hd.data
    Hd.shape = 'rectangle'
    Hd.color = 'yellow'
    
    return Hd

Swap = Id(2).swap(1, 1)


class X(Spider):
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='X')
        self.color = "red"
        
    @property
    def array(self):
        n, m = len(self.dom), len(self.cod)
        
        s = T_diag(H_dagger(), n) >> Z(n, m, self.phase) >> T_diag(H(), m)
        scalar = d**((m+n-2)/2)
        return Tensor(Dim(d)**n, Dim(d)**m, pyfile.eval(s).transpose() * scalar)