#file for converting statevectors to polynomial representation.

from sympy import Poly, symbols
import numpy as np


def state_to_polystring(state, var_list=None):
    if not var_list:
        size = int(np.log2(state.shape[0]))
        var_list = symbols(['x_' + str(i) for i in range(size)])
    
    n = len(var_list)
    assert state.shape == (2**n, 1)
    poly_string = 0
    for i, a_i in enumerate(state[:, 0]):
        connections = format(i, '0' + str(n) + 'b')
        mon = a_i
        for k, v_k in enumerate(var_list):
            if connections[k] == '1':
                mon *= v_k
        
        poly_string += mon
        
    return poly_string

def get_poly(state, n_A=0, n_B=0):
    if n_A == 0 == n_B: #make half split
        n_A = n_B = int(np.log2(state.shape[0]))//2
    
    assert state.shape == (2**(n_A + n_B), 1), ("wrong sizes", n_A, n_B, state.shape)
        
    x_vars = symbols(['x_' + str(i) for i in range(n_A)])
    y_vars = symbols(['y_' + str(i) for i in range(n_B)])
    all_vars = x_vars + y_vars
    
    poly_string = state_to_polystring(state, all_vars) # still not tecnically a polynomial
    
    if n_B > 0 and False:
        poly = Poly(poly_string, domain='RR' + str(y_vars))# might want to be y_vars but whatever
    else:
        poly = Poly(poly_string, all_vars)
    
    return poly

def is_prod(state, n_A=None, n_B=None):
    if n_A is None and n_B is None: #make half split
        n_A = n_B = int(np.log2(state.shape[0]))//2
    assert (2**(n_A + n_B), 1) == state.shape, ("wrong sizes", n_A, n_B, state.shape)
    
    
    _, factors = get_poly(state, n_A, n_B).factor_list()
    #if len(factors) == 0: return True
    for f in factors:
        #if f has both x's and y's
        degrees = f[0].degree_list()
        if np.any(degrees[:n_A]) and np.any(degrees[n_A:]):
            return False
    return True