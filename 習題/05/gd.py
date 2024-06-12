import numpy as np
from numpy.linalg import norm
from micrograd.engine import Value

def gradientDescendent(f, p0, h=0.01, max_loops=100000, dump_period=1000):
    p = p0.copy()
    for i in range(max_loops):
        fp = f(p)
        fp.backward() 
        gp = [x.grad for x in p] 
        glen = norm(gp)  
        if i % dump_period == 0:
            print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp.data, str([x.data for x in p]), str(gp), glen))
        if glen < 0.00001:  
            break
        gh = np.multiply(gp, -1 * h) 
        p = [x + gh[i] for i, x in enumerate(p)]  
        for x in p:
            x.grad = 0  
    print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp.data, str([x.data for x in p]), str(gp), glen))
    return p  

def f(p):
    [x, y, z] = p
    return (x-1)**2+(y-2)**2+(z-3)**2

p = [Value(0), Value(0), Value(0)]
gradientDescendent(f, p)
