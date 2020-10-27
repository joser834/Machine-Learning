import numpy as np

# Part 1 -----------------------------------------------------------------------
x = np.array([3,14,1])
W = np.array([
    [1,0,-1],
    [0,1,-1],
    [-1,0,-1],
    [0,-1,-1]
])
V = np.array([
    [1, 1, 1, 1, 0],
    [-1, -1, -1, -1, 2]
])

z = np.append(np.where(np.dot(x,W.T)<=0,0,np.dot(x,W.T)), 1)
#z = np.array([2,13,-6,-6,1])
u = np.where(np.dot(z,V.T)<=0,0,np.dot(z,V.T))

o1 = np.exp(u[0])/(np.exp(u[0])+np.exp(u[1]))
o2 = np.exp(u[1])/(np.exp(u[0])+np.exp(u[1]))
print(o1, o2)

# Part 2 -----------------------------------------------------------------------
Wfh = 0
Wfx = 0
bf = -100
Wch = -100
Wih = 0
Wix = 0
bi = 100
Wcx = 50
Woh = 0
Wox = 100
bo = 0
bc = 0

h = 0
c = 0
X = np.array([0,0,1,1,1,0])

for x in X:
    A = Wfh*h + Wfx*x + bf
    ft = 1/(1+np.exp(-A))

    B = Wih*h + Wix*x + bi
    it = 1/(1+np.exp(-B))

    C = Woh*h + Wox*x + bo
    ot = 1/(1+np.exp(-C))

    D = Wch*h + Wcx*x + bc
    c = ft*c + it*np.tanh(D)

    h = np.round(ot*np.tanh(c),0)
    print(h)

