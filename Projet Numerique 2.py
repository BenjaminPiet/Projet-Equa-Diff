import numpy as np
import matplotlib.pyplot as plt

def solve_euler_explicit(f,x0,dt,n) :
    """
    Entrées :
        f, une fonction
        x0, condition initiale (x(0)=x0)
        dt, pas de temps
        n, nombre de points
    Sortie :
    renvoie le vecteur des temps tj et de la solution xj du schéma d'Euler explicite appliqué à dx/dt = f(x)
    """
    T = np.linspace(0,dt*(n-1), n)
    X = np.zeros(n)
    X[0]=x0
    for j in range(1,n) :
        X[j] = X[j-1] + dt*f(X[j-1])
    return(T,X)

def solve_heun_explicit (f,x0,dt,n) :
    """
    Entrées :
        f, une fonction
        x0, condition initiale (x(0)=x0)
        dt, pas de temps
        n, nombre de points
    Sortie :
    renvoie le vecteur des temps tj et de la solution xj du schéma de Heun explicite appliqué à dx/dt = f(x)
    """
    T = np.linspace(0,dt*(n-1), n)
    X = np.zeros(n)
    X[0]=x0
    for j in range(1,n) :
        X[j] = X[j-1] + (dt/2) * ( f(X[j-1]) + f( X[j-1] + dt*f(X[j-1]) ) )
    return(T,X)

## TEST

def f1(x) :
    return x

n=5000
x0=1
dt=10**-2
T,X1=solve_euler_explicit(f1,x0,dt,n)
T,X2=solve_heun_explicit(f1,x0,dt,n)
Y = np.exp(T)
plt.plot(T,X1,'r')
plt.plot(T,X2,'g')
plt.plot(T,Y,'b')


plt.show()