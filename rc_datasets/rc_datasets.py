
# numerical packages
import numpy as np
import matplotlib.pyplot as plt

class rc_dataset:
    u=[]
    t=[]

    def __init__(self,u1,t1):
        self.u=u1
        self.t=t1

    def plot(self,nfig=1):
        ff=plt.figure(nfig)
        plt.plot(self.t,self.u)
        return ff

def mackeyglass(fac,Nt,dt,x0,a=0.2,b=0.1,tau=17):
    # fac=10;                                 # interpolation factor (see lit.)
    # Nt=fac*(Ttot);                          # number of timesteps (total);
    # x0=1.8;                                 # initial condition (lit.)
    # dt=0.1;                                 # time-step (lit.)
    
    # mackeyglass_raw(sample_n,deltat,x0,a=0.2,b=0.1,tau=17):
    X, T = mackeyglass_raw(Nt,dt,x0,a,b,tau)
    
    #interpolate 
    u0=X[np.arange(0, len(X), fac)]
    t0=T[np.arange(0, len(T), fac)]

    u0=u0-np.min(u0)+1
    u0=u0/np.max(u0)

    return rc_dataset(u0,t0)

def mackeyglass_raw(sample_n,deltat,x0,a,b,tau):
    time=0
    index=0
    history_length=int(np.floor(tau/deltat))
    x_history=np.zeros(history_length)
    x_t=x0
    x_t_minus_tau=0 # here we assume x(t)=0 for -tau <= t < 0

    X=np.zeros(sample_n+history_length)
    T=np.zeros(sample_n+history_length)

    for ii in np.arange(sample_n+history_length):
        X[ii] = x_t;
        if tau == 0:
            x_t_minus_tau = 0.0;
        else:
            x_t_minus_tau = x_history[index]
        
        x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b)
        
        if (tau!=0):
            x_history[index] = x_t_plus_deltat
            index = int(index%(history_length-1)+1)

        time = time + deltat
        T[ii] = time
        x_t = x_t_plus_deltat

    return X[-sample_n:], T[-sample_n:]

def mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b):
    k1 = deltat* mackeyglass_eq(x_t,          x_t_minus_tau, a, b);
    k2 = deltat* mackeyglass_eq(x_t+0.5*k1,   x_t_minus_tau, a, b);
    k3 = deltat* mackeyglass_eq(x_t+0.5*k2,   x_t_minus_tau, a, b);
    k4 = deltat* mackeyglass_eq(x_t+k3,       x_t_minus_tau, a, b);
    x_t_plus_tau = x_t + k1/6. + k2/3. + k3/3. + k4/6.
    return x_t_plus_tau

def mackeyglass_eq(x_t,x_t_minus_tau,a,b):
    # This function returns dx/dt of Mackey-Glass delayed differential equation
    # $$\frac{dx(t)}{dt}=\frac{ax(t-\tau)}{1+x(t-\tau)^{10}}-bx(t)$$
    return -b * x_t + (a * x_t_minus_tau)/(1. + x_t_minus_tau**10.0)





