#!/usr/bin/env python
# coding: utf-8


import numpy as np

########################################## Método de Dormand-Prince ####################################

def DP(f1,f2,f3,f4,x0,y0,z0,w0,T,n,order=5):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    y=np.empty(n)
    z=np.empty(n)
    w=np.empty(n)
    
    x[0]=x0
    y[0]=y0
    z[0]=z0
    w[0]=w0
    
    for i in range(n-1):
        k1=f1(t[i]       ,x[i],y[i],z[i],w[i])
        k2=f1(t[i]+1/5*h ,x[i]+h*k1/5,y[i]+h*k1/5,z[i]+h*k1/5,w[i]+h*k1/5)
        k3=f1(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2),y[i]+h*(3/40*k1+9/40*k2),z[i]+h*(3/40*k1+9/40*k2),w[i]+h*(3/40*k1+9/40*k2))
        k4=f1(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3),y[i]+h*(44/45*k1-56/15*k2+32/9*k3),z[i]+h*(44/45*k1-56/15*k2+32/9*k3),w[i]+h*(44/45*k1-56/15*k2+32/9*k3))
        k5=f1(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),y[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),z[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),w[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4))
        k6=f1(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),y[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),z[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),w[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5))
        k7=f1(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6))
        if order==5:
            x[i+1]=x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        else:
            x[i+1]=x[i]+h*(5179/57600*k1+7571/16695*k3+393/640*k4-92097/339200*k5+187/2100*k6+1/40*k7)
        
        k1=f2(t[i]       ,x[i],y[i],z[i],w[i])
        k2=f2(t[i]+1/5*h ,x[i]+h*k1/5,y[i]+h*k1/5,z[i]+h*k1/5,w[i]+h*k1/5)
        k3=f2(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2),y[i]+h*(3/40*k1+9/40*k2),z[i]+h*(3/40*k1+9/40*k2),w[i]+h*(3/40*k1+9/40*k2))
        k4=f2(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3),y[i]+h*(44/45*k1-56/15*k2+32/9*k3),z[i]+h*(44/45*k1-56/15*k2+32/9*k3),w[i]+h*(44/45*k1-56/15*k2+32/9*k3))
        k5=f2(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),y[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),z[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),w[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4))
        k6=f2(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),y[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),z[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),w[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5))
        k7=f2(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6))
        if order==5:
            y[i+1]=y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        else:
            y[i+1]=y[i]+h*(5179/57600*k1+7571/16695*k3+393/640*k4-92097/339200*k5+187/2100*k6+1/40*k7)
        
        k1=f3(t[i]       ,x[i],y[i],z[i],w[i])
        k2=f3(t[i]+1/5*h ,x[i]+h*k1/5,y[i]+h*k1/5,z[i]+h*k1/5,w[i]+h*k1/5)
        k3=f3(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2),y[i]+h*(3/40*k1+9/40*k2),z[i]+h*(3/40*k1+9/40*k2),w[i]+h*(3/40*k1+9/40*k2))
        k4=f3(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3),y[i]+h*(44/45*k1-56/15*k2+32/9*k3),z[i]+h*(44/45*k1-56/15*k2+32/9*k3),w[i]+h*(44/45*k1-56/15*k2+32/9*k3))
        k5=f3(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),y[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),z[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),w[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4))
        k6=f3(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),y[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),z[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),w[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5))
        k7=f3(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6))
        if order==5:
            z[i+1]=z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        else:
            z[i+1]=z[i]+h*(5179/57600*k1+7571/16695*k3+393/640*k4-92097/339200*k5+187/2100*k6+1/40*k7)
        
        k1=f4(t[i]       ,x[i],y[i],z[i],w[i])
        k2=f4(t[i]+1/5*h ,x[i]+h*k1/5,y[i]+h*k1/5,z[i]+h*k1/5,w[i]+h*k1/5)
        k3=f4(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2),y[i]+h*(3/40*k1+9/40*k2),z[i]+h*(3/40*k1+9/40*k2),w[i]+h*(3/40*k1+9/40*k2))
        k4=f4(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3),y[i]+h*(44/45*k1-56/15*k2+32/9*k3),z[i]+h*(44/45*k1-56/15*k2+32/9*k3),w[i]+h*(44/45*k1-56/15*k2+32/9*k3))
        k5=f4(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),y[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),z[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),w[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4))
        k6=f4(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),y[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),z[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),w[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5))
        k7=f4(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6))
        if order==5:
            w[i+1]=w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        else:
            w[i+1]=w[i]+h*(5179/57600*k1+7571/16695*k3+393/640*k4-92097/339200*k5+187/2100*k6+1/40*k7)
            
        
    return t,x,y,z,w
################################################# Funciones de Error#####################################################

def Error_Abs(aprox,real):
    err=abs(aprox-real)
    return err

def Error_Rela(aprox,real):
    err=abs(aprox-real)/abs(real)
    return err


def MSE(aprox,real):
    mse=np.square(aprox-real).mean()
    return mse

#########################################################################################################################