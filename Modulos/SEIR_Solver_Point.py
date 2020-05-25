#!/usr/bin/env python
# coding: utf-8

# Importar librerias Necesarias

import numpy as np

########################### Modelo SEIR ###############################################

# Modelo SEIR:
#
# Parámetros Experimentales:
#
# β(t)=Tasa de Transmision
# σ(t)=Tasa de Incubación
# γ(t)=Tasa de Recuperación
# μ(t)=Tasa de Natalidad (Mortalidad Hans)
# ν(t)=Tasa de Mortalidad (Vacuna Hans)
#   n = Número de Particiones en el Dominio Temporal (Default=100)

def SEIR(T,β,σ,γ,μ,ν,n=100):
    
    ## Condiciones Iniciales
    ## Depende de Datos Reales y será modificada con alguna Base de Datos en Tiempo Real
    ## Entonces, luego de estas pruebas, deberá ser un parámetro de Entrada
    
    S0=10 # Susceptibles Iniciales (Debería incluir Nacimientos)
    E0=1  # Expuestos    Iniciales (Debería incluir Nacimientos)
    I0=0  # Infectados   Iniciales (Debería incluir Nacimientos)
    R0=0  # Recuperados  Iniciales (Debería incluir Nacimientos)
    N=S0+E0+I0+R0 # Población Total
    
    print("Población Total:",N)
    
    # Modelo de Ecuaciones Diferenciales SEIR (IDM, No Vital D.)
    
    '''
    dSdt=lambda t,S,E,I,R,β,σ,γ: -β*S*I/N
    dEdt=lambda t,S,E,I,R,β,σ,γ:  β*S*I/N-σ*E
    dIdt=lambda t,S,E,I,R,β,σ,γ:      σ*E-γ*I
    dRdt=lambda t,S,E,I,R,β,σ,γ:      γ*I
    '''
    
    # Modelo de Ecuaciones Diferenciales SEIR (IDM. Vital D.)
    
    dSdt=lambda t,S,E,I,R,β,σ,γ,μ,ν: μ*N-ν*S-β*S*I/N
    dEdt=lambda t,S,E,I,R,β,σ,γ,μ,ν: β*S*I/N-(ν+σ)*E
    dIdt=lambda t,S,E,I,R,β,σ,γ,μ,ν:     σ*E-(γ+ν)*I
    dRdt=lambda t,S,E,I,R,β,σ,γ,μ,ν:     γ*I-ν*R
    
    # Modelo de Ecuaciones Diferenciales SEIR (Hans Nesse, Vital D. + Vacc)
    
    '''
    dSdt=lambda t,S,E,I,R,β,σ,γ,μ,ν: μ*(N-S)-β*S*I/N-ν*S
    dEdt=lambda t,S,E,I,R,β,σ,γ,μ,ν:   β*S*I/N-(μ+σ)*E
    dIdt=lambda t,S,E,I,R,β,σ,γ,μ,ν:     σ*E-(μ+γ)*I
    dRdt=lambda t,S,E,I,R,β,σ,γ,μ,ν:     γ*I-μ*R+ν*S
    '''
    
    # Solucionar el Modelo Usando el Método Numérico Dormand-Prince
    
    # Solución=Método(Tiempo,Ecuaciones,Condiciones Iniciales, Parámetros)
    
    t,S,E,I,R=DP(T,dSdt,dEdt,dIdt,dRdt,S0,E0,I0,R0,β,σ,γ,μ,ν,n)
    
    # Concatenar Todo en una Matriz
    
    t=t.reshape(-1,1)
    S=S.reshape(-1,1)
    E=E.reshape(-1,1)
    I=I.reshape(-1,1)
    R=R.reshape(-1,1)
    
    Sol=np.hstack((t,S,E,I,R))
    
    return Sol 

########################################## Método de Dormand-Prince ####################################

#### dx/dt=f1(t,x,y,z,w), x(t0)=x0
#### dy/dt=f2(t,x,y,z,w), y(t0)=y0
#### dz/dt=f3(t,x,y,z,w), z(t0)=z0
#### dw/dt=f4(t,x,y,z,w), w(t0)=w0

### Nota: Es mejor tener una implementación vectorial a cambio de cambiar la notación (pierde contexto filosófico)

def DP(T,f1,f2,f3,f4,x0,y0,z0,w0,β,σ,γ,μ,ν,n,order=5):
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
        k1=f1(t[i]       ,x[i],y[i],z[i],w[i],β[i],σ[i],γ[i],μ[i],ν[i])
        k2=f1(t[i]+1/5*h ,x[i]+h*k1/5,y[i]+h*k1/5,z[i]+h*k1/5,w[i]+h*k1/5,β[i],σ[i],γ[i],μ[i],ν[i])
        k3=f1(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2),y[i]+h*(3/40*k1+9/40*k2),z[i]+h*(3/40*k1+9/40*k2),w[i]+h*(3/40*k1+9/40*k2),β[i],σ[i],γ[i],μ[i],ν[i])
        k4=f1(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3),y[i]+h*(44/45*k1-56/15*k2+32/9*k3),z[i]+h*(44/45*k1-56/15*k2+32/9*k3),w[i]+h*(44/45*k1-56/15*k2+32/9*k3),β[i],σ[i],γ[i],μ[i],ν[i])
        k5=f1(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),y[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),z[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),w[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),β[i],σ[i],γ[i],μ[i],ν[i])
        k6=f1(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),y[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),z[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),w[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),β[i],σ[i],γ[i],μ[i],ν[i])
        k7=f1(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),β[i],σ[i],γ[i],μ[i],ν[i])
        if order==5:
            x[i+1]=x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        else:
            x[i+1]=x[i]+h*(5179/57600*k1+7571/16695*k3+393/640*k4-92097/339200*k5+187/2100*k6+1/40*k7)
        
        k1=f2(t[i]       ,x[i],y[i],z[i],w[i],β[i],σ[i],γ[i],μ[i],ν[i])
        k2=f2(t[i]+1/5*h ,x[i]+h*k1/5,y[i]+h*k1/5,z[i]+h*k1/5,w[i]+h*k1/5,β[i],σ[i],γ[i],μ[i],ν[i])
        k3=f2(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2),y[i]+h*(3/40*k1+9/40*k2),z[i]+h*(3/40*k1+9/40*k2),w[i]+h*(3/40*k1+9/40*k2),β[i],σ[i],γ[i],μ[i],ν[i])
        k4=f2(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3),y[i]+h*(44/45*k1-56/15*k2+32/9*k3),z[i]+h*(44/45*k1-56/15*k2+32/9*k3),w[i]+h*(44/45*k1-56/15*k2+32/9*k3),β[i],σ[i],γ[i],μ[i],ν[i])
        k5=f2(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),y[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),z[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),w[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),β[i],σ[i],γ[i],μ[i],ν[i])
        k6=f2(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),y[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),z[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),w[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),β[i],σ[i],γ[i],μ[i],ν[i])
        k7=f2(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),β[i],σ[i],γ[i],μ[i],ν[i])
        if order==5:
            y[i+1]=y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        else:
            y[i+1]=y[i]+h*(5179/57600*k1+7571/16695*k3+393/640*k4-92097/339200*k5+187/2100*k6+1/40*k7)
        
        k1=f3(t[i]       ,x[i],y[i],z[i],w[i],β[i],σ[i],γ[i],μ[i],ν[i])
        k2=f3(t[i]+1/5*h ,x[i]+h*k1/5,y[i]+h*k1/5,z[i]+h*k1/5,w[i]+h*k1/5,β[i],σ[i],γ[i],μ[i],ν[i])
        k3=f3(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2),y[i]+h*(3/40*k1+9/40*k2),z[i]+h*(3/40*k1+9/40*k2),w[i]+h*(3/40*k1+9/40*k2),β[i],σ[i],γ[i],μ[i],ν[i])
        k4=f3(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3),y[i]+h*(44/45*k1-56/15*k2+32/9*k3),z[i]+h*(44/45*k1-56/15*k2+32/9*k3),w[i]+h*(44/45*k1-56/15*k2+32/9*k3),β[i],σ[i],γ[i],μ[i],ν[i])
        k5=f3(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),y[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),z[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),w[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),β[i],σ[i],γ[i],μ[i],ν[i])
        k6=f3(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),y[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),z[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),w[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),β[i],σ[i],γ[i],μ[i],ν[i])
        k7=f3(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),β[i],σ[i],γ[i],μ[i],ν[i])
        if order==5:
            z[i+1]=z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        else:
            z[i+1]=z[i]+h*(5179/57600*k1+7571/16695*k3+393/640*k4-92097/339200*k5+187/2100*k6+1/40*k7)
        
        k1=f4(t[i]       ,x[i],y[i],z[i],w[i],β[i],σ[i],γ[i],μ[i],ν[i])
        k2=f4(t[i]+1/5*h ,x[i]+h*k1/5,y[i]+h*k1/5,z[i]+h*k1/5,w[i]+h*k1/5,β[i],σ[i],γ[i],μ[i],ν[i])
        k3=f4(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2),y[i]+h*(3/40*k1+9/40*k2),z[i]+h*(3/40*k1+9/40*k2),w[i]+h*(3/40*k1+9/40*k2),β[i],σ[i],γ[i],μ[i],ν[i])
        k4=f4(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3),y[i]+h*(44/45*k1-56/15*k2+32/9*k3),z[i]+h*(44/45*k1-56/15*k2+32/9*k3),w[i]+h*(44/45*k1-56/15*k2+32/9*k3),β[i],σ[i],γ[i],μ[i],ν[i])
        k5=f4(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),y[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),z[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),w[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4),β[i],σ[i],γ[i],μ[i],ν[i])
        k6=f4(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),y[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),z[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),w[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5),β[i],σ[i],γ[i],μ[i],ν[i])
        k7=f4(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),y[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),z[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),w[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6),β[i],σ[i],γ[i],μ[i],ν[i])
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