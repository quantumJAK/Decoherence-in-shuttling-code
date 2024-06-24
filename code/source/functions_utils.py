import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from numpy import genfromtxt
import scipy.special as special
import scipy.integrate as integrate
plt.rc('text', usetex=True)
import seaborn as sns
from numpy import genfromtxt

def cartesian(arrays, out=None):

    arrays = [np.asarray(x) for x in arrays]

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)])

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
        
   
    return out

def cart_dict(out,names):
    dicts = []
    for o in out:
        dicts.append({names[k]:i for k,i in enumerate(o)})
    return dicts

def fgauss(kz,dev):
    return np.exp(-kz**2*dev["lz"]**2/4)

def coth(x):
    return np.cosh(x)/np.sinh(x)


def Sou(om,sig,tc):
    return 2*sig**2*tc/(1+om**2*tc**2/hbar**2) 

exp = np.exp
sqrt = np.sqrt
sin = np.sin
cos = np.cos
Erfi = special.erfi
PI = np.pi;
hbar = 0.6582119514;
kb = 86.17333262* 1e-3

SiMos = {"lxy": 20, "lz": 2.5, "dx": 50,"name":"SiMos"}
Malinowski = {"lxy": 40, "lz": 10, "dx": 150,"name":"Malinowski"}
Petta = {"lxy": 38, "lz": 2.5, "dx": 150,"name":"Petta"}
Struck = {"lxy": 20, "lz": 2.5, "dx": 100,"name":"Struck"}
Srinivasa = {"lxy": 15*np.sqrt(2), "lz": 15*np.sqrt(2), "dx": 110 ,"name":"Srinivasa"}

SiGe = {"rho": 2330*0.95*1e-2, "Xiu": 5*1e6, "Xid": 8.77*1e6, 
        "c": {"L":9.15 * 1e3 , "T":5 *1e3},"p":0,"name":"SiGe", "sign":np.sqrt(2)*hbar/20/1e3}
GaAs = {"rho": 5300*0.95*1e-2, "Xiu": 0, "Xid": 7 * 1e6, 
        "c": {"L":5.3 * 1e3 , "T":2.5 *1e3}, "p":1.4 * 1e6, "name":"GaAs", "sign":np.sqrt(2)*hbar/0.05/1e3}
GaAsN = {"rho": 5300*0.95*1e-2, "Xiu": 0, "Xid": 7 * 1e6, 
        "c": {"L":5.3 * 1e3 , "T":2.5 *1e3}, "p":1.4 * 1e6, "name":"GaAs", "sign":np.sqrt(2)*hbar/2/1e3}

# NOISE PARAMETERS
R = 50      #resistatnce of Jonhnson noise source
alf = 0.01  #amplitude of tunnel coupling versus detuning noise
T = 100     #temperature of the system in mK
f = fgauss  #model of the wavefunction (gaussian)
S1 = 1      #amplitude of overf noise mueV^2/Hz.


# COMPUTE PHONON SPECTRA
D = lambda dev: {"L":lambda th: dev["Xid"] + dev["Xiu"] * np.cos(th)**2,
                 "T":lambda th: -dev["Xiu"]*np.sin(th)*np.cos(th)}

    
def PiezX(f,pol,Lam, dev, mat):
    I = integrate.quad(lambda th: sin(th) *
                np.exp(-(Lam*dev["lxy"]/mat["c"][pol]/hbar)**2*sin(th)**2/2)*
                np.abs(f(Lam/mat["c"][pol]/hbar*cos(th),dev))**2*np.exp(-dev["dx"]**2/dev["lxy"]**2/2
                )*(
                    3-4*special.jv(0,dev["dx"]*Lam/mat["c"][pol]/2/hbar*np.sin(th)
                        )+special.jv(0,dev["dx"]*Lam/mat["c"][pol]/hbar*np.sin(th))),
                       0, np.pi )[0]
    return np.pi*I

def PiezP(f,pol,Lam, dev, mat):
    I = integrate.quad(lambda th: sin(th) *
                np.exp(-(Lam*dev["lxy"]/mat["c"][pol]/hbar)**2*sin(th)**2/2)*
                np.abs(f(Lam/mat["c"][pol]/hbar*cos(th), dev))**2
                       *(1-special.jv(0,dev["dx"]*Lam/mat["c"][pol]/hbar*np.sin(th))), 0, np.pi )[0]
    return np.pi*I
    

def Mx(f,pol,Lam, dev, mat):
    I = integrate.quad(lambda th: sin(th) *D(mat)[pol](th)**2 *
                       np.exp(-(Lam*dev["lxy"]/mat["c"][pol]/hbar)**2*sin(th)**2/2)*
                       np.abs(f(Lam/mat["c"][pol]/hbar*cos(th), dev))**2*
                       np.exp(-dev["dx"]**2/dev["lxy"]**2/2)*(
                        3-4*special.jv(0,dev["dx"]*Lam/mat["c"][pol]/2/hbar*np.sin(th)
                        )+special.jv(0,dev["dx"]*Lam/mat["c"][pol]/hbar*np.sin(th)))
                       , 0, np.pi )[0]
    return np.pi*I

def Mp(f,pol,Lam, dev, mat):
    I = integrate.quad(lambda th: sin(th) * D(mat)[pol](th)**2 
                       *np.exp(-(Lam*dev["lxy"]/mat["c"][pol]/hbar)**2*sin(th)**2/2)*
                        np.abs(f(Lam/mat["c"][pol]/hbar*cos(th),dev))**2*
                        (1-special.jv(0,dev["dx"]*Lam/mat["c"][pol]/hbar*np.sin(th))), 0, np.pi )[0]
    return np.pi*I

def S_ph_z(f, om, dev, mat, ps= ["L","T"],piezo=True):
    g = 0
    for p in ps:
        g+=1./mat["c"][p]**5 * Mp(f, p, om, dev, mat)
        if mat["p"]>0 and piezo:
            g+=1./mat["c"][p]**3*mat["p"]**2*hbar**2/om**2*PiezP(f,p,om, dev, mat)  
    g *= om**3/2./np.pi**2/hbar**2/mat["rho"]
    return g
        
def S_ph_x(f, om, dev, mat, ps= ["L","T"], piezo=True):
    g = 0
    for p in ps:
        g+=1./mat["c"][p]**5 * Mx(f,p,om, dev, mat)
        if mat["p"]>0 and piezo:
            g+=1/mat["c"][p]**3*mat["p"]**2*hbar**2/om**2*PiezX(f,p,om, dev, mat)
    g *= om**3/2/np.pi**2/hbar**2/mat["rho"]
    return g

#-----------------End of phonon spectra-----------------


def be(om,T):
    return 1/(-1+np.exp(om/kb/T))

def S_overf(om,S,T):
    return 2*hbar*np.pi*S**2/om * T/100

def S_john1(om,R,T):
    return R/1e3/13*om/(1-np.exp(-om/kb/T))*hbar



def S_x(e0, mat, T):
    f = fgauss
    return S_ph_x(f,e0,mat[1],mat[0])+S_overf(e0,alf*S1,T)+ S_john1(e0,alf*R,T)

def S_z(e0, mat, T):
    f = fgauss
    return S_ph_z(f,e0,mat[1],mat[0])+S_overf(e0,S1,T)+ S_john1(e0,R,T)

#Analytical expression for the mechanisms:
def om(tun,eps):
    return np.sqrt(tun**2 + eps**2)
    
def TCQ(v,dz,deps):
    return dz**2/v**2*deps**2/hbar**2

def wait(v,dz,gz,tun,T):
    print(v,dz,gz,tun,T)
    return SEAL(v,gz,tun,T)*dz**2/gz**2/hbar**2

def T2star(v,T2,eps0):
    return (2*eps0/v)**2/T2**2

def relax(v,gz,tun,tunp,Ez):
    return gz*tunp**2/4/v*tun/(Ez-tun)**2

def SEAL(v,gz,tun,T):
    return np.sqrt(2*np.pi)/v*np.sqrt(tun*kb*T)*gz*np.exp(-tun/kb/T)

def HEAL(vs,gx,gz,tun,eps0,T):
    print(vs,gx,gz,tun,eps0,T)
    def om(tun,eps):
        return np.sqrt(tun**2 + eps**2)
    chi = [integrate.quad(lambda t: (tun/om(tun,v*t))**2*gz + (v*t/om(tun,v*t))**2*gx,0,eps0/v)[0] for v in vs]
    return SEAL(vs,gz,tun,T)*np.exp(-np.array(chi))

def HEAL_f(vs,fx,fz,tun,eps0,T):
    def om(tun,eps):
        return np.sqrt(tun**2 + eps**2)
    chi = [integrate.quad(lambda t: (tun/om(tun,v*t))**2*fz(om(tun,v*t)
                    )/4/hbar**2 + (v*t/om(tun,v*t))**2*fx(om(tun,v*t))/4/hbar**2,
                          0,eps0/v)[0] for v in vs]
    gz = fz(tun)/4/hbar**2
    return SEAL(vs,gz,tun,T)*np.exp(-np.array(chi))

def LZ(v,tun):
    return np.exp(-tun**2/v/hbar*np.pi/2)

