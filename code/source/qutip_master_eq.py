import numpy as np
import qutip as qt
from scipy import interpolate

from functions_utils import * 

sx = qt.sigmax(); sy=qt.sigmay(); sz = qt.sigmaz(); s0 = qt.qeye(2); sp = qt.sigmap();sm = qt.sigmam()

hbar = 0.6582119514     # hbar in meV*ns
kb = 8.617333262145e-2  # Boltzmann constant in meV/K
ketupL = qt.tensor([qt.basis(2,0),qt.basis(2,0)])
ketupR = qt.tensor([qt.basis(2,0),qt.basis(2,1)])
ketdnL = qt.tensor([qt.basis(2,1),qt.basis(2,0)])
ketdnR = qt.tensor([qt.basis(2,1),qt.basis(2,1)])
s0sx = qt.tensor(sx,s0);
sxs0 = qt.tensor(sx,s0);
szs0 = qt.tensor(sz,s0);
s0sz = qt.tensor(s0,sz);
s0sy = qt.tensor(s0,sy);
szsz = qt.tensor(sz,sz);
sysy = qt.tensor(sy,sy);
sys0 = qt.tensor(sy,s0);
szsx = qt.tensor(sz,sx);
sxsz = qt.tensor(sx,sz);
sxsy = qt.tensor(sx,sy);
c_op = qt.tensor(sp,s0-sz)/2.;
up = qt.basis(2,0)
dn = qt.basis(2,1)
s0sm = qt.tensor(s0,sm);
s0sp = qt.tensor(s0,sp);
sms0 = qt.tensor(sm,s0);
sps0 = qt.tensor(sp,s0);
Q_op = qt.tensor(s0,s0+sz)/2.;

def acot(x):
	return np.pi / 2.0 - np.arctan(x)

def get_E0(v, t, dz, tun, Ez,deps):
    Eeup = 0.5*(Ez + np.sqrt(tun**2 + (v*t + dz/2+deps)**2))
    Egup = 0.5*(Ez - np.sqrt(tun**2 + (v*t + dz/2+deps)**2))
    Eedn = 0.5*(-Ez + np.sqrt(tun**2 + (v*t - dz/2+deps)**2))
    Egdn = 0.5*(-Ez - np.sqrt(tun**2 + (v*t - dz/2+deps)**2))
    return [Eeup,Egup,Eedn,Egdn]

def get_eigstates(v, t, dz, tun, Ez):
    ths = [acot((v*t+dz/2)/tun),acot((v*t-dz/2)/tun)]
    st = []
    st.append(np.cos(ths[0]/2)*ketupL+np.sin(ths[0]/2)*ketupR)
    st.append(np.sin(ths[0]/2)*ketupL-np.cos(ths[0]/2)*ketupR)
    st.append(np.cos(ths[1]/2)*ketdnL+np.sin(ths[1]/2)*ketdnR)
    st.append(np.sin(ths[1]/2)*ketdnL-np.cos(ths[1]/2)*ketdnR)
    return st

def get_theta(v,t,dz,tun,deps):
    return acot((v*t+dz/2+deps)/tun)

def get_vso(ax,ay,bx,theta2):
    return 1/4*bx*s0sx*(np.cos(theta2[0])-np.cos(theta2[1])
        )+ 1/4*bx*s0sz*(np.sin(theta2[0])-np.sin(theta2[1])
        )+ 1/2*ay*sxs0*np.sin((theta2[0]-theta2[1])/2
        )+ 1/2*ax*sxsy*np.cos((theta2[0]-theta2[1])/2
        )- 1/2*ax*sys0*np.sin((theta2[0]-theta2[1])/2
        )+ 1/2*ay*sysy*np.cos((theta2[0]-theta2[1])/2
        )+ 1/4*bx*szsx*np.cos((theta2[0]+theta2[1])/2
        )+ 1/4*bx*szsz*np.sin((theta2[0]+theta2[1])/2)

def get_vso_fast(ay,theta2):
     return 1/2*ay*sxs0*np.sin((theta2[0]-theta2[1])/2
          )+1/2*ay*sysy*np.cos((theta2[0]-theta2[1])/2)



def get_interpolated_spectrum(Es, mat, T):
    '''
        Interpolate the spectrum of the material at a given temperature

        Parameters
        ----------
        Es : array
            Array of energies
        mat : list[material, device]
            List of material and device parameters
        T : float
            Temperature of the system

        Returns
        -------
        tuple
            Interpolated spectra of S_x and S_z
    '''

    #For def of S_x and S_z see utils.py
    fx = interpolate.interp1d(Es, np.array([S_x(E, mat, T) for E in Es]))
    fz = interpolate.interp1d(Es, np.array([S_z(E, mat, T) for E in Es]))  
    return fx, fz


def run_master_qutip(mat, mn, data, trials, psi0 = (1,1), phase_correct = True):
    '''
        Run adiabatic master equation for a given set of parameters [data]

        Parameters
        ----------
        mat : list[material, device]
            List of material and device parameters
        mn : int
            Index of the material in the list
        data : dict 
            Dictionary of the parameters
        trials : int
            Number of trials to average over
        psi0 : tuple
            Initial SPIN state of the system [default]
        phase_correct : bool
            Correct the phase of the eigenvectors
        
        Returns
        -------
        res_mater: array[6]
            [0] Coherence in the ground state
            [1] Coherence in the excited state
            [2] Population in the excited dn state
            [3] Population in the excited up state
            [4] Population in the ground dn state
            [5] Population in the ground up state
            
    '''

    options = qt.Options(norm_tol=1e-14, rtol=1e-14, atol=1e-14) # increse accuracy of qutip
    T = data["T"]
    sign = mat[0]["sign"]
    Es = np.linspace(0.1,2*1e3,501) # energies to interpolate the spectrum on
    fx, fz = get_interpolated_spectrum(Es, mat, T)

    # initialization
    res_master = np.zeros((6), dtype=complex)
    Qup = 0
    Qdn = 0
    print(mn, data["v"]) #flag to check the progress of the simulation

    #LOOP OVER TRIALS
    for tr in range(trials):
        times = np.linspace((-data["eps0"])/data["v"],(data["eps0"])/data["v"],50001)
        
        #draw realisation of noise
        deps = np.random.normal(0,data["sige"])
        dtc = np.random.normal(0,data["sigt"])
        dzl =  np.random.normal(0,sign)
        dzr = np.random.normal(0,sign)
        tun = data["tun"] + dtc

        #get the energies and the eigenstates
        E0 = get_E0(data["v"],times,data["dz"]+(dzl-dzr),
                    tun,data["Ez"]+(dzl+dzr)/2.,deps)

        #get the orbital gap
        oms0 = np.sqrt((data["v"]*times+deps)**2+tun**2)

        #get the orbital angle
        theta = get_theta(data["v"],times,0,tun,deps)

        #get the adiabatic coupdling term
        dtheta =tun*data["v"]/oms0**2
        
        #Construct the Hamiltonian
        E0 = np.array(E0)
        H = qt.Qobj(np.diag(E0[:,0]),dims = [[2, 2], [2, 2]] 
                   )/hbar +  dtheta[0]/2*s0sy
        v = H.eigenstates()
        
        #If phase correct, set 
        if phase_correct:
              for k in range(4):
                v[1][k] = qt.Qobj(v[1][k]/complex(v[1][k][0])*np.abs(v[1][k][0]),
                              dims = [[2, 2], [1, 1]])
        
        #Construct the initial state
        psi = (psi0[0]*v[1][0]+psi0[1]*v[1][1])/np.sqrt(psi0[0]**2 + psi0[1]**2)
        
        #orbital relaxation rate gamma(t)
        gam_mn_z = np.sin(theta)**2*fz(oms0)/4/hbar**2
        gam_mn_x = np.cos(theta)**2*fx(oms0)/4/hbar**2
        
        #spin relaxation rate gamma_spin(t)
        spin_factor = np.abs(((data["ar"]-1j*data["aim"])*data["Ez"] + data["bx"]*tun)/(oms0**2-data["Ez"]**2))**2
        gam_mn_spin = spin_factor*(
            np.sin(theta)**2*fz(data["Ez"])/4/hbar**2+ np.cos(theta)**2*fx(data["Ez"])/4/hbar**2)
        
        #excitation reates
        gam_pl_z = gam_mn_z*np.exp(-oms0/kb/T)
        gam_pl_spin = gam_mn_spin*np.exp(-data["Ez"]/kb/T)
        gam_pl_x = gam_mn_x*np.exp(-oms0/kb/T)


        # run the master equation
        res = qt.mesolve([[qt.ket2dm(ketupL),E0[0,:]/hbar],
            [qt.ket2dm(ketupR),E0[1,:]/hbar],
            [qt.ket2dm(ketdnL),E0[2,:]/hbar],
            [qt.ket2dm(ketdnR),E0[3,:]/hbar],
            [s0sy,dtheta/2.]],
           psi,times,[[s0sp,np.sqrt(gam_pl_z+gam_pl_x)],
                       [s0sm,np.sqrt(gam_mn_z+gam_mn_x)],
                       [sms0, np.sqrt(gam_mn_spin)],
                       [sps0, np.sqrt(gam_pl_spin)]],
            options = options)


        #get the final eigenstates
        H = qt.Qobj(np.diag(E0[:,-1]),dims = [[2, 2], [2, 2]] 
            )/hbar +  dtheta[-1]/2*s0sy 
        v = H.eigenstates()
        if phase_correct:
            for k in range(4):
                v[1][k] = qt.Qobj(v[1][k]/complex(v[1][k][0])*np.abs(v[1][k][0]),
                              dims = [[2, 2], [1, 1]])
    
        #project the final state on the final eigenstates
        coh0 = v[1][1].dag()*res.states[-1]*v[1][0] #coherence in ground state
        cohe = v[1][3].dag()*res.states[-1]*v[1][2] #coherence in excited state
        Qdn = v[1][2].dag()*res.states[-1]*v[1][2]  #population in excited dn state
        Qup =  v[1][3].dag()*res.states[-1]*v[1][3] #population in excited up state
        Pdn = v[1][0].dag()*res.states[-1]*v[1][0]  #population in ground dn state
        Pup = v[1][1].dag()*res.states[-1]*v[1][1]  #population in ground up state
        
        #average over trials
        res_master[0] += coh0[0][0][0]/trials
        res_master[1] += cohe[0][0][0]/trials
        res_master[2] += Qdn[0][0][0]/trials
        res_master[3] += Qup[0][0][0]/trials
        res_master[4] += Pdn[0][0][0]/trials
        res_master[5] += Pup[0][0][0]/trials
        
    return res_master



