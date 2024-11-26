# This code is written by Chandan and Surjit. It uses the three band TB Hamiltonian for MoS2 monolayer from the 
# paper Phys. Rev. B 88, 085433 (2013). We uses the all tb parameters from the paper.

from numpy import *
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.sparse as sps
from numpy import linalg as LA

def kpath(a):
    N= 100
    kx1=linspace(0,(4/(3*a))*pi,N)
    kx2=linspace((4/(3*a))*pi,pi/a,N)
    kx3=linspace(pi/a,0,N)
    kx = concatenate((kx1,kx2,kx3))
    ky1=linspace(0,0,N)
    ky2=linspace(0,pi/(sqrt(3)*a),N)
    ky3=linspace(pi/(sqrt(3)*a),0,N)
    ky = concatenate((ky1,ky2,ky3))
    return kx,ky

def tb_model_tnn(kx,ky,a,E1,E2,t0,t1,t2,t11,t12,t22,r0,r1,r2,r11,r12,u0,u1,u2,u11,u12,u22,lam):
    
    alpha = 0.5*kx*a
    beta = (sqrt(3)/2)*ky*a    
    
    Ham = zeros([3,3],dtype='complex')
    Re = zeros([3,3],dtype='complex')
    Im = zeros([3,3],dtype='complex')
    
   # for spin-orbit coupling
    Ham_soc = zeros([6,6],dtype='complex')
    sz = array([[1,0],[0,-1]])
    Imat = array([[1,0],[0,1]])
    Lz = array([[0,0,0],[0,0,2j],[0,-2j,0]])
    Ham_LS = (lam/2)*kron(sz,Lz)
    
    # Real part of Hamiltonian
    
    Re[0,0] = E1 + 2*t0*(2*cos(alpha)*cos(beta) + cos(2*alpha)) + 2*r0*(2*cos(3*alpha)*cos(beta) + cos(2*beta)) + 2*u0*(2*cos(2*alpha)*cos(2*beta) + cos(4*alpha))
    Re[1,1] = E2 + (t11+3*t22)*cos(alpha)*cos(beta) + 2*t11*cos(2*alpha) + 4*r11*cos(3*alpha)*cos(beta) + 2*(r11+sqrt(3)*r12)*cos(2*beta) + (u11+3*u22)*cos(2*alpha)*cos(2*beta) + 2*u11*cos(4*alpha)
    Re[2,2] = E2 + (3*t11+t22)*cos(alpha)*cos(beta) + 2*t22*cos(2*alpha) + 2*r11*(2*cos(3*alpha)*cos(beta)+cos(2*beta)) + (2/sqrt(3))*r12*(4*cos(3*alpha)*cos(beta)-cos(2*beta)) + (3*u11+u22)*cos(2*alpha)*cos(2*beta) + 2*u22*cos(4*alpha)
    
    Re[0,1] = -2*sqrt(3)*t2*sin(alpha)*sin(beta) + 2*(r1+r2)*sin(3*alpha)*sin(beta) - 2*sqrt(3)*u2*sin(2*alpha)*sin(2*beta)
    Re[0,2] = 2*t2*(cos(2*alpha)-cos(alpha)*cos(beta)) - (2/sqrt(3))*(r1+r2)*(cos(3*alpha)*cos(beta)-cos(2*beta)) +2*u2*(cos(4*alpha)-cos(2*alpha)*cos(2*beta))
    Re[1,2] = sqrt(3)*(t22-t11)*sin(alpha)*sin(beta) + 4*r12*sin(3*alpha)*sin(beta) + sqrt(3)*(u22-u11)*sin(2*alpha)*sin(2*beta)
    
    
    # Imaginary part of Hamiltonian
    Im[0,1] = 2*t1*sin(alpha)*(2*cos(alpha) + cos(beta)) + 2*(r1-r2)*sin(3*alpha)*cos(beta) + 2*u1*sin(2*alpha)*(2*cos(2*alpha) + cos(2*beta))
    Im[0,2] = (2*sqrt(3)*t1*cos(alpha)*sin(beta)) + (2/sqrt(3))*sin(beta)*(r1-r2)*(cos(3*alpha)+2*cos(beta)) + 2*sqrt(3)*u1*cos(2*alpha)*sin(2*beta)
    Im[1,2] = 4*t12*sin(alpha)*(cos(alpha)-cos(beta)) + 4*u12*sin(2*alpha)*(cos(2*alpha)-cos(2*beta))

#Hamiltonian
    Ham[0,0] = Re[0,0]
    Ham[1,1] = Re[1,1]
    Ham[2,2] = Re[2,2]
        
    Ham[0,1] = Re[0,1] + Im[0,1]*1j
    Ham[0,2] = Re[0,2] + Im[0,2]*1j
        
    Ham[1,0] = Ham[0,1].T.conj()
    Ham[1,2] = Re[1,2] + Im[1,2]*1j
        
    Ham[2,0] = Ham[0,2].T.conj()
    Ham[2,1] = Ham[1,2].T.conj()
    
    # for spin-orbit coupling
    Hamnew = kron(Imat,Ham)
    Ham_soc = Hamnew + Ham_LS
    
    return Ham_soc

t0 = -0.146; t1 = -0.114; t2 = 0.506; t11 =  0.085; t12 = 0.162 ; t22 =  0.073;
r0 = 0.06;  r1 = -0.236; r2 = 0.067; r11 = 0.016; r12 = 0.087; u0 = -0.038;
u1 = 0.046; u2 = 0.001; u11 = 0.266; u12=-0.176; u22 = -0.150;E1 = 0.683; E2 =  1.707;
lam = 0.073

a = 3.19

nk = 301

klt = 2

ts = linspace(-pi*klt,pi*klt,nk)

kx0,ky0 =kpath(3.19)


nk = len(kx0)

ssp = zeros([6,nk])

kk = []

for i in arange(nk):
    
    kx = kx0[i] ; ky = ky0[i]
    
    k = sqrt(kx0[i]**2+ky0[i]**2)
    kk.append(k)
    
    kp = cumsum(kk)   #cummalative sum
    
    Hamil = tb_model_tnn(kx,ky,a,E1,E2,t0,t1,t2,t11,t12,t22,r0,r1,r2,r11,r12,u0,u1,u2,u11,u12,u22,lam)
    evals, evecs = LA.eigh(Hamil)
    evals = sorted(evals)
    ssp[:,i] = evals

X0 = kp[0]/kp[299]
X1 = kp[99]/kp[299]
X2 = kp[199]/kp[299]
X3 = kp[299]/kp[299]


fig, ax = plt.subplots()

k_node = array([X0,X1,X2,X3])
for n in range (len(k_node)):
    ax.axvline(x=k_node[n], linewidth=0.5, color='k')



# Y-axis label
ax.set_ylabel("$E-E_F$ (eV)", fontsize=20)
ax.set_xlim(k_node[0],k_node[-1])

# X-axis values
ax.set_xticks([X0, X1, X2, X3])

# X-axis label for the above values
ax.set_xticklabels(['$\Gamma$', 'K', 'M', '$\Gamma$'], fontsize=20)
#ax.set_yticklabels(fontsize=20)

#plotting
plt.plot(kp/kp[299], ssp[0,:], 'blue', label='TB-model', linewidth=1.5)
plt.plot(kp/kp[299], ssp[1,:], 'blue', linewidth=1.5)
plt.plot(kp/kp[299], ssp[2,:], 'blue', linewidth=1.5)
    
#plt.xlabel(r'$k$')
#plt.ylabel(r'$E$')
plt.ylim(-2.0,2.0)
plt.yticks(fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()
# make PDF
fig.tight_layout()
fig.savefig("band-soc-MoS2.pdf")
print('Done.\n')
