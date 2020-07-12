import time, sys
import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d

class Detla_Sigma_gas():
    def __init__(self,rad,Sigma):
        
        self.Sigma = interp1d(rad,Sigma,fill_value="extrapolate")

    def integrand(self,x):
        res = self.Sigma(x) * 2. * np.pi * x
        return res
        
    def TotSigma(self,r):
        rarr = np.arange(0,r,0.001)
        res = np.trapz(self.integrand(rarr),x=rarr)
        #res = integrate.quad(self.integrand, 0., r, epsabs=0., epsrel=1.e-10)[0]
        return res
        
    def BarSigma(self,r):
        rarr = np.arange(0,r,0.001)
        res = np.trapz(self.integrand(rarr),x=rarr)
        #res = integrate.quad(self.integrand, 0., r, epsabs=0., epsrel=1.e-10)[0]
        res /= np.pi * r**2
        return res
    
    def Sig_mean(self,r):
        res = map(self.BarSigma, r)
        BarS = np.array(list(res))
        #BarS = r * 0.0
        #for i in range(len(r)):
        #    BarS[i] = self.BarSigma(r[i])
        return BarS
    
    def DelSigma(self,r):
        res = self.Sig_mean(r) - self.Sigma(r)
        return res

