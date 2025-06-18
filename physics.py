import numpy as np
from scipy.optimize import root
from scipy.interpolate import CubicSpline

class dimensionless_numbers:
    def Reynolds(self, D, u, nu):
        return u * D / nu


class physical_constants:
    def __init__(self):
        self.R = 8.314462618 # J * ml^-1 * K^-1
        self.F = 96485.3321 # A*s*mol^-1
        self.g = 9.8 # m*s^-2


class species(physical_constants):
    def __init__(self, input_dict):
        super().__init__()
        self.Mw = input_dict["Mw"]
        self.molecule = input_dict["molecule"]


class phase_model(dimensionless_numbers, physical_constants):
    def __init__(self, conditions):
        physical_constants.__init__(self)
        self.T = conditions["T"] # K
        self.p = conditions["p"]
        self.G = conditions["G"]
        self.liquid = conditions["liquid"]
        self.gas = conditions["gas"]
        self.eG = conditions["eG"]
        self.rho = conditions["rho"]
        self.mu = conditions["mu"]
        self.sigma = conditions["sigma"]
        
    
    def effective_property(self, G_gas, key):
        if key == "density":
            rho_G = self.gas_density()
            Q_G = G_gas / rho_G
            rho = self.G * self.liquid["rho"] / (self.G + self.liquid["rho"] * Q_G - G_gas)
            return rho
        
        elif key == "viscosity":
            e_G = self.gas_vol_frac(G_gas)
            mu = self.liquid["mu"] * (1 - e_G) + self.gas["mu"] * e_G
            return mu
        
        elif key == "conductivity":
            e_G = self.gas_vol_frac(G_gas)
            sigma = self.liquid["sigma"] * (1 - e_G)**1.5
            return sigma
    

    def gas_vol_frac(self, G_gas):
        rho = self.effective_property(G_gas, "density")
        rho_G = self.gas_density()
        e_G = (self.liquid["rho"] - rho) / (self.liquid["rho"] - rho_G)
        return e_G

    def gas_density(self):
        rho  = self.p / (self.R * self.T) * self.gas["Mw"]
        return rho
    

    def friction_factor(self, D, nu, u, eps, model):
        Re = self.Reynolds(D, u, nu)
        if model == "Colebrook":
            Re_min = min(Re*0.9)
            Re_max = max(Re*1.1)
            Rex = np.linspace(Re_min, Re_max, 6000)
            f_fx = lambda x: -2 * np.log(eps / (3.7 * D) + 2.51 / (Rex * x**0.5))
            fx_v = (1/f_fx(Rex))**2
            finterp = CubicSpline(Rex, fx_v)
            fx = finterp(Re)
            
            # f_fx = lambda x: 1 / (x **0.5) + 2 * np.log(eps / (3.7 * D) + 2.51 / (Re * x**0.5)) 
            # solution = root(f_fx, 0.01*np.ones(Re.shape), method="lm")
            # fx = solution.x
        elif model == "Churchill":
            f_Re = (7 / Re)**0.9 + 0.27 * eps / D
            theta_a = (-2.457 * np.log(f_Re))**16 
            theta_b = (37530 / Re)**16
            fx = 8 * ((8 / Re)**12 + 1 / ((theta_a + theta_b)**1.5))**(1/12)
        else:
            raise RuntimeError(f"Currently the only available models are: <Colebrook> <Churchill>")
        
        idx = np.where(Re < 2e3)
        fx[idx] = 64 / Re[idx] 
        return fx



        