import numpy as np
from physics import physical_constants
from scipy.optimize import root


class kinetic_model:
    def __init__(self, j0, b):
        """
        Parameters:
        ba: Anodic Tafel slope (V/decade).
        bc: Cathodic Tafel slope (V/decade).
        j0: Exchange current density (A/cm^2).
        """
        self.b = b
        self.nb = len(b)
        self.j0 = j0

    def current_density(self, eta):
        if self.nb == 2:
            # calculate the current density given an overpotential
            j = self.j0 * (np.exp(eta / self.b[0]) - np.exp(-eta / self.b[1]))
        elif self.nb == 1:
            j = self.j0 * np.exp(eta / self.b[0])
        else:
            raise RuntimeError("Inconsistent number of kinetic parameters!!")
        return j
    
    def overpotential(self, j):
        if self.nb == 2:
            # Calculate the overpotential given a current density
            func = lambda eta: (j - self.current_density(eta))**2 # objective function
            sz = len(j.T)
            eta_initial_guess = np.zeros(sz) + 0.0001*np.sign(j)
            eta_solution = root(func, eta_initial_guess, method='lm')
        elif self.nb == 1:
            eta_solution = self.b[0] * np.log(j / self.j0)
        else:
            raise RuntimeError("Inconsistent number of kinetic parameters!!")
        return eta_solution.x


class electrochemical_process(kinetic_model, physical_constants):
    def __init__(self, conditions):
        physical_constants.__init__(self)
        self.ne = conditions["ne"]
        self.Erev = conditions["Erev"]
        self.Rc = conditions["Rc"]
        b = np.array(conditions["b"])
        j0 = conditions["j0"]
        kinetic_model.__init__(self, j0, b)

    def cell_voltage(self, j):
        eta = self.overpotential(j)
        return self.Erev + eta + self.Rc * j
    
    def gas_product(self, I, ne):
        Nr = np.abs(I) / (ne * self.F) # mol / s
        return Nr


class circuit_model(physical_constants):
    def __init__(self, conditions):
        physical_constants.__init__(self)
        self.R = conditions["R"]
        self.I = conditions["I"]
        self.V = conditions["V"]
        
    def analogous_resistance(self, L, D, sigma_eff):
        As = np.pi / 4 * D**2
        R = L / (sigma_eff * As)
        return R
    
    def parallel_resistance(self, R_list):
        f = 0
        for Rv in R_list:
            f += 1 / Rv
        return 1 / f
    
    def ohmic_voltage(self, I, R):
        return I * R 
