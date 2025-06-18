import numpy as np
from physics import phase_model
from electroChem import circuit_model, electrochemical_process


class electrode_side(phase_model):
    def __init__(self, data):   
        phase_model.__init__(self, data["phase"])
        self.type = data["side"]
        self.As = data["As"]
        self.H = data["height"]
        self.L = data["length"]
        self.nf = data["internal_flow_channels"]
        self.xv = data["obstruction_factor"]

    def mass_flow_resistance_nf(self):
        nu = self.mu / self.rho
        lc = self.H
        hc = self.L
        nf = self.nf
        wc = lc / (2 * nf)
        dh = 2 * wc * hc / (wc + hc)
        return 32 * nu * lc / (nf * wc * hc * dh**2)
    
    def mass_flow_resistance(self):
        nu = self.mu / self.rho
        hc = self.H
        wc = self.L
        xv = self.xv
        dh = 2 * wc * hc / (wc + hc)
        return 32 * nu * hc / ((1 - xv) * wc * hc * dh**2)
    
    def pressure_drop(self):
        d_p = self.mass_flow_resistance_nf() * self.G
        return d_p
    

class cell(electrode_side, electrochemical_process):
    def __init__(self, data):
        electrochemical_process.__init__(self, data)
        self.anode_side = electrode_side(data["anode_side"])
        self.cathode_side = electrode_side(data["cathode_side"])
   

class channel(phase_model, circuit_model):
    def __init__(self, data):
        circuit_model.__init__(self, data["circuit"])
        phase_model.__init__(self, data["phase"])
        self.outlet = data["outlet"]
        self.type = data["side"]
        self.L = data["length"]
        self.D = data["diameter"]

    def mass_flow_resistance(self, eps, model):
        rho = self.rho
        mu = self.mu
        Ac = np.pi / 4 * self.D**2
        u = self.G / (rho * Ac)
        fx = self.friction_factor(self.D, mu / rho, u, eps, model)
        Wch = (fx / 2) * (self.L / self.D) * u / Ac
        return Wch

    def minor_resistance(self, fm):
        rho = self.rho
        Ac = np.pi / 4 * self.D**2
        u = self.G / (rho * Ac)
        Wm = (fm / 2) * (u / Ac)
        return Wm
    
    def pressure_drop(self, eps, model):
        W = self.mass_flow_resistance(eps, model)
        d_p = W * self.G
        return d_p
    
    def minor_pressure_drop(self, fm):
        Wm = self.minor_resistance(fm)
        d_p = Wm * self.G
        return d_p