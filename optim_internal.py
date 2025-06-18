from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import root
from pyDOE3 import *
from itertools import product
import json
import pandas as pd
import numpy as np

from stackSimulator import *


class stack_opt_problem(ElementwiseProblem):
    def __init__(self, **kwargs):
        # alpha_tau, alpha_w, alpha_m, alpha_ch, alpha_L, c_ml, c_mk, c_chl, c_chk, c_L, Ref
        self.parameters = {"alpha_tau": [], "alpha_w": [], "alpha_m": [], "alpha_ch": [], "alpha_L": [],
                           "c_ml": [], "c_mk": [], "c_chl": [], "c_chk": [], "Ref": []}
        # rho electrolyte
        self.p_op = 15 * 1e5 # Pa
        self.rho_w = 1250 # kg/m^3
        self.mu_w = 0.001 # Pa*s
        # limit lengths
        self.w_lims = [2.5e-3, 2e-2] # m
        self.Dman_lims = [1e-2, 0.15] # m
        self.Dch_lims = [2e-3, 1.5e-2] # m
        self.Lch_lims = [4e-3, 0.15] # m
        self.tau_lims = [80, 350] # s
        # max F
        self.max_F = 1e2
        self.min_F = 1e10 # initial value
        # boundaries
        xl_ = np.array([-1.5, -1.5, -1.5, -1.5, -1.5, 1., 1., 1., 1., 0.1]) # alpha_tau, alpha_w, alpha_m, alpha_ch, alpha_L, c_ml, c_mk, c_chl, c_chk, c_L, Ref   
        xu_ = np.array([0., 0., 0., 0., 0., 1.5, 1.5, 1.5, 1.5, 4000]) # alpha_tau, alpha_w, alpha_m, alpha_ch, alpha_L, c_ml, c_mk, c_chl, c_chk, c_L, Ref
        super().__init__(n_var=10, n_obj=1, xl=xl_, xu=xu_, **kwargs)
        # filename
        self.path = "templates/"
        # evaluation points
        self.v_max = np.array([1.2, 350]) # j_load, H, n_cell
        self.v_min = np.array([0.5, 50]) # j_load, H, n_cell
        self.eval_points = self._optimization_design()
    
    def min_max_norm(self, x, mode):
        if mode == "inverse":
            x_ = (x + 1) * (self.v_max - self.v_min) / 2 + self.v_min
        else:
            x_ = 2 * (x - self.v_min) / (self.v_max - self.v_min) - 1
        return x_
        
    def _optimization_design(self):
        normalized_design = ccdesign(2, center=(0,1), alpha="r", face="cci")
        return self.min_max_norm(normalized_design, "inverse")
            
    def compute_shunt_current_batch(self, idx, filename, x, max_F):
        try:
            _ = self.set_system(filename, x, self.eval_points[idx])
            stack = stackProblem("templates/mod_"+filename, "Colebrook")
            stack.solve_problem()
            cta_dp_ = abs((stack.Cell.cathode_side.p - stack.Cell.anode_side.p).mean()) * 1e-2 # mbar
            if cta_dp_ > 100:
                shunt_rate_ = max_F*4 + cta_dp_
            elif cta_dp_ > 5:
                shunt_rate_ = max_F*2 + cta_dp_
            else:
                shunt_rate_ = 100 * (1 - stack.stack_current_density() / stack.Conditions.j_load).mean()
        except:
            shunt_rate_ = self.max_F * 1e6
        return shunt_rate_
    
        
    def _evaluate(self, x, out, *args, **kwargs):
        filename = "optim_four_manifolds.json"
        shunt_rate = []
        cta_dp = []
        status = 1
        n_success = 1
        for i in range(len(self.eval_points)):
            status = self.set_system(filename, x, self.eval_points[i])
            if status == 0: break 
            else: n_success += 1 
        if status == 1:
            with ProcessPoolExecutor(max_workers=40) as executor:
                futures = [
                    executor.submit(self.compute_shunt_current_batch, idx, filename, x, self.max_F)
                    for idx in range(len(self.eval_points))
                ]
            shunt_rate = [future.result() for future in as_completed(futures)]
            # objective function
            shunt_rate = np.array(shunt_rate)
            out["F"] = shunt_rate.mean()
        else:
            out["F"] = self.max_F * 1e6 / n_success # set a very high value if constraints are not satisfied
        # show update
        if out["F"] < self.min_F: 
            self.min_F = out["F"]
            print("shunt rate: ", out["F"])
    
    def starting_parameters(self, Y, x, eval_point):
        H, n_cell = eval_point # (_, m, -)
        nu = self.mu_w / self.rho_w # m^2/s
        [tau, w, Re] = Y # (s, m)
        [at, aw] = x[0:2]
        Rex = x[-1]
        Dh = 2 * w * H / (w + H) # m
        V = np.pi / 4 * w * H**2 # m^3
        Ac = w * H # m^2
        Re_ = (V * Dh) / (Ac * tau * nu) # Reynolds number
        tau_ = (self.tau_lims[-1] - self.tau_lims[0]) / (1 + np.exp(n_cell**at * (Re - Rex))) + self.tau_lims[0] # s 
        w_ = (self.w_lims[-1] - self.w_lims[0]) / (1 + np.exp(n_cell**aw * (Re - Rex))) + self.w_lims[0] # m
        Y_ = np.array([tau_, w_, Re_]) # s, m
        return (Y - Y_)**2        
        
    def system_parameters(self, x, eval_point):
        # alpha_tau, alpha_w, alpha_m, alpha_ch, alpha_L, c_ml, c_mk, c_chl, c_chk, c_L, Ref
        H, n_cell = eval_point # (_, m, -)
        [am, ach, aL, cml, cmk, cchl, cchk, Rex] = x[2:10]
        Yi = [80, 2.5e-3, 100] # s, m, -
        res = root(self.starting_parameters, Yi, args=(x, eval_point), method="lm")
        tau, w, Re = res.x # residence time and half cell's width
        Dm = (self.Dman_lims[-1] - self.Dman_lims[0]) / (1 + np.exp(n_cell**am * (Re - Rex))) + self.Dman_lims[0] # inlet manifold diameter
        Dch = (self.Dch_lims[-1] - self.Dch_lims[0]) / (1 + np.exp(n_cell**ach * (Re - Rex))) + self.Dch_lims[0] # inlet channel diameter
        Lch = (self.Lch_lims[-1] - self.Lch_lims[0]) / (1 + np.exp(n_cell**aL * (Re - Rex))) + self.Lch_lims[0] # channel length 
        Dml = cml * Dm # anodic outlet manifold diameter
        Dmk = cmk * Dm # cathodic outlet manifold diameter
        Dchl = cchl * Dch # anodic outlet channel diameter
        Dchk = cchk * Dch # cathodic outlet channel diameter
        xdim = [Dch, Lch, Dchl, Dchk, Lch, Dm, Dml, Dmk, w]   
        return [xdim, tau, res.success]
    
    def set_system(self, filename, x, eval_point):
        H, n_cell = eval_point
        # set flow compartment width
        xdim, tau, convergence = self.system_parameters(x, eval_point)
        # status = self.check_constraints(xdim, H, convergence)
        status = 1
        if status:
            Q_in = n_cell * xdim[-1] * np.pi / 4 * H**2 / tau # m^3/s
            As = np.pi / 4 * H**2 # m^2
            with open(self.path + filename, "r") as file:
                data = json.load(file)
            # set number of cells
            data["settings"]["N"]["value"] = int(n_cell)
            # set load 
            data["conditions"]["j_load"]["value"] = 4000 # A/m^2
            # set flow rate
            data["conditions"]["Q_in"]["value"] = [float(Q_in)/2, float(Q_in)/2]
            # set pressure 
            data["conditions"]["p_out"]["value"] = float(self.p_op) # Pa
            # set cell parameters
            data["cell"]["height"]["value"] = float(H) # height
            data["cell"]["length"]["value"] = float(xdim[-1]) # width
            data["cell"]["As"]["value"] = float(As) # surface area
            # set inlet channels
            data["channel"][0]["diameter"]["value"] = float(xdim[0])
            data["channel"][0]["length"]["value"] = float(xdim[1])
            data["channel"][1]["diameter"]["value"] = float(xdim[0])
            data["channel"][1]["length"]["value"] = float(xdim[1])
            # set inlet manifolds
            data["manifold"][0]["diameter"]["value"] = float(xdim[5])
            data["manifold"][1]["diameter"]["value"] = float(xdim[5])
            # set outlet channels
            data["channel"][2]["diameter"]["value"] = float(xdim[2])
            data["channel"][2]["length"]["value"] = float(xdim[4])
            data["channel"][3]["diameter"]["value"] = float(xdim[3])
            data["channel"][3]["length"]["value"] = float(xdim[4])
            # set outlet manifolds
            data["manifold"][2]["diameter"]["value"] = float(xdim[6])
            data["manifold"][3]["diameter"]["value"] = float(xdim[7])
            # update json file
            with open("templates/mod_"+filename, "w") as file:
                json.dump(data, file, indent=4)
        return status
    
    def check_constraints(self, xdim, H, convergence=True):
        status = 1
        if any([xdim[0] > xdim[1], # is inlet channel diameter greater than the length?
                xdim[0] < 1.5e-3, # is inlet channel diameter greater than 1.5mm?
                xdim[0] > xdim[3], # is inlet channel diameter greater than outlet channel diameter? (cathodic)
                xdim[0] > xdim[8], # is inlet channel diameter greater than the flow compartments width?
                xdim[2] > xdim[3], # is anodic channel diameter greater than the cathodic?
                xdim[0] > xdim[5], # is diameter inlet channel greater than the inlet manifolds diameter?
                xdim[5] > xdim[7], # is inlet manifolds diameter greater than the outlet diameter? (cathodic)
                xdim[8] > xdim[5], # is the flow compartment width greater than the inlet manifold diameter?
                convergence == False, # did the cell width converge? 
                any(np.array(xdim) < 0), # check for negative values   
                any(np.isnan(np.array(xdim))) # check for nan      
                ]):
            status = 0
        return status


if __name__=="__main__":
    problem = stack_opt_problem()
    algorithm = PSO(pop_size=200, sampling=LHS())
    res = minimize(problem, algorithm, termination=("n_gen", 300), seed=0, verbose=True)
    print("elapsed time: ", res.exec_time)
    # save results to excel file
    for key, value in zip(problem.parameters.keys(), res.X):
        problem.parameters[key].append(value)
    df = pd.DataFrame(problem.parameters)
    df.to_csv("internal_optim_params/params_I_15bar.csv", sep=",", index=False)
    # save hystorical
    history_df = pd.DataFrame(res.history)
    history_df.to_csv("internal_optim_params/history_I_15bar.csv", sep=",", index=False)
