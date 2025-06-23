from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
import multiprocessing
from scipy.optimize import root
from pyDOE3 import *
import json
import pandas as pd
import numpy as np

from stackSimulator import *


class stack_opt_problem(ElementwiseProblem):
    def __init__(self, **kwargs):
        # stack parameters
        self.N = 350 # number of cells
        self.j_load = 4000 # A/m^2 
        # tau, Dm_in, Dch_in, Lch_in, Lch_out, Dch_L, Dch_k, Dm_L, Dm_k
        self.parameters = {"tau": [], "Dm_in": [], "Dch_in": [], "Lch_in": [],
                           "Dch_l": [], "Dch_k": [], "Dm_l": [], "Dm_K": []}
        # rho electrolyte
        self.p_op = 50 * 1e5 # Pa
        self.rho_w = 1250 # kg/m^3
        self.mu_w = 0.001 # Pa*s
        # max F
        self.max_F = 1e2
        self.min_F = 1e10 # initial value
        # cell parameters
        self.H = 1.5 # m
        self.thickness = 5e-3 # m
        # boundaries
        xl_ = np.array([80, 3e-2, 2e-3, 3e-3, 2e-3, 2e-3, 3e-2, 3e-2]) # tau, Dm_in, Dch_in, Lch_in, Dch_L, Dch_k, Dm_L, Dm_k 
        xu_ = np.array([400, 1.5e-1, 4e-3, 5e-2, 4e-3, 4e-3, 1.5e-1, 1.5e-1]) # tau, Dm_in, Dch_in, Lch_in, Dch_L, Dch_k, Dm_L, Dm_k
        super().__init__(n_var=8, n_obj=2, n_ieq_constr=5, xl=xl_, xu=xu_, **kwargs)
        # filename
        self.path = "templates/"
            
    def compute_shunt_current_batch(self, filename, x):
        try:
            self.set_system(filename, x)
            stack = stackProblem("templates/mod_"+filename, "Colebrook")
            stack.solve_problem()
            cta_dp_ = abs((stack.Cell.cathode_side.p - stack.Cell.anode_side.p).mean()) * 1e-2 # mbar
            shunt_rate_ = 100 * (1 - stack.stack_current_density() / stack.Conditions.j_load).mean() # %
        except:
            cta_dp_ = 1e8
            shunt_rate_ = self.max_F * 1e6
        return [shunt_rate_, cta_dp_] # shunt rate, cathodic pressure drop
    
        
    def _evaluate(self, x, out, *args, **kwargs):
        filename = "optim_three_manifolds.json"
        out["F"] = self.compute_shunt_current_batch(filename, x)
        # constraints
        G1 = out["F"][-1] - 5 # cta pressure difference lower than 5 mbar
        G2 = x[1] - x[-2] # inlet manifold diameter smaller than anodic outlet manifold diameter
        G3 = x[1] - x[-1] # inlet manifold diameter smaller than cathodic outlet manifold diameter
        G4 = x[2] - x[-4] # inlet channel diameter smaller than anodic outlet channel diameter
        G5 = x[2] - x[-3] # inlet channel diameter smaller than cathodic outlet channel diameter
        out["G"] = np.array([G1, G2, G3, G4, G5])
    
    def set_system(self, filename, x):
        n_cell = self.N # number of cells
        H = self.H # cell height
        width = self.thickness # cell width
        [tau, Dm_in, Dch_in, Lch, Dch_l, Dch_k, Dm_l, Dm_k] = x
        # set parameters
        Q_in = n_cell * width * np.pi / 4 * H**2 / tau # m^3/s
        As = np.pi / 4 * (H**2 - (Dm_in**2 + Dm_l**2 - Dm_k**2)) # m^2
        with open(self.path + filename, "r") as file:
            data = json.load(file)
        # set number of cells
        data["settings"]["N"]["value"] = int(n_cell)
        # set load 
        data["conditions"]["j_load"]["value"] = self.j_load
        # set flow rate
        data["conditions"]["Q_in"]["value"] = [2*float(Q_in)]
        # set pressure 
        data["conditions"]["p_out"]["value"] = float(self.p_op) # Pa
        # set cell parameters
        data["cell"]["height"]["value"] = float(H) # height
        data["cell"]["length"]["value"] = float(width) # width
        data["cell"]["As"]["value"] = float(As) # surface area
        # set inlet channels
        data["channel"][0]["diameter"]["value"] = float(Dch_in)
        data["channel"][0]["length"]["value"] = float(Lch)
        data["channel"][1]["diameter"]["value"] = float(Dch_in)
        data["channel"][1]["length"]["value"] = float(Lch)
        # set inlet manifolds
        data["manifold"][0]["diameter"]["value"] = float(Dm_in)
        # set outlet channels
        data["channel"][2]["diameter"]["value"] = float(Dch_l)
        data["channel"][2]["length"]["value"] = float(Lch)
        data["channel"][3]["diameter"]["value"] = float(Dch_k)
        data["channel"][3]["length"]["value"] = float(Lch)
        # set outlet manifolds
        data["manifold"][1]["diameter"]["value"] = float(Dm_l)
        data["manifold"][2]["diameter"]["value"] = float(Dm_k)
        # update json file
        with open("templates/mod_"+filename, "w") as file:
            json.dump(data, file, indent=4)
        return 0


if __name__=="__main__":
    n_proccess = 10
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)
    problem = stack_opt_problem(elementwise_runner=runner)
    algorithm = NSGA2(pop_size=100, sampling=LHS())
    res = minimize(problem, algorithm, termination=("n_gen", 100), seed=0, verbose=True, save_history=True)
    print("elapsed time: ", res.exec_time)
    # save results to excel file
    for key, value in zip(problem.parameters.keys(), res.X):
        problem.parameters[key] = [value]
    df = pd.DataFrame(problem.parameters)
    df.to_csv("internal_optim_params_single/params_I_5bar.csv", sep=",", index=False)
    # save hystorical
    n_evals = []             # corresponding number of function evaluations
    hist_F = []              # the objective space values in each generation
    for algo in res.history:
        # function evaluations
        n_evals.append(algo.evaluator.n_eval)
        
        # retrieve optimum
        opt = algo.opt
        hist_F.append(float(opt.get("F")[0,0]))

    history_df = pd.DataFrame({"n_evals": n_evals, "F": hist_F})
    history_df.to_csv("internal_optim_params_single/history_I_5bar.csv", sep=",", index=False)
