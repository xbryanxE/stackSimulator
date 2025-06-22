from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from simple_optimization import stack_opt_problem as sop
from single_simple_optimization import stack_opt_problem as ssop
from external_simple_optimization import stack_opt_problem as esop
import multiprocessing
import pandas as pd
import numpy as np

def map_optimization(problem, p_vals, H_vals):
    out_dict = {"Y": [], "cta_dp": [], "p": [], "H": []}
    for key, value in zip(problem.parameters.keys(), problem.parameters.values()):
        out_dict[key] = []
    for p_ in p_vals:
        for H_ in H_vals:
            problem.p_op = p_ # Pa
            problem.H = H_ # m
            algorithm = NSGA2(pop_size=100, sampling=LHS())
            res = minimize(problem, algorithm, termination=("n_gen", 300), seed=0, verbose=True, save_history=True) 
            print("elapsed time: ", res.exec_time)
            z = np.where(res.F[0,:] < 5)
            zz = np.where(res.F[:,0] == min(res.F[:,0]))
            F = res.F[zz]
            X = res.X[zz]
            for key, value in zip(out_dict.keys(), np.concatenate([F[0], [p_], [H_], X[0]])):
                out_dict[key].append(value)
    return out_dict

# internal manifolds with separate inlets
p_vals = np.array([5, 15, 30, 50]) * 1e5 # Pa
H_vals = np.array([0.7, 1., 1.3, 1.6]) # m
# parallelization scheme
n_process = 50
pool = multiprocessing.Pool(n_process)
runner = StarmapParallelization(pool.starmap)
problem = sop(elementwise_runner=runner)
out_dict = map_optimization(problem, p_vals, H_vals)
df = pd.DataFrame(out_dict)
df.to_csv("internal_optim_params/optimization_map.csv", index=False)
# single manifold with separate inlets
problem = ssop(elementwise_runner=runner)
out_dict = map_optimization(problem, p_vals, H_vals)
df = pd.DataFrame(out_dict)
df.to_csv("internal_optim_params_single/optimization_map.csv", index=False)
# external manifolds with separate inlets
p_vals = np.array([1.01325]) * 1e5 # Pa
problem = esop(elementwise_runner=runner)
out_dict = map_optimization(problem, p_vals, H_vals)
df = pd.DataFrame(out_dict)
df.to_csv("external_optim_params/optimization_map.csv", index=False)
print("Optimization completed and results saved to CSV files.")
