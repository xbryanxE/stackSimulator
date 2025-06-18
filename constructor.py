import json
import numpy as np
from stack_sections import *

def iter_dict(info_dict):
    out = {}
    for key, val in info_dict.items():
        if isinstance(val, dict):
            value = val.get("value") or val.get("name") or val.get("type")
        else:
            value = val
        if key != "name":
            out[key] = value
        else:
            out["molecule"] = value 
    return out

def iter_list_of_dict(dict_list):
    out = []
    for element in dict_list:
        out_ = iter_dict(element)
        out.append(out_)
    return out

def look_for(data_dict, out_dict, names, alias):
    n = len(names)
    for key, val in data_dict.items():
        # If key matches one of the names we're looking for
        if key in names:
            index = names.index(key)
            out_dict[alias[index]] = val["value"]
        # If the value is a dictionary, recurse into it
        elif isinstance(val, dict):
            look_for(val, out_dict, names, alias)
    return out_dict

def build_circuit_dict(N):
    out_dict = {
                "I": np.zeros(N),
                "R": np.zeros(N),
                "V": np.zeros(N)
                }
    return out_dict

def build_phase_dict(liquid_dict, gas_dict, conditions, N):
    phase = {
             "liquid": liquid_dict, 
             "gas": gas_dict,
             "T": conditions["T"] * np.ones(N),
             "p": conditions["p_out"] * np.ones(N),
             "G": conditions["Q_in"][0] * liquid_dict["rho"] * np.ones(N),
             "eG": np.zeros(N),
             "rho": liquid_dict["rho"] * np.ones(N),
             "mu": liquid_dict["mu"] * np.ones(N),
             "sigma": liquid_dict["sigma"] * np.ones(N)
             }
    return phase

def build_electrode_side(phase, cell_dict, side_type):
    out_dict = iter_dict(cell_dict)
    out_dict["phase"] = phase
    out_dict["side"] = side_type
    
    return out_dict

def build_channel(phase_list, circuit, channel_dict_list):
    out = []
    for element in channel_dict_list:
        out_dict = iter_dict(element)
        out_dict["phase"] = phase_list[0] if element["side"] == "anodic" else phase_list[-1]
        out_dict["circuit"] = circuit
        out.append(out_dict)
    return out

def build_manifold(phase_list, circuit, manifold_dict_list, L):
    out = []
    for element in manifold_dict_list:
        out_dict = iter_dict(element)
        out_dict["length"] = L
        out_dict["phase"] = phase_list[0] if element["side"] == "anodic" else phase_list[-1]
        out_dict["circuit"] = circuit
        out.append(out_dict)
    return out

def load_components(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    # build liquid properties dictionary
    info_dict = data.get("liquid", {})
    liquid = iter_dict(info_dict)

    # build gas properties dictionaries
    dict_list = data.get("gas", {})
    [gas_l, gas_k] = iter_list_of_dict(dict_list)

    # build settings dictionary
    info_settings = data.get("settings", {})
    settings = iter_dict(info_settings)

    # build error parameters dictionary
    error_dict = data.get("error_params", {})
    error_settings = iter_dict(error_dict)

    # build dictionary of operational conditions
    conditions_dict = data.get("conditions", {})
    conditions = iter_dict(conditions_dict)

    # build species dict
    species_list = []
    species_list.append({"Mw": gas_l["Mw"], "molecule": gas_l["molecule"]})
    species_list.append({"Mw": gas_k["Mw"], "molecule": gas_k["molecule"]})

    # build circuit dict
    circuit_dict = build_circuit_dict(settings["N"])

    # build phase dictionary
    phase_l = build_phase_dict(liquid, gas_l, conditions, settings["N"])
    phase_k = build_phase_dict(liquid, gas_k, conditions, settings["N"])

    # build cell
    cell_dict = data.get("cell", {})
    anode_side = build_electrode_side(phase_l, cell_dict, "anodic")
    cathode_side = build_electrode_side(phase_k, cell_dict, "cathodic")
    cell = {"anode_side": anode_side, "cathode_side": cathode_side}
    # build kinetics and equilibrium dictioanry
    names = ["Erev", "ne", "Rc", "Tafel_slope", "j_exchange"]
    alias = ["Erev", "ne", "Rc", "b", "j0"]
    electrochem_data = look_for(data, {}, names, alias)
    cell = {**cell, **electrochem_data}

    # build channels and manifolds
    channel_dict_list = data.get("channel", {})
    manifold_dict_list = data.get("manifold", {})
    L_man = cell_dict["length"]["value"] * 2
    channel_list = build_channel([phase_l, phase_k], circuit_dict, channel_dict_list)
    manifold_list = build_manifold([phase_l, phase_k], circuit_dict, manifold_dict_list, L_man)

    output = {"cell": cell, 
              "channels": channel_list,
              "manifolds": manifold_list,
              "settings": settings,
              "conditions": conditions,
              "error_params": error_settings
              }
    
    return output

# if __name__=="__main__":
#     output = load_components("templates/conditions_mod.json")
#     print("Finished")