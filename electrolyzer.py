import numpy as np
from scipy.interpolate import CubicSpline
from types import SimpleNamespace
from constructor import load_components
from electroChem import electrochemical_process, circuit_model
from physics import phase_model, physical_constants
from stack_sections import *
# import matplotlib.pyplot as plt
# from math_utilities import mse, iter_bounds, piccard_iteration, linear_constraint
# from time import time
# from transport_models.steady.scalar_transport import problem


class electrolyzer(electrochemical_process, circuit_model, phase_model, physical_constants):
    def __init__(self, filename):
        physical_constants.__init__(self)
        data = load_components(filename)
        self.Settings = SimpleNamespace(**data["settings"])
        self.Error_params = SimpleNamespace(**data["error_params"])
        self.Conditions = SimpleNamespace(**data["conditions"])
        self.Cell = cell(data["cell"])
        self.Channels = []
        self.Manifolds = []
        for i in range(self.Settings.n_manifolds):
            self.Channels.append(channel(data["channels"][i]))
            self.Manifolds.append(channel(data["manifolds"][i]))
        if self.Settings.n_manifolds < 4:
            self.Channels.append(channel(data["channels"][-1]))

    def update_flow_fields(self, G_out, Gx, sides, lp, model, verbose=False):
        for i in range(len(G_out)):
            G_in = (self.Conditions.Q_in[0] * self.Manifolds[0].liquid["rho"] if self.Settings.n_manifolds < 4 
                    else self.Conditions.Q_in[i] * self.Manifolds[0].liquid["rho"])
            # update mass flow rates
            self.update_mass_flow(G_in, G_out[i], Gx[i], sides[i], lp) 
            # calculate gas product
            Ic = self.stack_current_density() * self.Cell.anode_side.As
            ne = self.Cell.ne[0] if sides[i] == "anodic" else self.Cell.ne[-1]
            Mw = self.Cell.anode_side.gas["Mw"] if sides[i] == "anodic" else self.Cell.cathode_side.gas["Mw"]
            G_gas = self.gas_product(Ic, ne) * Mw # gas product
            # update properties
            self.update_properties(G_gas, sides[i])
            # update pressure
            self.update_pressure(model, sides[i])
            if verbose:
                print("Flow fields has been updated ... \n")

    
    def update_shunt_fields(self, I_man):
        n_man = self.Settings.n_manifolds
        for i in range(len(I_man)):
            self.Manifolds[i].I = I_man[i]
            if self.Manifolds[i].outlet and n_man < 4:
                inlet_channels = [obj for obj in self.Channels if getattr(obj, "outlet", None) == 0]
                Rd = self.get_paralle_resistance(inlet_channels)
                Ich_ = np.gradient(I_man[i], edge_order=2)
                for channel_ in inlet_channels:
                    channel_.I = Rd * Ich_ / self.get_resistance(channel_) # individual channel current
            else:
                # channels
                Rd = self.get_resistance(self.Channels[i])
                self.Channels[i].I = np.gradient(I_man[i], edge_order=2)



    def update_mass_flow(self, G_in, G_out, Gx, side, lp):
        # inlet manifold
        in_manifold = self.get_section("inlet-manifold", side)
        in_manifold.G = lp[-1] * G_in + lp[0] * (G_out + Gx) # inlet mass rate
        # inlet channel
        in_channel = self.get_section("inlet-channel", side)
        in_channel.G = -lp[0] * np.gradient(G_out, edge_order=2)
        # get half cell compartment
        half_cell = self.get_section("cell", side)
        half_cell.G = -lp[0] * np.gradient(G_out, edge_order=2)
        # get outlet channel
        out_channel = self.get_section("outlet-channel", side)
        out_channel.G = -lp[0] * np.gradient(G_out, edge_order=2)
        if out_channel.G[out_channel.G < 0].any():
            print("backflow... Not a feassible solution")
        # get outlet manifold
        out_manifold = self.get_section("outlet-manifold", side)
        out_manifold.G = G_out

    def update_properties(self, G_gas, side, verbose=False):
        outflow = self.Settings.outflow
        G_gas_man = np.cumsum(G_gas) if outflow == "forward" else np.flip(np.cumsum(np.flip(G_gas)))
        alias = ["rho", "mu", "sigma", "eG"]
        labels = ["density", "viscosity", "conductivity", "gas_vol_frac"]
        sections = ["inlet-manifold", "inlet-channel", "cell", "outlet-channel", "outlet-manifold"]
        gas_product = [0.*G_gas, 0*G_gas, G_gas, G_gas, G_gas_man]
        for label, name in zip(labels, alias):
            for section, gas_rate in zip(sections, gas_product):
                flow_obj = self.get_section(section, side)
                if section == "cell" and name != "eG":
                    val = (flow_obj.effective_property(gas_rate, label) + flow_obj.liquid[name]) / 2
                    setattr(flow_obj, name, val.copy())
                elif label == "gas_vol_frac":
                    flow_obj.eG = flow_obj.gas_vol_frac(gas_rate) / 2
                else:
                    val = flow_obj.effective_property(gas_rate, label)
                    setattr(flow_obj, name, val.copy())
        if verbose:
            print("properties updated ... \n")

    

    def update_pressure(self, model, side, verbose=False):
        px = self.Conditions.p_out # outlet pressure
        outflow = self.Settings.outflow # outflow direction
        eps = self.Settings.roughness_factor # rhoughness coefficient
        fm = self.Settings.minor_losses_factor # minor losses coefficient

        pathways = [outflow, "downward", "downward", "downward", "forward"]
        sections = ["outlet-manifold", "outlet-channel", "cell", "inlet-channel", "inlet-manifold"]
        for pw, section in zip(pathways, sections):
            flow_obj = self.get_section(section, side)
            p_vector = flow_obj.p.copy()
            M = self.Settings.N if pw == "forward" or pw == "backward" else 1
            if M == 1:
                if hasattr(flow_obj, "H"):
                    delta_p = flow_obj.pressure_drop()
                else:
                    delta_p = flow_obj.pressure_drop(eps, model)
                    delta_p += flow_obj.minor_pressure_drop(fm)
                p_vector = px + 0.5 * delta_p if pw == "downward" else px - 0.5 * delta_p
                flow_obj.p = p_vector.copy()
                px = px + delta_p if pw == "downward" else px - delta_p
            elif section == "outlet-manifold":
                delta_p = flow_obj.pressure_drop(eps, model)
                if pw == "backward":
                    for i in range(1, M):
                        f = (delta_p[i-1] + delta_p[i])
                        p_vector[i] = px + 0.5 * f 
                        px += delta_p[i-1] 
                elif pw == "forward":
                    for i in range(M-2, -1, -1):
                        f = (delta_p[i+1] + delta_p[i])
                        p_vector[i] = px + 0.5 * f
                        px += delta_p[i+1]
                px = p_vector.copy()
                flow_obj.p = p_vector.copy()
            elif section == "inlet-manifold":
                if pw == "forward":
                    p_vector[0] = float(px[0])
                    px = float(px[0])
                    delta_p = flow_obj.pressure_drop(eps, model)
                    for i in range(1, M):
                        f = (delta_p[i-1] + delta_p[i])
                        p_vector[i] = px - 0.5 * f
                        px -= delta_p[i-1]
                elif pw == "backward":
                    p_vector[-1] = float(px[-1])
                    px = float(px[-1])
                    for i in range(M-2, -1, -1):
                        f = (delta_p[i+1] + delta_p[i])
                        p_vector[i] = px + 0.5 * f
                        px += delta_p[i+1]
                flow_obj.p = p_vector.copy()
        # forward correction
        # inlet_manifold = self.get_section("inlet-manifold", side)
        # delta_p = inlet_manifold.pressure_drop(eps, model)
        # px = inlet_manifold.p[0]
        # M = len(inlet_manifold.p)
        # for i in range(1, M):
        #     f = (delta_p[i-1] + delta_p[i])
        #     inlet_manifold.p[i] = px - 0.5 * f
        #     px -= delta_p[i-1]
        if verbose:
            print("pressure updated ... \n")                



    def get_section(self, key, side):
        if key == "inlet-manifold":
            if self.Settings.n_manifolds > 3:
                out_obj = [obj for obj in self.Manifolds 
                            if getattr(obj, "type", None) == side 
                            and getattr(obj, "outlet", None) == 0][0]
            else:
                out_obj = [obj for obj in self.Manifolds 
                            if getattr(obj, "type", None) == "shared" 
                            and getattr(obj, "outlet", None) == 0][0]
        elif key == "inlet-channel":
            out_obj = [obj for obj in self.Channels 
                        if getattr(obj, "type", None) == side
                        and getattr(obj, "outlet", None) == 0][0]
        elif key == "cell":
            out_obj = self.Cell.anode_side if side == "anodic" else self.Cell.cathode_side
        elif key == "outlet-channel":
            out_obj = [obj for obj in self.Channels 
                        if getattr(obj, "type", None) == side 
                        and getattr(obj, "outlet", None) == 1][0]
        elif key == "outlet-manifold":
            out_obj = [obj for obj in self.Manifolds 
                        if getattr(obj, "type", None) == side
                        and getattr(obj, "outlet", None) == 1][0]
        return out_obj
    

    def get_resistance(self, obj):
        return obj.analogous_resistance(obj.L, obj.D, obj.sigma)
    
    def get_paralle_resistance(self, obj_list):
        R = []
        for obj in obj_list:
            R_ = self.get_resistance(obj)
            R.append(R_)
        return self.parallel_resistance(R)
    
    def upward_flow_resistance(self, obj_list, lv, model="Colebrook"):
        eps = self.Settings.roughness_factor
        fm = self.Settings.minor_losses_factor
        [in_channel, _, _] = obj_list
        WT = 0
        for element in obj_list:
            if hasattr(element, "nf"):
                WT += element.mass_flow_resistance_nf() if element.nf > 1 else element.mass_flow_resistance()
            else:
                WT += element.mass_flow_resistance(eps, model)
        WT += in_channel.minor_resistance(fm)
        return WT * lv
    
    def forward_flow_resistance(self, obj_list, lv, model="Colebrook"):
        eps = self.Settings.roughness_factor
        WMT = 0
        for element in obj_list:
            WMT += element.mass_flow_resistance(eps, model)
        return WMT * lv

    def total_shunt(self):
        Is = 0
        for i in range(self.Settings.n_manifolds):
            Is += self.Manifolds[i].I
        return Is

    def stack_current_density(self):
        j = (self.Conditions.j_load * self.Cell.anode_side.As - self.total_shunt()) / self.Cell.anode_side.As
        return j

    def stack_voltage(self, j):
        V = self.Cell.cell_voltage(j)
        return V

    def mapping_voltage(self, u):
        # create mapping function
        eta_max = np.log(0.71)
        eta_vals_l = np.logspace(-3, eta_max, 3000)
        eta_max = np.log(1.15)
        eta_vals_k = -np.logspace(eta_max, -3, 3000)
        eta = np.concatenate([eta_vals_k, eta_vals_l])
        jv = self.Cell.current_density(eta)
        f_interp = CubicSpline(jv, eta)
        As = self.Cell.anode_side.As
        Iw = self.Conditions.j_load * As
        j = self.stack_current_density()
        return f_interp(u) + self.Cell.Erev + j*self.Cell.Rc


if __name__=="__main__":
    stack = electrolyzer("templates/conditions_four_manifolds.json")
#     # stack.flow_solve("Colebrook", verbose=True)
#     stack.coupled_solution("Colebrook", verbose=True)
#     end = time()
#     LPH_l = stack.Cell.anode_side.G / stack.Cell.anode_side.rho * 1e3 * 3600
#     LPH_k = stack.Cell.cathode_side.G / stack.Cell.cathode_side.rho * 1e3 * 3600
#     plt.plot(stack.Manifolds[-1].I, label="anode side flow")
#     # plt.plot(stack.Cell.cathode_side.p*1e-5, label="cathode side flow")
#     plt.xlabel("Cells in stack")
#     plt.ylabel("y")
#     plt.legend()
#     plt.show()

#     elapsed = end - start
#     print("finished")
#     print(f"elapsed time: {elapsed:.4f} s")
         