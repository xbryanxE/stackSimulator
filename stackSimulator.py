import dolfinx.fem.petsc
from electrolyzer import electrolyzer
from mpi4py import MPI
import dolfinx
import basix.ufl
import ufl
import numpy as np
import pyvista
import os, sys
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from scipy.interpolate import CubicSpline
from petsc4py import PETSc
import matplotlib.pyplot as plt


class stackProblem(electrolyzer):
    def __init__(self, filename, friction_model):
        super().__init__(filename)
        self.mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, self.Settings.N-1, [0, self.Settings.N-1])
        self.friction_model = friction_model
        self._set_mixed_space()
        self._boundary_init()
        self._init_problem()

    def _set_mixed_space(self):
        n = self.Settings.n_manifolds
        W = [basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1) for _ in range(n)] # shunt space
        mixed_shunt = basix.ufl.mixed_element(W)
        self.W = dolfinx.fem.functionspace(self.mesh, mixed_shunt) # shunt current space
        V = [basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1) for _ in range(2)] # flow rate space
        mixed_flow = basix.ufl.mixed_element(V)
        self.V = dolfinx.fem.functionspace(self.mesh, mixed_flow) # mass rates space

    # initialize fields
    def _init_problem(self):
        # current
        for i in range(self.Settings.n_manifolds):
            self.Manifolds[i].I = np.zeros(self.Settings.N)
        I_man = np.array([element.I for element in self.Manifolds])
        self.update_shunt_fields(I_man)
        # flow
        G_in = np.array(self.Conditions.Q_in) * self.Manifolds[0].liquid["rho"]
        outlet_manifold_l = self.get_section("outlet-manifold", side="anodic")
        outlet_manifold_k = self.get_section("outlet-manifold", side="cathodic")
        outlet_manifolds = [outlet_manifold_l, outlet_manifold_k]
        for i in range(2):
            if self.Settings.n_manifolds < 4:
                bound = ([0.5 * G_in[0] / self.Settings.N, 0.5 * G_in[0] * (1 - 1/self.Settings.N)] if self.Settings.outflow == "forward"
                         else [0.5 * G_in[0] * (1 - 1/self.Settings.N), 0.5 * G_in[0] / self.Settings.N])
            else:
                bound = ([G_in[i] / self.Settings.N, G_in[i] * (1 - 1/self.Settings.N)] if self.Settings.outflow == "forward"
                         else [G_in[i] * (1 - 1/self.Settings.N), G_in[i] / self.Settings.N])
            
            outlet_manifolds[i].G = np.linspace(bound[0], bound[-1], self.Settings.N)
        
        other_outlet_manifolds = [outlet_manifolds[-1], outlet_manifolds[0]]     
        sides = ["anodic", "cathodic"]
        lp = [-1, 1] if self.Settings.outflow == "forward" else [1, 0]
        G_init = np.array([element.G for element in outlet_manifolds])
        Gx = np.array([element.G for element in other_outlet_manifolds])
        Gx = Gx * 0 if self.Settings.n_manifolds > 3 else Gx
        self.update_flow_fields(G_init, Gx, sides, lp, self.friction_model)
        

    # boundary locators
    def left_boundary(self, x):
        return np.isclose(x[0], 0)
    
    def right_boundary(self, x):
        return np.isclose(x[0], self.Settings.N-1)

    def _apply_bound(self, bound, field):
        fdim = self.mesh.topology.dim - 1
        bc = []
        if field == "shunt":
            for i in range(self.Settings.n_manifolds):
                W0 = self.W.sub(i)
                W, V_to_W0 = W0.collapse()
                uD_left = dolfinx.fem.Function(W)
                uD_left.x.array[:] = bound[i][0]
                uD_right = dolfinx.fem.Function(W)
                uD_right.x.array[:] = bound[i][-1]
                dofs_left = dolfinx.fem.locate_dofs_topological((W0, W), fdim, self.left_facet)
                dofs_right = dolfinx.fem.locate_dofs_topological((W0, W), fdim, self.right_facet)
                bc_left = dolfinx.fem.dirichletbc(uD_left, dofs_left, W0)
                bc_right = dolfinx.fem.dirichletbc(uD_right, dofs_right, W0)
                bc.append(bc_left)
                bc.append(bc_right)
        elif field == "flow":
            for i in range(2):
                V0 = self.V.sub(i)
                V, V_to_V0 = V0.collapse()
                uD_left = dolfinx.fem.Function(V)
                uD_left.x.array[:] = bound[i][0]
                uD_right = dolfinx.fem.Function(V)
                uD_right.x.array[:] = bound[i][-1]
                dofs_left = dolfinx.fem.locate_dofs_topological((V0, V), fdim, self.left_facet)
                dofs_right = dolfinx.fem.locate_dofs_topological((V0, V), fdim, self.right_facet)
                bc_left = dolfinx.fem.dirichletbc(uD_left, dofs_left, V0)
                bc_right = dolfinx.fem.dirichletbc(uD_right, dofs_right, V0)
                bc.append(bc_left)
                bc.append(bc_right)
        return bc        

    def _boundary_init(self):
        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        fdim = self.mesh.topology.dim - 1
        self.left_facet = dolfinx.mesh.locate_entities_boundary(self.mesh, fdim, self.left_boundary)
        self.right_facet = dolfinx.mesh.locate_entities_boundary(self.mesh, fdim, self.right_boundary)
        bcs = []
        # shunt current boundary conditions
        Is_bounds = [[0., 0.]]*self.Settings.n_manifolds
        self.bc_W = self._apply_bound(Is_bounds, "shunt")
        # mass rate bounds
        G_in = np.array(self.Conditions.Q_in) * self.Manifolds[0].liquid["rho"]
        G_bounds = []
        uD_left = dolfinx.fem.Function(self.V)
        uD_right = dolfinx.fem.Function(self.V)
        for i in range(2):
            if self.Settings.n_manifolds < 4:
                bound = ([0.5 * G_in[0] / self.Settings.N, 0.5 * G_in[0] * (1 - 1/self.Settings.N)] if self.Settings.outflow == "forward"
                         else [0.5 * G_in[0] * (1 - 1/self.Settings.N), 0.5 * G_in[0] / self.Settings.N])
                
            else:
                bound = ([G_in[i] / self.Settings.N, G_in[i] * (1 - 1/self.Settings.N)] if self.Settings.outflow == "forward"
                         else [G_in[i] * (1 - 1/self.Settings.N), G_in[i] / self.Settings.N])
            G_bounds.append(bound)
        self.bc_V = self._apply_bound(G_bounds, "flow")
        
        
    def _boundary_update(self):
        Is_bounds = []
        # update shunt current boundaries
        for i in range(self.Settings.n_manifolds):
            Rd, Ra, V = self._get_shunt_coeff(self.Manifolds[i])
            Ix = self.Manifolds[i].I
            if self.Manifolds[i].type == "anodic":
                Rax = Ra[1]
                Rdx = Rd[1]
                dRdx = Rd[1] - Rd[0]
                Ix_ = (V[1] + (Rax + 2 * Rdx - dRdx) * Ix[1] - Rdx * Ix[2]) / (Rdx - dRdx)
                bound = [Ix_, 0]
            elif self.Manifolds[i].type == "cathodic":
                Rax = Ra[-2]
                Rdx = Rd[-2]
                dRdx = Rd[-1] - Rd[-2]
                Ix_ = (V[-2] + (Rax + 2 * Rdx - dRdx) * Ix[-2] - Rdx * Ix[-3]) / (Rdx + dRdx)
                bound = [0, Ix_]
            else:
                # left
                Rax_l = Ra[1]
                Rdx_l = Rd[1]
                dRdx_l = Rd[1] - Rd[0]
                Ix_l = (V[1] + (Rax_l + 2 * Rdx_l - dRdx_l) * Ix[1] - Rdx_l * Ix[2]) / (Rdx_l - dRdx_l)
                # right
                Rax_r = Ra[-2]
                Rdx_r = Rd[-2]
                dRdx_r = Rd[-1] - Rd[-2]
                Ix_r = (V[-2] + (Rax_r + 2 * Rdx_r - dRdx_r) * Ix[-2] - Rdx_r * Ix[-3]) / (Rdx_r + dRdx_r)
                bound = [Ix_l, Ix_r]
            Is_bounds.append(bound)
        self.bc_W = self._apply_bound(Is_bounds, "shunt")
        # update flow boundaries
        outlet_manifold_l = self.get_section("outlet-manifold", side="anodic")
        outlet_manifold_k = self.get_section("outlet-manifold", side="cathodic")
        outlet_manifolds = [outlet_manifold_l, outlet_manifold_k]
        sides = ["anodic", "cathodic"]
        other_outlet = [outlet_manifold_k, outlet_manifold_l]
        G_bounds = []
        for i in range(2):
            Gx = outlet_manifolds[i].G
            WT, WMT, St = self._get_flow_coeff(outlet_manifolds[i], other_outlet[i], sides[i])
            # left boundary
            WTx_l = WT[1]
            WMTx_l = WMT[1]
            dWTx_l = WT[1] - WT[0]
            Gx_l = (St[1] + (WMTx_l + 2 * WTx_l - dWTx_l) * Gx[1] - WTx_l * Gx[2]) / (WTx_l - dWTx_l)
            # right boundary
            WTx_r = WT[-2]
            WMTx_r = WMT[-2]
            dWTx_r = WT[-1] - WT[-2]
            Gx_r = (St[-2] + (WMTx_r + 2 * WTx_r + dWTx_r) * Gx[-2] - WTx_r * Gx[-3]) / (WTx_r + dWTx_r)
            bound = [Gx_l, Gx_r]
            G_bounds.append(bound)
        self.bc_V = self._apply_bound(G_bounds, "flow")
    
    def _update_state(self, uh, wh):
        # shunt current fields
        n_man = self.Settings.n_manifolds
        Isv = np.array([wh.sub(i).collapse().x.array[:] for i in range(n_man)])
        Gsv = np.array([uh.sub(i).collapse().x.array[:] for i in range(2)])
        self.update_shunt_fields(Isv)
        # flow fileds
        outflow = self.Settings.outflow
        Gx = 0 * Gsv if n_man > 3 else np.flipud(Gsv)
        lp = [-1, 1] if outflow == "forward" else [1, 0]
        self.update_flow_fields(Gsv, Gx, ["anodic", "cathodic"], lp, self.friction_model)


    # Coefficients
    def _get_flow_coeff(self, out_manifold, other_out_manifold, side):
        outflow = self.Settings.outflow
        n_man = self.Settings.n_manifolds
        N = self.Settings.N
        G_init = out_manifold.G
        Gx = other_out_manifold.G if self.Settings.n_manifolds < 4 else 0. * other_out_manifold.G
        if n_man > 3:
            Q_in = self.Conditions.Q_in[0] if side == "anodic" else self.Conditions.Q_in[-1]
        else:
            Q_in = self.Conditions.Q_in[0]
        G_in = Q_in * self.Manifolds[0].liquid["rho"] 
        
        lp = [-1, 1] if outflow == "forward" else [1, 0]

        # get inlet manifold
        self.update_mass_flow(G_in, G_init, Gx, side, lp)
        # forward flow sections
        in_manifold = self.get_section("inlet-manifold", side)
        out_manifold = self.get_section("outlet-manifold", side)
        obj_list_forward = [in_manifold, out_manifold]
        # upward flow sections
        in_channel = self.get_section("inlet-channel", side)
        half_cell = self.get_section("cell", side)
        out_channel = self.get_section("outlet-channel", side)
        obj_list_upward = [in_channel, half_cell, out_channel]

        # mass flow resistances
        WT = self.upward_flow_resistance(obj_list_upward, lp[0])
        WMT = self.forward_flow_resistance(obj_list_forward, lp[0])
        # source term
        Wm = in_manifold.mass_flow_resistance(self.Settings.roughness_factor, self.friction_model)
        St = -(self.g * half_cell.H * np.gradient(half_cell.rho, edge_order=2) + Wm * (lp[-1] * G_in + lp[0] * Gx))
        
        return WT, WMT, St
    
    def _get_shunt_coeff(self, manifold_obj):
        n_man = self.Settings.n_manifolds
        I_calc = []
        Ra = self.get_resistance(manifold_obj)

        if manifold_obj.outlet == 0 and n_man < 4:
            inlet_channels = [obj for obj in self.Channels if getattr(obj, "outlet", None) == 0]
            Rd = self.get_paralle_resistance(inlet_channels)
        else:
            Channel = [obj for obj in self.Channels if getattr(obj, "outlet", None) == manifold_obj.outlet and 
                        getattr(obj, "type", None) == manifold_obj.type][0]
            # channels
            Rd = self.get_resistance(Channel)
            
        # get cell voltage
        j = self.stack_current_density()
        V = self.mapping_voltage(j)

        return Rd, Ra, V

    # variational problem
    def _variational_form(self, uh, wh, v, q):
        xp = np.linspace(0, self.Settings.N-1, self.Settings.N)
        # Coefficient functions
        Rd, WT = dolfinx.fem.Function(self.W), dolfinx.fem.Function(self.V)
        Ra, WMT = dolfinx.fem.Function(self.W), dolfinx.fem.Function(self.V)
        Vs, St = dolfinx.fem.Function(self.W), dolfinx.fem.Function(self.V)
        psi = ufl.split(uh)
        # shunt currents
        F_u = 0.
        # flow rates
        out_manifolds = [element for element in self.Manifolds if element.outlet]
        other_out_manifolds = [out_manifolds[0], out_manifolds[-1]]
        sides = [element.type for element in self.Manifolds if element.outlet]
        for i in range(2):
            coeff_vals = self._get_flow_coeff(out_manifolds[i], other_out_manifolds[i], sides[i])
            WT_Fncs, WMT_Fncs, St_Fncs = [CubicSpline(xp, vals) for vals in coeff_vals]
            WT.sub(i).interpolate(lambda x: WT_Fncs(x[0]))
            WMT.sub(i).interpolate(lambda x: WMT_Fncs(x[0]))
            St.sub(i).interpolate(lambda x: St_Fncs(x[0]))
            F_u += (WT.sub(i) * ufl.inner(ufl.grad(psi[i]), ufl.grad(v[i])) * ufl.dx)  # diffusive term
            F_u += (WMT.sub(i) * ufl.inner(psi[i], v[i]) * ufl.dx)  # absorption term
            F_u -= ufl.inner(St.sub(i), v[i]) * ufl.dx  # source term
        # shunt
        F_w = 0.
        psi = ufl.split(wh)
        for i in range(self.Settings.n_manifolds):
            coeff_vals = self._get_shunt_coeff(self.Manifolds[i])
            Rd_Fncs, Ra_Fncs, Vs_Fncs = [CubicSpline(xp, vals) for vals in coeff_vals]
            Rd.sub(i).interpolate(lambda x: Rd_Fncs(x[0]))
            Ra.sub(i).interpolate(lambda x: Ra_Fncs(x[0]))
            Vs.sub(i).interpolate(lambda x: Vs_Fncs(x[0]))
            F_w += (Rd.sub(i) * ufl.inner(ufl.grad(psi[i]), ufl.grad(q[i])) * ufl.dx)  # diffusive term
            F_w += (Ra.sub(i) * ufl.inner(psi[i], q[i]) * ufl.dx)  # absorption term
            F_w -= ufl.inner(Vs.sub(i), q[i]) * ufl.dx  # source term   

        return F_u, F_w

    def initialize_fields(self, yh):
        # discretization
        xp = np.linspace(0, self.Settings.N-1, self.Settings.N)
        # flow fields
        outlet_manifold_l = self.get_section("outlet-manifold", side="anodic")
        outlet_manifold_k = self.get_section("outlet-manifold", side="cathodic")
        outlet_manifolds = [outlet_manifold_l, outlet_manifold_k]
        for i in range(2):
            G = outlet_manifolds[i].G
            f = CubicSpline(xp, G)
            yh[0].sub(i).interpolate(lambda x: f(x[0]))
        # shunt fields
        n_man = self.Settings.n_manifolds
        for i in range(n_man):
            I = self.Manifolds[i].I
            f = CubicSpline(xp, I)
            yh[1].sub(i).interpolate(lambda x: f(x[0]))
        return yh
    
    # solver
    def solve_problem(self):
        # solutino control
        uh_prev = dolfinx.fem.Function(self.V)
        wh_prev = dolfinx.fem.Function(self.W)
        yh_prev = [uh_prev, wh_prev]
        du = dolfinx.fem.Function(self.V)
        dw = dolfinx.fem.Function(self.W)
        uh_prev.x.array[:] = 1e3 # initialize for a large error in flow
        wh_prev.x.array[:] = 1e3 # initialze for a large error in shunt current
        dy = [du, dw]
        
        bc_u = self.bc_V
        bc_w = self.bc_W
        bcs = [bc_u, bc_w]
        
        uh_ = dolfinx.fem.Function(self.V)
        wh_ = dolfinx.fem.Function(self.W)
        yh = [uh_, wh_]
        yh = self.initialize_fields(yh)
        # self._update_state(uh_, wh_)
        
        v = ufl.TestFunctions(self.V)
        q = ufl.TestFunction(self.W)
        
        
        F_u, F_w = self._variational_form(uh_, wh_, v, q)
        
        J_u = ufl.derivative(F_u, uh_)
        J_w = ufl.derivative(F_w, wh_)
        
        residual_u = dolfinx.fem.form(F_u)
        residual_w = dolfinx.fem.form(F_w)
        residuals = [residual_u, residual_w]
        
        jacobian_u = dolfinx.fem.form(J_u)
        jacobian_w = dolfinx.fem.form(J_w)
        jacobians = [jacobian_u, jacobian_w]
        
        # matrix handling
        A_u = dolfinx.fem.petsc.create_matrix(jacobian_u)
        L_u = dolfinx.fem.petsc.create_vector(residual_u)
        
        A_w = dolfinx.fem.petsc.create_matrix(jacobian_w)
        L_w = dolfinx.fem.petsc.create_vector(residual_w)
        
        A = [A_u, A_w]
        L = [L_u, L_w]
        
        # solvers
        solver_u = PETSc.KSP().create(self.mesh.comm)
        solver_w = PETSc.KSP().create(self.mesh.comm)
        
        solver_u.setOperators(A_u)
        solver_w.setOperators(A_w)
        solvers = [solver_u, solver_w]
        # solver.krylov_solver.set_monitor(True)
        # iterations
        counter = 0
        error_u = dolfinx.fem.form(ufl.inner(uh_ - uh_prev, uh_ - uh_prev) * ufl.dx(metadata={"quadrature_degree": 4}))
        error_w = dolfinx.fem.form(ufl.inner(wh_ - wh_prev, wh_ - wh_prev) * ufl.dx(metadata={"quadrature_degree": 4}))
        error = [error_u, error_w]

        L2_error = [[],[]]
        du_norm = [[],[]]
        while counter < self.Error_params.max_iter:
            for i in range(2):
                # assemble jacobian and residual
                with L[i].localForm() as loc_L:
                    loc_L.set(0)
                    [bc.set(loc_L.array) for bc in bcs[i]]
                A[i].zeroEntries()
                dolfinx.fem.petsc.assemble_matrix(A[i], jacobians[i], bcs=bcs[i])
                A[i].assemble()

                dolfinx.fem.petsc.assemble_vector(L[i], residuals[i])
                L[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                L[i].scale(-1)

                # Compute b - J(u_D-u_(i-1))
                dolfinx.fem.petsc.apply_lifting(L[i], [jacobians[i]], [bcs[i]], x0=[yh[i].x.petsc_vec], alpha=0.8)
                # Set du|_bc = u_{i-1}-u_D
                dolfinx.fem.petsc.set_bc(L[i], bcs[i], yh[i].x.petsc_vec, alpha=0.8)
                L[i].ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

                # solve linear problem
                solvers[i].solve(L[i], dy[i].x.petsc_vec)
                dy[i].x.scatter_forward()

                # Update u_{i+1} = u_i + delta u_i
                yh_prev[i].x.array[:] = yh[i].x.array[:]
                yh[i].x.array[:] += dy[i].x.array
                
                
                self._update_state(yh[0], yh[1]) # update fields
                self._boundary_update() # update boundary conditions
                
                # Compute norm of update
                correction_norm = dy[i].x.petsc_vec.norm(0) 
                
                # conpute L2 error compared to previous solution
                L2_error[i].append(np.sqrt(self.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error[i]), op=MPI.SUM)))
                du_norm[i].append(correction_norm)
                field = "mass rates" if i == 0 else "shunt currents"
                # print(f"Iteration {counter} - {field}: Correction norm {correction_norm}, L2 error: {L2_error[i][-1]}")   
                
            max_correction_norm = max([du_norm[0][-1], du_norm[1][-1]])
            max_error = max([L2_error[0][-1], L2_error[1][-1]])           
                
            counter+=1

            if max_correction_norm < 1e-9 and max_error < self.Error_params.tol:
                break
        
if __name__ == "__main__":
    s = stackProblem("./templates/conditions_three_manifolds.json", friction_model="Colebrook")
    s.solve_problem()
    Q_l = s.Cell.anode_side.G / s.Cell.anode_side.rho
    Q_k = s.Cell.cathode_side.G / s.Cell.cathode_side.rho
    # plt.plot((Q_l + Q_k) * 1e3 * 3600)
    # plt.plot(s.Manifolds[-1].G)
    # plt.plot(s.Manifolds[0].p - s.Cell.anode_side.p)
    plt.plot(s.Manifolds[0].p - s.Manifolds[-2].p)
    # plt.plot(s.Manifolds[0].p)
    # plt.plot(s.stack_current_density() / s.Conditions.j_load)
    plt.show()
    # print(f"finished")

    

