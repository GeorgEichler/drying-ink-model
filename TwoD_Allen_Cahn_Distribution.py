import fenics as fe
import ufl #needed to use exp, tanh etc. function for fenics code
import config as cfg
import os
import plot_results as plot



class AllenCahnInk2D:

    def __init__(self, config, phi_init_option, n_init_option, times_to_plot = [0], store_values = False):
        """
        - config: class setting all parameters for the equation
        - phi_init_options: set initial conditions for phi (see config file)
        - n_init_options: set initial conditions for n (see config file)
        - store_values: if true store heatmaps for phi and n
        """

        self.config = config
        self.phi_init_option = phi_init_option
        self.n_init_option = n_init_option
        self.times_to_plot = times_to_plot
        self.store_values = store_values

        self.phi_k, self.n_k = self.config.set_ics(phi_init_option, n_init_option)
        self.phi_0 = self.phi_k.copy(deepcopy=True)
        self.n_0 = self.n_k.copy(deepcopy=True)

        # Weak formulation
        self.v = fe.TestFunction(self.config.V)
        self.phi = fe.Function(self.config.V)
        self.n = fe.Function(self.config.V)

        # Allen-Cahn equation for phi
        F_phi = ((self.phi - self.phi_k) / config.dt ) * self.v * fe.dx \
                + fe.inner(fe.grad(self.phi), fe.grad(self.v)) * fe.dx \
                - (self.phi - self.phi**3 - config.eps * self.n_k + config.mu) * self.v * fe.dx 
        J_phi = fe.derivative(F_phi, self.phi)

        # Ink distribution diffusion equation
        F_n = ((self.n - self.n_k) / config.dt) * self.v * fe.dx \
            + (1/config.alpha) * config.D(self.phi,self.n) * fe.inner(fe.grad(self.n) + config.eps * self.n * fe.grad(self.phi), fe.grad(self.v)) * fe.dx
        J_n = fe.derivative(F_n, self.n)

        # Define the free energy
        #free_energy = (-(phi**2)/2 + (phi**4)/4 + 1/2*fe.inner(fe.grad(phi), fe.grad(phi)) \
        #            - config.mu*phi + config.eps*n*phi + n*(ufl.ln(n) - 1))*fe.dx


        # Set up solvers
        problem_phi = fe.NonlinearVariationalProblem(F_phi, self.phi, J=J_phi)
        self.solver_phi = fe.NonlinearVariationalSolver(problem_phi)

        problem_n = fe.NonlinearVariationalProblem(F_n, self.n, J=J_n)
        self.solver_n = fe.NonlinearVariationalSolver(problem_n)

        if self.store_values:
            self.init_pvd_file()

        
    def init_pvd_file(self):
        output_dir_phi = "output/phi_solutions"
        output_dir_n = "output/n_solutions"
        os.makedirs(output_dir_phi, exist_ok=True)
        os.makedirs(output_dir_n, exist_ok=True)

        """
        phi_xdmf = fe.XDMFFile(os.path.join(output_dir_phi, "phi_solution.xdmf"))
        n_xdmf = fe.XDMFFile(os.path.join(output_dir_n, "n_solution.xdmf"))
        
        phi_xdmf.parameters["flush_output"] = True
        n_xdmf.parameters["flush_output"] = True
        phi_xdmf.parameters["functions_share_mesh"] = True
        n_xdmf.parameters["functions_share_mesh"] = True

        phi_xdmf.write(phi_k, 0)
        n_xdmf.write(n_k, 0)
        """

        self.phi_file = fe.File(os.path.join(output_dir_phi, "phi_solution.pvd"))
        self.n_file = fe.File(os.path.join(output_dir_n, "n_solution.pvd"))

        # Parameters or txt file
        params = f"""
        Simulation Parameter:
        ---------------------
        Total time: {self.config.T}
        Time step number: {self.config.num_steps}
        Time step size: {self.config.dt}
        Moisture parameter (mu): {self.config.mu}
        Domain length: {self.config.L}
        Ink solvent interaction (eps) {self.config.eps}
        Evaporation coefficient * surface tension = M * sigma = {self.config.alpha}
        Mesh Resolution: {self.config.nx} x {self.config.ny}"""

        txt_name = "Parameter_values.txt"
        param_file_phi = os.path.join(output_dir_phi, txt_name)
        param_file_n = os.path.join(output_dir_n, txt_name)

        with open(param_file_phi, "w") as f:
            f.write(params)
        with open(param_file_n, "w") as f:
            f.write(params)

        #storing initial conditions
        self.write_pvd_file(self.phi_k, self.n_k, 0)

    def write_pvd_file(self, phi, n, t):
        self.phi_file << (phi, t)
        self.n_file << (n, t)

    @staticmethod
    def kullback_leibler_divergence(f, f_0):
        try:
            f_0_normalised = f_0 / (fe.assemble(f_0 * fe.dx))
            f_normalised = f/ (fe.assemble(f * fe.dx))
            result = fe.assemble(f_normalised*ufl.ln(f_normalised/f_0_normalised) * fe.dx)
            return result
        except ValueError as e:
            print(f"Error {e}")
            return None

    @staticmethod
    def distance_measure(f, f_0):
        try:
            result = fe.assemble((f-f_0)**2 * fe.dx)/(fe.assemble(f_0*fe.dx) * fe.assemble(f*fe.dx)) 
            return result
        except ValueError as e:
            print(f"Error {e}")
            return None       

    @staticmethod
    def normalised_distance_measure(f, f_0):
        try:
            result = fe.assemble((f - f_0)**2 * fe.dx)/( fe.assemble(f_0**2 * fe.dx) + fe.assemble(f**2 * fe.dx) )
            return result
        except ValueError as e:
            print(f"Error {e}")
            return None

    def solve(self):
        phi_solutions = [self.phi_0]
        n_solutions = [self.n_0]

        for i in range(self.config.num_steps):
            t = (i+1)*self.config.dt
            self.solver_phi.solve()
            self.phi_k.assign(self.phi)

            self.solver_n.solve()
            self.n_k.assign(self.n)

            if (i+1) in self.times_to_plot:
                phi_solutions.append(self.phi.copy(deepcopy=True))
                n_solutions.append(self.n.copy(deepcopy=True))

            if self.store_values:
                self.write_pvd_file(self.phi_k, self.n_k, t)

        distance_measure_phi = self.normalised_distance_measure(self.phi, self.phi_0)
        distance_measure_n = self.normalised_distance_measure(self.n, self.n_0)
        kullback_leibler_n = self.kullback_leibler_divergence(self.n, self.n_0)

        return phi_solutions, n_solutions, distance_measure_n, distance_measure_phi, kullback_leibler_n