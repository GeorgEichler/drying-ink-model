import fenics as fe
import numpy as np

class Config:
    def __init__(self, domain_size = (100,100), domain_length = 25, num_steps = 50, time_interval = 10,
                 mu = 0.0, eps = -1, M = 0.2, sigma = 0.03, constant_phi = 0, constant_n = 0):

        self.domain = domain_size
        self.L = domain_length
        self.num_steps = num_steps
        self.T = time_interval
        self.dt = self.time_interval/self.num_steps
        self.mu = mu
        self.eps = eps
        self.alpha = M * sigma
        self.c_phi = constant_phi
        self.c_n = constant_n

        self.nx, self.ny = domain_size
        self.mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(self.L, self.L), self.nx, self.ny)
        self.V = fe.FunctionSpace(self.mesh, "Lagrange", 1)

        self.phi_options = {
            "constant": fe.Constant(self.c_phi),
            "sine_checkerboard": fe.Expression("sin(x[0])*sin(x[1])", degree=2),
            "gaussian": fe.Expression("0.1*exp(-pow(x[0]-L/2, 2)/0.01) * exp(-pow(x[1]-L/2, 2)/0.01)", L=self.L, degree=2),
            "random": self.random()
        }

        self.n_options = {
            "constant": fe.Constant(self.c_n),
            "gaussian": fe.Expression("0.1*exp(-pow(x[0]-L/2, 2)/0.01) * exp(-pow(x[1]-L/2, 2)/0.01)", L=self.L, degree=2),
            "cross_gaussian": fe.Expression("0.1 * exp(-(pow(x[0] - L/2, 2) + pow(x[1] - L/2, 2))) * (exp(-pow(x[0] + x[1] - L, 2)/(2*0.1)) + exp(-pow(x[0] - x[1], 2)/(2*0.1)))",
                                    L=self.L, degree=2),
            "half_domain": fe.Expression("x[0] < L/2 ? 0.2 : 0", L=self.L, degree=1)
        }

    def random(self):
        random_values = np.random.uniform(-0.1, 0.1, (self.nx+1,self.ny+1))
        phi_init = fe.Function(self.V)
        phi_init.vector()[:] = random_values.flatten()
        return phi_init
        
    def set_ics(self, phi_option, n_option):
        phi_init = fe.interpolate(self.phi_options[phi_option])
        n_init = fe.interpolate(self.n_options[n_option])

        return phi_init, n_init
