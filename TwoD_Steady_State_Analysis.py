import fenics as fe
import matplotlib.pyplot as plt

def solve_steady_state(config, phi_init, n_init, max_iter = 20, tol = 1e-6):
    """
    Solving for the steady state of the PDE

    Parameters:
    - config: Configuration instance for PDE
    - phi_init, n_init: Initial guesses for phi,n (FEniCS Function)
    - max_iter: Maximum iterations for convergence
    - tol: Convergence tolearance

    Returns:
    - phi, n: Solutions of the iteration process
    """

    # Define test function
    v = fe.TestFunction(config.V)

    phi = fe.Function(config.V)
    n = fe.Function(config.V)

    phi.assign(phi_init)
    n.assign(n_init)

    F_phi_steady = fe.inner(fe.grad(phi), fe.grad(v)) * fe.dx \
                - (phi - phi**3 - config.eps * n + config.mu) * v * fe.dx
    J_phi_steady = fe.derivative(F_phi_steady, phi)

    F_n_steady = (1/config.alpha) * config.D(phi,n) * fe.inner(fe.grad(n) + config.eps * n * fe.grad(phi), fe.grad(v)) * fe.dx
    J_n_steady = fe.derivative(F_n_steady, n)

    # Set up the solvers
    problem_phi = fe.NonlinearVariationalProblem(F_phi_steady, phi, J=J_phi_steady)
    solver_phi = fe.NonlinearVariationalSolver(problem_phi)

    problem_n = fe.NonlinearVariationalProblem(F_n_steady, n, J=J_n_steady)
    solver_n = fe.NonlinearVariationalSolver(problem_n)

    for i in range(max_iter):
        phi_old = phi.copy(deepcopy=True)
        n_old = n.copy(deepcopy=True)

        solver_phi.solve()
        solver_n.solve()

        phi_diff = fe.errornorm(phi, phi_old, norm_type="L2")
        n_diff = fe.errornorm(n, n_old, norm_type = "L2")

        if phi_diff < tol and n_diff < tol:
            print(f"Converged in {i+1} iterations.")
            break
        else:
            print("Warning: Maximum iterations reached without full convergence.")


    # Plot results
    plt.figure()
    p = fe.plot(phi)
    plt.colorbar(p)
    plt.title('Solution for phi')

    plt.figure()
    n_plot = fe.plot(n)
    plt.colorbar(n_plot)
    plt.title('Solution for n')

    plt.show()