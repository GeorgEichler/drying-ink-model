import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmaps(phi_solutions, n_solutions, times_to_plot, dt):
    fig, axes = plt.subplots(2, len(times_to_plot), figsize=(15, 6))
    phi_cmap = "RdBu_r" #colormap red (low) to blue (high)
    n_cmap = "gray_r"   #colormap white (low) to black (high)

    for idx, t_idx in enumerate(times_to_plot):
        plt.sca(axes[0, idx])
        c_phi = fe.plot(phi_solutions[idx], mode="color")
        c_phi.set_cmap(phi_cmap)
        axes[0, idx].set_title(f"phi at t={t_idx*dt:.2f}")
        plt.colorbar(c_phi, ax=axes[0, idx])

        plt.sca(axes[1, idx])
        c_n = fe.plot(n_solutions[idx], mode="color")
        c_n.set_cmap(n_cmap)
        axes[1, idx].set_title(f"n at t={t_idx*dt:.2f}")
        plt.colorbar(c_n, ax=axes[1, idx])
    
    plt.tight_layout()
    plt.show()

def plot_free_energy(free_enrgy_vals, dt, num_steps):
    plt.figure(figsize=(12, 6))
    plt.plot([dt * k for k in range(num_steps)], free_enrgy_vals)
    plt.xlabel("Time t")
    plt.ylabel("Free Energy F")
    plt.title("Evolution of free energy")

    plt.show()

def plot_horizontal_slice_n(n_solutions, L, timestamps, ycoord, num_x_points):
    """
    Horizontal slice of the ink distribution n
    n: ink distribution solution
    L: domain length
    timestamps: times of slices
    ycoord: y coordinate where the horizontal slice starts
    num_x_points: number of points along the horizontal slice
    """

    xvals = np.linspace(0, L, num_x_points)
    n_vals = []
    plt.figure()
    for i in range(len(n_solutions)):
        n_vals.append([n_solutions[i](xvals[j], ycoord) for j in range(len(xvals))])
        plt.plot(xvals, n_vals[i], label=f"t={round(timestamps[i], 2)}")
    plt.xlabel("x")
    plt.ylabel("n")
    plt.legend()
    plt.show()
