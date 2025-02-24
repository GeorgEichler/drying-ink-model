import fenics as fe
import matplotlib.pyplot as plt

def plot_heatmaps(phi_solutions, n_solutions, times_to_plot, dt):
    fig, axes = plt.subplots(2, len(times_to_plot), figsize=(15, 6))
    cmap = "coolwarm"

    for idx, t_idx in enumerate(times_to_plot):
        plt.sca(axes[0, idx])
        c_phi = fe.plot(phi_solutions[idx], mode="color")
        c_phi.set_cmap(cmap)
        axes[0, idx].set_title(f"phi at t={t_idx*dt:.2f}")
        plt.colorbar(c_phi, ax=axes[0, idx])

        plt.sca(axes[1, idx])
        c_n = fe.plot(n_solutions[idx], mode="color")
        c_n.set_cmap(cmap)
        axes[1, idx].set_title(f"n at t={t_idx*dt:.2f}")
        plt.colorbar(c_n, ax=axes[1, idx])
    
    plt.tight_layout()
    plt.show()