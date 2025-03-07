import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import csv

class FigureHandler:

    def __init__(self, config):
        """
        config: configuration instance for PDE values
        """
        self.config = config
        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100 #change resolution, standard is 100
        })

    def plot_heatmaps(self, phi_solutions, n_solutions, times_to_plot, savefig = False):
        """
        - phi_solutions, n_solutions: Array of phi, n values
        - times_to_plot: times for the solutions
        - savefig: binary variable if figure should be saved
        """

        solutions_path = "output"
        phi_cmap = "RdBu_r" #colormap red (low) to blue (high)
        n_cmap = "gray_r"   #colormap white (low) to black (high)

        for k in range(len(phi_solutions)):
            plt.figure()
            c_phi = fe.plot(phi_solutions[k], mode="color")
            c_phi.set_cmap(phi_cmap)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(rf"$\phi$ at t = {round(times_to_plot[k], 2)}")
            plt.colorbar(c_phi)
            if savefig:
                plt.savefig(f"{solutions_path}/phi_t={round(times_to_plot[k],2)}.png")

            plt.figure()
            c_n = fe.plot(n_solutions[k], mode = "color")
            c_n.set_cmap(n_cmap)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"n at t = {round(times_to_plot[k], 2)}")
            plt.colorbar(c_n)
            if savefig:
                plt.savefig(f"{solutions_path}/n_t={round(times_to_plot[k],2)}.png")
        
        plt.show()


    def plot_free_energy(self, free_energy_vals, savefig=False):
        """
        Plot the evolution of the free energy
        - free_energy_vals: array containing free energy values
        - savefig: binary variable whether figures should be saved 
        """

        path = "output"
        plt.figure(figsize=(12, 6))
        plt.plot([self.config.dt * k for k in range(self.config.num_steps)], free_energy_vals)
        plt.xlabel("Time t")
        plt.ylabel("Free energy F")
        plt.title("Evolution of free energy")

        if savefig:
            plt.savefig(f"{path}/free_energy.png")

        plt.show()


    def plot_horizontal_slice_n(self, n_solutions, times_to_plot, ycoord, num_x_points, savefig=False):
        """
        Horizontal slice of the ink distribution n
        - n_solutions: ink distribution solution
        - times_to_plots: times of slices
        - ycoord: y coordinate where the horizontal slice starts
        - num_x_points: number of points along the horizontal slice
        """

        path = "output"
        xvals = np.linspace(0, self.config.L, num_x_points)
        n_vals = []
        plt.figure()
        for i in range(len(n_solutions)):
            n_vals.append([n_solutions[i](xvals[j], ycoord) for j in range(len(xvals))])
            plt.plot(xvals, n_vals[i], label=f"t={round(times_to_plot[i], 2)}")
        plt.xlabel("x")
        plt.ylabel("n")
        plt.legend()

        if savefig:
            plt.savefig(f"{path}/slice_plot.png")
        plt.show()

    @staticmethod
    def read_csv(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            return [float(row[0]) for row in reader]
        
    def mu_dependence_plots(self, mu_list, path_distance_phi, path_distance_n, path_kullback_n):
        distance_phi_list = self.read_csv(path_distance_phi)
        distance_n_list = self.read_csv(path_distance_n)
        kullback_n_list = self.read_csv(path_kullback_n)

        plt.figure()
        plt.plot(mu_list, distance_phi_list, label="distance\nmeasure phi")
        plt.legend(loc="lower left")

        plt.figure()
        plt.plot(mu_list, distance_n_list, label="distance\nmeasure n")
        plt.legend(loc="upper left")

        plt.figure()
        plt.plot(mu_list, kullback_n_list, label="kullback\nmeasure n")
        plt.legend(loc="upper left")
        plt.show()
