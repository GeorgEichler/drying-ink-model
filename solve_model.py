import TwoD_Allen_Cahn_Distribution as AC
import config as cfg
import plot_results as fh
import matplotlib.pyplot as plt

config = cfg.Config(mu = 0, num_steps=20, final_time=5)

#phi initial condition options: constant, tanh, cosine, gaussian, sine_checkerboard, random
phi_init_option = "gaussian"

#available n initial condition options: constant, gaussian
n_init_option = "constant"

times_to_plot = [0, config.num_steps //2, config.num_steps]
allen_cahn = AC.AllenCahnInk2D(config, phi_init_option, n_init_option, times_to_plot, store_values = True)
phi_solutions, n_solutions, distance_measure_n, distance_measure_phi, kullback_measure_n = allen_cahn.solve()
phi_0, n_0 = phi_solutions[0], n_solutions[0]

figure_handler = fh.FigureHandler(config)

save_heatmap = False
save_free_energy = False
save_slices = False

figure_handler.plot_heatmaps(phi_solutions, n_solutions, times_to_plot, save_heatmap)
#figure_handler.plot_free_energy(free_energy_vals, save_free_energy)
figure_handler.plot_horizontal_slice(phi_solutions, times_to_plot, config.L/2, config.grid[0],variable = "$\phi$")
figure_handler.plot_horizontal_slice(n_solutions, times_to_plot, config.L/2, config.grid[0], variable = "n")
plt.show()