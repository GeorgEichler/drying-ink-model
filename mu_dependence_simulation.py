import TwoD_Allen_Cahn_Distribution as AC
import config as cfg
import csv
import numpy as np

"""
Solve this model for various values of mu and saves mu list the different measures in a csv file.
"""

def write_csv(mu_list, values_list, name):
    with open(f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(["mu", name])
        # Write data
        for mu, value in zip(mu_list, values_list):
            writer.writerow([mu, value])


# available phi ic options: constant, tanh, cosine, gaussian, sine_checkerboard, random
phi_init_option = "gaussian"

# available n ic options: constant, gaussian, half_domain, two_drop
n_init_option = "gaussian"

mu_list = np.linspace(-1.5, 1, 41)
distance_phi_list = []
distance_n_list = []
kullback_n_list = []
for mu in mu_list:
    print(f"mu = {mu}")
    config = cfg.Config(mu = mu)
    allen_cahn = AC.AllenCahnInk2D(config, phi_init_option, n_init_option)
    _, _, distance_measure_n, distance_measure_phi, kullback_measure_n = allen_cahn.solve()
    distance_phi_list.append(distance_measure_phi)
    distance_n_list.append(distance_measure_n)
    kullback_n_list.append(kullback_measure_n)
    print("#############################################################################")

print(distance_phi_list)
print(distance_n_list)
print(kullback_n_list)
write_csv(mu_list, distance_phi_list, "Results/distance_phi_list")
write_csv(mu_list, distance_n_list, "Results/distance_n_list")
write_csv(mu_list, kullback_n_list, "Results/kullback_n_list")
