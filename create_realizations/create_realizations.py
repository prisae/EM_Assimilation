import geostatspy.GSLIB as GSLIB                          # GSLIB utilities, viz and wrapped functions
import geostatspy.geostats as geostats                    # GSLIB converted to Python
import matplotlib.pyplot as plt                           # plotting
import scipy.stats                                        # summary stats of nd arrays
import numpy as np
import random as rand
import pandas as pd
import time
import statsmodels


# Add samples with random values according to the moments of the field.
def add_samples(array, seed, nx, ny, cell_size):
    np.random.seed(seed)
    for i in range(25):  # Amount of random samples.
        array.loc[len(array.index)] = [np.random.randint(0, nx * cell_size), np.random.randint(0, ny * cell_size),
                                       np.random.normal(0.2192, 0.0891)]  # Mean and std of the field.
    return array


def make_realizations(left_bound, right_bound):
    nx = 140
    ny = 140
    cell_size = 20

    # Add initial sample data if available. (example)
    sample_data = pd.DataFrame(np.c_[[800.0, 2000.0],
                                     [2400.0, 2400.0],
                                     [0.2900, 0.2900]], columns=["X", "Y", "Porosity"])

    # Create a variogram for the sequential gaussian simulation.
    var = GSLIB.make_variogram(nug=0, nst=1, it1=1, cc1=1, azi1=0, hmaj1=1000, hmin1=1000)

    # Go through the left and right boundary of the seed numbers.
    counter = 0
    for i in range(left_bound, right_bound):
        print(i)

        # Add additional random samples to the existing samples.
        sample_data2 = add_samples(sample_data.copy(), i, nx, ny, cell_size)
        cond_por_sim = GSLIB.sgsim(1, sample_data2, 'X', 'Y', 'Porosity', nx, ny, cell_size, i, var,
                                   'simulation_cond.out')

        porosity = cond_por_sim
        porosity = porosity.reshape(nx, ny)

        # Boundaries on the porosity, in this case porosity 0.05 <= phi <= 0.40.
        porosity[porosity > 0.40] = 0.40
        porosity[porosity < 0.05] = 0.05

        # Empirical relationship to link porosity to permeability.
        permeability = 196449 * porosity ** 4.3762

        # Check to only create a realization if the porosity and permeability at the well are sufficiently large.
        if 800 <= permeability[70:71, 40] <= 1000 and 800 <= permeability[70:71, 100] <= 1000:
            title = "Porosity Realization " + str(i)

            # Plot the field and its statistics.
            plt.figure(figsize=(16, 12))

            plt.subplot(2, 2, 1)
            plt.imshow(porosity, vmin=0.05, vmax=0.40)
            plt.gca().invert_yaxis()
            plt.colorbar(label="(-)")
            plt.title(title)
            plt.xlabel("x-index")
            plt.ylabel("y-index")


            plt.subplot(2, 2, 2)
            plt.imshow(permeability)
            plt.gca().invert_yaxis()
            plt.colorbar(label="mD")
            plt.title("Permeability")
            plt.xlabel("x-index")
            plt.ylabel("y-index")


            plt.subplot(2, 2, 3)
            GSLIB.hist_st(permeability.flatten(), 0, np.max(permeability), log=False, cumul=False, bins=10,
                          weights=None, xlabel="Permeability (mD)", title="Permeability histogram")

            plt.show()

            # Save the realization.
            perm_name = "realizations/permx" + str(counter) + ".txt"
            poro_name = "realizations/poro" + str(counter) + ".txt"
            np.savetxt(perm_name, permeability, delimiter='\n', newline='\n', header="PERMX", footer="/", comments='')
            np.savetxt(poro_name, porosity, delimiter='\n', newline='\n', header="PORO", footer="/", comments='')
            print("Realization: " + str(i) + " is valid!")
            counter += 1


# Create realizations between a certain set of seeds.
make_realizations(0, 10000)
