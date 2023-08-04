import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import emg3d
from matplotlib.colors import LogNorm

# List of directories of the fields to show the moments of.
locations = ['fields_well_750_1400', 'fields_well_1500_1400', 'fields_well_2000_1400']

# Show the temperature field moments.
print_temperatures = True

# Number of ensemble members.
Ne = 100

# Iterate through the folders of which to show the moments of.
for location in locations:

    # Load one of the files to get the dimensions of various datasets.
    directory = f'data/{location}'
    name = f'{directory}/e_fields_0.h5'
    temporary = emg3d.io.load(name)
    temporary_e_field_1 = temporary['e_field_1']
    temporary_e_field_2 = temporary['e_field_2']
    temporary_t_field = temporary['t_field']
    temporary_grid_ext = temporary['grid_ext']
    temporary_ratio = temporary_e_field_1.fz.ravel('F')/temporary_e_field_2.fz.ravel('F')

    # Iterate through all the files to load in all the data.
    field_array = np.zeros((len(temporary_ratio), Ne))
    temperature_array = np.zeros((len(temporary_t_field), Ne))
    for i in range(Ne):
        name = f'{directory}/e_fields_{i}.h5'

        test = emg3d.io.load(name)
        e_field_1 = test['e_field_1']
        e_field_2 = test['e_field_2']
        grid_ext = test['grid_ext']
        field_array[:, i] = e_field_1.fz.ravel('F') / e_field_2.fz.ravel('F')
        temperature_array[:, i] = test['t_field']

        # Code to see the amplitude difference between the initial and end state.
        # receiver = (400, 1400, -2050, 0, 90)
        # test_1 = np.abs(e_field_1.get_receiver(receiver))
        # test_2 = np.abs(e_field_2.get_receiver(receiver))
        # print(test_1, test_2)

    # Calculate the moments.
    field_array = np.nan_to_num(field_array, nan=1)  # To remove any potential NaNs due to the division for the ratio.
    mean = np.mean(field_array, axis=1)
    std = np.std(field_array, axis=1)

    # Plot the mean electric field.
    fig = plt.figure()
    temporary_grid_ext.plot_3d_slicer(
       mean, view='abs', v_type='Ez', xlim=[0, 2800], ylim=[0, 2800], zlim=[-2100, -2000], clim=[0.5, 1.0], fig=fig
    )

    axs = fig.get_children()
    fig.suptitle('Mean electric field ratio (750, 1400).', y=0.95, va="center", fontsize=12)
    axs[1].set_ylabel("Y-axis (m)", fontsize=12)
    axs[2].set_xlabel("X-axis (m)", fontsize=12)
    axs[2].set_ylabel("Z-axis (m)", fontsize=12)
    axs[4].set_ylabel("Ratio (-)", fontsize=12)
    fig.show()

    # Plot the standard deviation of the electric fields.
    fig = plt.figure()
    temporary_grid_ext.plot_3d_slicer(
       std, view='abs', v_type='Ez', xlim=[0, 2800], ylim=[0, 2800], zlim=[-2100, -2000], clim=[0, 0.2], fig=fig
    )
    axs = fig.get_children()
    fig.suptitle('Standard deviation electric field ratio (750, 1400).', y=0.95, va="center", fontsize=12)
    axs[1].set_ylabel("Y-axis (m)", fontsize=12)
    axs[2].set_xlabel("X-axis (m)", fontsize=12)
    axs[2].set_ylabel("Z-axis (m)", fontsize=12)
    axs[4].set_ylabel("Ratio (-)", fontsize=12)
    fig.show()

    # Print the moments of the temperature field.
    if print_temperatures:
        t_mean = np.mean(temperature_array, axis=1)
        t_std = np.std(temperature_array, axis=1)

        # Code to draw circles to indicate source locations.
        radius = 50
        circle1 = plt.Circle((750, 1400), radius, edgecolor='r', facecolor='r')
        circle2 = plt.Circle((1500, 1400), radius, edgecolor='b', facecolor='b')
        circle3 = plt.Circle((2000, 1400), radius, edgecolor='g', facecolor='g')
        plt.imshow(t_mean.reshape(140, 140), extent=[0, 2800, 0, 2800], clim=[310, 350])
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(circle3)

        plt.title('Mean temperature field after 25 years.', fontsize=12)
        color_bar = plt.colorbar()
        color_bar.set_label('Mean temperature (K)', fontsize=12)
        plt.xlabel('Y-axis (m)', fontsize=12)
        plt.ylabel('X-axis (m)', fontsize=12)
        plt.show()

        circle4 = plt.Circle((750, 1400), radius, edgecolor='r', facecolor='r')
        circle5 = plt.Circle((1500, 1400), radius, edgecolor='b', facecolor='b')
        circle6 = plt.Circle((2000, 1400), radius, edgecolor='g', facecolor='g')

        plt.imshow(t_std.reshape(140, 140), extent=[0, 2800, 0, 2800])
        plt.gca().add_patch(circle4)
        plt.gca().add_patch(circle5)
        plt.gca().add_patch(circle6)
        plt.title('Standard deviation of the temperature field after 25 years.', fontsize=12)
        color_bar = plt.colorbar()
        color_bar.set_label('Standard deviation temperature (K)', fontsize=12)
        plt.xlabel('Y-axis (m)', fontsize=12)
        plt.ylabel('X-axis (m)', fontsize=12)
        plt.show()

        print_temperatures = False  # This data only needs to be plotted once.
