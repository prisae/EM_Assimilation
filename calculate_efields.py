from utils import *
from model_esmda import Model
import dask.array
from dask.distributed import Client
import emg3d
import math
from matplotlib.colors import LogNorm
# import h5py


# Function to convert the t_ph vector to temperature.
def backwards_t_ph_vector(p, h):
    i_element = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6]
    j_element = [0, 1, 2, 6, 22, 32, 0, 1, 2, 3, 4, 10, 32, 10, 32, 10, 32, 32, 32, 32]
    n = [-0.23872489924521e3, 0.40421188637945e3, 0.11349746881718e3,
         -0.58457616048039e1, -0.15285482413140e-3, -0.10866707695377e-5,
         -0.13391744872602e2, 0.43211039183559e2, -0.54010067170506e2,
         0.30535892203916e2, -0.65964749423638e1, 0.93965400878363e-2,
         0.11573647505340e-6, -0.25858641282073e-4, -0.40644363084799e-8,
         0.66456186191635e-7, 0.80670734103027e-10, -0.93477771213947e-12,
         0.58265442020601e-14, -0.15020185953503e-16]

    pr = p / 1
    nu = h / 2500
    t = np.zeros(p.shape)
    for i_index, j_index, ni_index in zip(i_element, j_element, n):
        t += ni_index * pr ** i_index * (nu + 1) ** j_index
    return t


# Create a directory to store the results.
def create_directory(search_directory, location):
    destination_directory = os.path.join(search_directory, location)
    # check if destination directory exists.
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    # # Check if the destination directory is empty. If not, delete the files.
    # if len(os.listdir(destination_directory)) > 0:
    #     files = glob.glob(destination_directory + '/*')
    #     for f in files:
    #         os.remove(f)

    return destination_directory


# Function to get the temperature from a reservoir instance.
def get_temperature(instance):
    nb = instance.reservoir.mesh.n_res_blocks
    nv = instance.physics.n_vars
    x_array = np.array(instance.physics.engine.X, copy=False)
    tempr = backwards_t_ph_vector(x_array[0:nb * nv:nv] / 10, x_array[1:nb * nv:nv] / 18.015)
    return tempr


# Function to get the electric field.
def calculate_e_field(instance, src_coo, plot_individual_responses_ff):
    # Get the temperature data.
    tempr = get_temperature(instance)

    salinity = 1e6 * 0.1  # Salinity in ppm.
    # Empirical relationship from the book from Dresser Industries.
    c_w = (0.0123 + 3647.5 / np.power(salinity, 0.955) * 82 * 1 / ((tempr - 273.15) * 1.8 + 39))
    c_w = np.reciprocal(c_w)
    c = c_w * instance.poro * instance.poro
    c[c < 1 / 15] = 1 / 15  # Minimum conductivity in S.
    cond = (c.reshape(instance.nx, instance.ny))

    # Get the initial conductivity state at year 0.
    c_w_begin = (0.0123 + 3647.5 / np.power(salinity, 0.955) * 82 * 1 / ((np.max(tempr) - 273.15) * 1.8 + 39))
    c_w_begin = np.reciprocal(c_w_begin)
    c_begin = c_w_begin * instance.poro * instance.poro
    c_begin[c_begin < 1/15] = 1 / 15  # Minimum conductivity in S.
    cond_begin = (c_begin.reshape(instance.nx, instance.ny))

    source = emg3d.TxElectricDipole(coordinates=src_coo)  # Initialize the source.

    # Reservoir model parameters, origin represents the bottom center of the model.
    origin = (0, 0, -2100)
    x_grid = np.ones(instance.nx) * 20
    y_grid = np.ones(instance.ny) * 20
    z_grid = np.ones(instance.nz) * 100
    grid = emg3d.TensorMesh([x_grid, y_grid, z_grid], origin)

    # Create a separate grid for both the start and end state.
    model = emg3d.Model(grid, property_x=np.rot90(cond, k=3), mapping='Conductivity')
    model_begin = emg3d.Model(grid, property_x=np.rot90(cond_begin, k=3), mapping='Conductivity')

    # Create a computational domain grid.
    grid_ext = emg3d.construct_mesh(
        frequency=1,  # 1 Hz => Source frequency.
        properties=1,  # 1 S/m => Conductivity assumed to calculate the necessary buffer zone.
        center=source.center,  # => Source center.
        vector=(grid.nodes_x, grid.nodes_y, grid.nodes_z),  # The vectors of the DARTS model.
        mapping='Conductivity',
    )

    # Interpolate survey domain to the computational domain.
    model_ext = model.interpolate_to_grid(grid_ext)
    model_ext_begin = model_begin.interpolate_to_grid(grid_ext)

    # Calculate the electric fields.
    e_field_1 = emg3d.solve_source(model=model_ext_begin, source=source, frequency=1, verb=4)
    e_field_2 = emg3d.solve_source(model=model_ext, source=source, frequency=1, verb=4)

    # Plot the individual responses for each ensemble member.
    if plot_individual_responses_ff:

        # Conductivity field year 0.
        fig = plt.figure()
        grid_ext.plot_3d_slicer(
            model_ext_begin.property_x, xlim=[0, 2800], ylim=[0, 2800], zlim=[-2100, -2000], clim=[0, 4], fig=fig
        )

        axs = fig.get_children()
        fig.suptitle('Conductivity field at year 0.', y=0.95, va="center", fontsize=12)
        axs[1].set_ylabel("Y-axis (m)", fontsize=12)
        axs[2].set_xlabel("X-axis (m)", fontsize=12)
        axs[2].set_ylabel("Z-axis (m)", fontsize=12)
        axs[4].set_ylabel("Conductivity (S)", fontsize=12)
        fig.show()

        # Conductivity field year at the end.
        fig = plt.figure()
        grid_ext.plot_3d_slicer(
            model_ext.property_x, xlim=[0, 2800], ylim=[0, 2800], zlim=[-2100, -2000], clim=[0, 4], fig=fig
        )

        axs = fig.get_children()
        fig.suptitle('Conductivity field at year 25.', y=0.95, va="center", fontsize=12)
        axs[1].set_ylabel("Y-axis (m)", fontsize=12)
        axs[2].set_xlabel("X-axis (m)", fontsize=12)
        axs[2].set_ylabel("Z-axis (m)", fontsize=12)
        axs[4].set_ylabel("Conductivity (S)", fontsize=12)
        fig.show()

        # Difference in the conductivity.
        fig = plt.figure()
        grid_ext.plot_3d_slicer(
            model_ext.property_x-model_ext_begin.property_x, xlim=[0, 2800], ylim=[0, 2800], zlim=[-2100, -2000],
            fig=fig
        )

        axs = fig.get_children()
        fig.suptitle('Change in conductivity.', y=0.95, va="center", fontsize=12)
        axs[1].set_ylabel("Y-axis (m)", fontsize=12)
        axs[2].set_xlabel("X-axis (m)", fontsize=12)
        axs[2].set_ylabel("Z-axis (m)", fontsize=12)
        axs[4].set_ylabel("Conductivity (S)", fontsize=12)
        fig.show()

        # Temperature field.
        model_temp = emg3d.Model(grid, property_x=np.rot90(tempr.reshape((140, 140)), k=3))
        fig = plt.figure()
        grid.plot_3d_slicer(
            model_temp.property_x, xlim=[0, 2800], ylim=[0, 2800], zlim=[-2100, -2000], fig=fig
        )

        axs = fig.get_children()
        fig.suptitle('Temperature field at year 25.', y=0.95, va="center", fontsize=12)
        axs[1].set_ylabel("Y-axis (m)", fontsize=12)
        axs[2].set_xlabel("X-axis (m)", fontsize=12)
        axs[2].set_ylabel("Z-axis (m)", fontsize=12)
        axs[4].set_ylabel("Temperature (K)", fontsize=12)
        fig.show()

        # Electric field for the initial state.
        fig = plt.figure()
        grid_ext.plot_3d_slicer(
            e_field_1.fz.ravel('F'), view='abs', v_type='Ez', xlim=[0, 2800], ylim=[0, 2800], zlim=[-2100, -2000],
            clim=[10e-11, 10e-13],
            pcolor_opts={'norm': LogNorm()}, fig=fig
        )

        axs = fig.get_children()
        fig.suptitle('Electric field amplitude at year 0.', y=0.95, va="center", fontsize=12)
        axs[1].set_ylabel("Y-axis (m)", fontsize=12)
        axs[2].set_xlabel("X-axis (m)", fontsize=12)
        axs[2].set_ylabel("Z-axis (m)", fontsize=12)
        axs[4].set_ylabel("Electric field amplitude (V/m)", fontsize=12)
        fig.show()

        # Electric field for the end state.
        fig = plt.figure()
        grid_ext.plot_3d_slicer(
            e_field_2.fz.ravel('F'), view='abs', v_type='Ez', xlim=[0, 2800], ylim=[0, 2800], zlim=[-2100, -2000],
            clim=[10e-11, 10e-13],
            pcolor_opts={'norm': LogNorm()}, fig=fig
        )

        axs = fig.get_children()
        fig.suptitle('Electric field amplitude at year 25.', y=0.95, va="center", fontsize=12)
        axs[1].set_ylabel("Y-axis (m)", fontsize=12)
        axs[2].set_xlabel("X-axis (m)", fontsize=12)
        axs[2].set_ylabel("Z-axis (m)", fontsize=12)
        axs[4].set_ylabel("Electric field amplitude (V/m)", fontsize=12)
        fig.show()

        # Ratio of the electric fields.
        fig = plt.figure()
        grid_ext.plot_3d_slicer(
            e_field_1.fz.ravel('F')/e_field_2.fz.ravel('F'), view='abs', v_type='Ez', xlim=[0, 2800], ylim=[0, 2800],
            zlim=[-2100, -2000], clim=[0.5, 1], fig=fig
        )

        axs = fig.get_children()
        fig.suptitle('Electric field ratio.', y=0.95, va="center", fontsize=12)
        axs[1].set_ylabel("Y-axis (m)", fontsize=12)
        axs[2].set_xlabel("X-axis (m)", fontsize=12)
        axs[2].set_ylabel("Z-axis (m)", fontsize=12)
        axs[4].set_ylabel("Ratio (-)", fontsize=12)
        fig.show()
    return e_field_1, e_field_2, model_ext_begin, model_ext, tempr, grid_ext,


def run_realizations(parameter_vector, time_range_f, time_step_f, parameter_type_f, index, destination_name,
                     source_location_f, plot_individual_responses_f):
    parameter_vector_copy = parameter_vector
    m = Model(parameter_vector_copy, parameter_type_f)
    m.init()

    for a in range(len(time_range_f)):
        m.run_python(time_step_f)

    e_field_1, e_field_2, cond_1, cond_2, t_field, grid_ext = calculate_e_field(m, source_location_f,
                                                                                plot_individual_responses_f)

    emg3d.save(f'{destination_name}/e_fields_{index}.h5', e_field_1=e_field_1, e_field_2=e_field_2,
               cond_1=cond_1, cond_2=cond_2, grid_ext=grid_ext, t_field=t_field)


# This __name__ check is required for the DASK parallelization.
if __name__ == '__main__':
    # Amount of cores and threads per core to use. Note that Numpy has limitations on the amount of files which can be
    # opened. For this reason do not use more than 50 cores
    client = Client(n_workers=1, threads_per_worker=1)

    # Plot the responses for each ensemble member (Conductivity, temperature, differences)
    plot_individual_responses = True

    # Reservoir grid.
    nx = 140
    ny = 140
    nz = 1

    Ne = 100  # Number of ensemble members.
    parameter_type = 0  # 0 is porosity, 1 is permeability

    # Define the time array.
    time_step = 365 * 1
    training_time = 365 * 25
    time_range = np.arange(0, training_time, time_step)

    # Define the directories.
    curDir = os.getcwd()
    srcDir = f'{curDir}'

    # Load prior ensemble models.
    parameter = np.zeros([nx * ny, Ne+1])
    for i in range(Ne+1):
        if parameter_type == 0:
            parameter[:, i] = np.genfromtxt('dap_well_like/poro' + str(i) + '.txt', skip_header=True, skip_footer=True)
        elif parameter_type == 1:
            parameter[:, i] = np.genfromtxt('dap_well_like/permx' + str(i) + '.txt', skip_header=True, skip_footer=True)
        else:
            print('Unknown parameter type, defaulting to porosity.')
            parameter[:, i] = np.genfromtxt('dap_well_like/poro' + str(i) + '.txt', skip_header=True, skip_footer=True)

    # Examples of how to save the electric field response for different source locations as shown in the paper.
    # Run the simulation for the entire time span.

    directory_name = create_directory(srcDir, f'data/fields_well_400_1400')
    source_location = (400, 1400, 0, 0, 90)  # (x, y, z, azimuth, elevation)
    results_dask = []
    for iteration in range(Ne):
        results_dask.append(dask.delayed(run_realizations)(parameter[:, iteration], time_range, time_step,
                                                           parameter_type, iteration, directory_name, source_location,
                                                           plot_individual_responses))

    dask.array.compute(*results_dask)

    directory_name = create_directory(srcDir, f'data/fields_well_750_1400')
    source_location = (750, 1400, 0, 0, 90)  # (x, y, z, azimuth, elevation)
    results_dask = []
    for iteration in range(Ne):
        results_dask.append(dask.delayed(run_realizations)(parameter[:, iteration], time_range, time_step,
                                                           parameter_type, iteration, directory_name, source_location,
                                                           plot_individual_responses))

    dask.array.compute(*results_dask)

    # Run the simulation for the entire time span.
    directory_name = create_directory(srcDir, f'data/fields_well_1500_1400')
    source_location = (1500, 1400, 0, 0, 90)  # (x, y, z, azimuth, elevation)
    results_dask = []
    for iteration in range(Ne):
        results_dask.append(dask.delayed(run_realizations)(parameter[:, iteration], time_range, time_step,
                                                           parameter_type, iteration, directory_name, source_location,
                                                           plot_individual_responses))

    dask.array.compute(*results_dask)

    # Run the simulation for the entire time span.
    directory_name = create_directory(srcDir, f'data/fields_well_2000_1400')
    source_location = (2000, 1400, 0, 0, 90)  # (x, y, z, azimuth, elevation)
    results_dask = []
    for iteration in range(Ne):
        results_dask.append(dask.delayed(run_realizations)(parameter[:, iteration], time_range, time_step,
                                                           parameter_type, iteration, directory_name, source_location,
                                                           plot_individual_responses))

    dask.array.compute(*results_dask)
