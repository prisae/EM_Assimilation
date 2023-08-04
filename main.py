from utils import *
from model_esmda import Model
import dask.array
from dask.distributed import Client
import emg3d
import math
import h5py


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


# Function to get the temperature from a reservoir instance.
def get_temperature(instance):
    nb = instance.reservoir.mesh.n_res_blocks
    nv = instance.physics.n_vars
    x_array = np.array(instance.physics.engine.X, copy=False)
    tempr = backwards_t_ph_vector(x_array[0:nb * nv:nv] / 10, x_array[1:nb * nv:nv] / 18.015)
    return tempr


# Function to get the electric field.
def calculate_e_field(instance, receivers_vector, source_vector_ff, em_scaling_ff):
    data_receivers = np.zeros(len(receivers_vector)*len(source_vector_ff))

    # Calculate the electric field response for each source.
    for src_index in range(len(source_vector_ff)):
        # Calculate the conductivity
        tempr = get_temperature(instance)
        salinity = 1e6 * 0.1  # Salinity in ppm.

        # Empirical relationship from the book from Dresser Industries.
        c_w = (0.0123 + 3647.5 / np.power(salinity, 0.955) * 82 * 1 / ((tempr - 273.15) * 1.8 + 39))
        c_w = np.reciprocal(c_w)

        c = c_w * instance.poro * instance.poro
        c[c < 1 / 15] = 1 / 15  # Minimum conductivity in S.
        cond = (c.reshape(instance.nx, instance.ny))

        source = emg3d.TxElectricDipole(coordinates=source_vector_ff[src_index])

        # Reservoir model parameters, origin represents the bottom center of the model.
        origin = (0, 0, -2100)
        x_grid = np.ones(instance.nx) * 20
        y_grid = np.ones(instance.ny) * 20
        z_grid = np.ones(instance.nz) * 100
        grid = emg3d.TensorMesh([x_grid, y_grid, z_grid], origin)
        model = emg3d.Model(grid, property_x=np.rot90(cond, k=3), mapping='Conductivity')

        # Create the computational domain to avoid boundary effects.
        grid_ext = emg3d.construct_mesh(
            frequency=1,  # 1 Hz => Source frequency.
            properties=1,  # 1 S/m => Conductivity assumed to calculate the necessary buffer zone.
            center=source.center,  # => Source center.
            vector=(grid.nodes_x, grid.nodes_y, grid.nodes_z),  # The vectors of the DARTS model.
            mapping='Conductivity',
        )

        # Interpolate the survey domain to the computational domain.
        model_ext = model.interpolate_to_grid(grid_ext)

        # Calculate the electric field.
        e_field = emg3d.solve_source(model=model_ext, source=source, frequency=1, verb=4)

        # Sample at each receiver, take the electric field amplitude, and scale the measurement.
        for receiver in range(len(receivers_vector)):
            data_receivers[receiver + src_index * len(receivers_vector)] =\
                np.abs(e_field.get_receiver(receivers_vector[receiver])) * em_scaling_ff
    return data_receivers


# Create a directory to store the results.
def create_directory(search_directory, location):
    destination_directory = os.path.join(search_directory, location)

    # check if destDir is exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # check if destDir is empty. If not, delete the files
    if len(os.listdir(destination_directory)) > 0:
        files = glob.glob(destination_directory + '/*')
        for f in files:
            os.remove(f)

    return destination_directory


# Simulate an ensemble model and gather the measurement data.
def run_realizations(parameter_vector, time_range_f, time_step_f, receivers_f, run_em, columns_name_list_f,
                     destination_name, i_name, save_temperature_f, use_gradients_f, parameter_type_f,
                     source_vector_f, em_scaling_f):
    parameter_vector_copy = parameter_vector
    m = Model(parameter_vector_copy, parameter_type_f)
    m.init()
    receiver_measurements = np.zeros((len(time_range_f), len(receivers_f)*len(source_vector_f)))

    for a in range(len(time_range_f)):
        m.run_python(time_step)
        if run_em == 1:
            receiver_measurements[a, :] = calculate_e_field(m, receivers_f, source_vector_f, em_scaling_f)

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)

    if run_em > 0:
        if only_EM:
            time_data_full = pd.concat([time_data, pd.DataFrame(receiver_measurements)], axis=1)
            time_data_full.to_pickle(f'{destination_name}/Simulation_{i_name}.pkl')
            time_data = pd.concat([time_data['time'], pd.DataFrame(receiver_measurements)], axis=1)
        else:
            time_data = pd.concat([time_data, pd.DataFrame(receiver_measurements)], axis=1)
            time_data.to_pickle(f'{destination_name}/Simulation_{i_name}.pkl')
    else:
        time_data.to_pickle(f'{destination_name}/Simulation_{i_name}.pkl')

    realization_dataset = time_data
    obs_values = 0

    if run_em == 0:
        obs_data = realization_dataset[realization_dataset['time'].isin(time_range_f + time_step_f)]
        obs_data = obs_data[columns_name_list_f]
        obs_values = np.array(obs_data)

    if run_em > 0:
        if only_EM:
            obs_values = receiver_measurements
        else:
            obs_data = realization_dataset[realization_dataset['time'].isin(time_range_f + time_step_f)]
            obs_data = obs_data[columns_name_list_f]
            obs_values = np.array(obs_data)
            obs_values = np.hstack((obs_values, receiver_measurements))

    if use_gradients_f:
        gradients = np.ones_like(obs_values)
        gradients[1:, :] = (obs_values[:-1, :] - obs_values[1:, :]) * 1 / time_step_f
        obs_values = np.vstack((obs_values, gradients[1:, :]))

    if save_temperature_f:
        tempr = get_temperature(m)
        np.save(f'{destination_name}/temperature_{i_name}.txt', tempr)
    np.save(f'{destination_name}/observations_{i_name}.txt', obs_values)
    d_obs = obs_values.T.flatten()
    d_obs = MultipliNegatives(d_obs)
    return d_obs


# This __name__ check is required for the DASK parallelization.
if __name__ == '__main__':
    # Amount of cores and threads per core to use. Note that Numpy has limitations on the amount of files which can be
    # opened. For this reason do not use more than 50 cores.
    client = Client(n_workers=50, threads_per_worker=1)

    # Reservoir grid.
    nx = 140  # Amount of grid blocks in the x-direction.
    ny = 140  # Amount of grid blocks in the y-direction.
    nz = 1  # Amount of grid blocks in the z-direction.

    # Data types.
    use_gradients = True  # Use the change of each measurement over time as additional observation data.
    save_temperature = True  # Save the final temperature field.
    parameter_type = 0  # 0 is porosity, 1 is permeability.
    RunEM = 1  # 0 no EM receiver observations, 1 use and simulate EM receiver data.
    only_EM = False  # If EM is used, toggle to only rely on well data and disregard well data.

    # ES-MDA.
    Ne = 100  # Amount of realizations
    es_mda_iterations = 6  # Amount of ES-MDA iterations.
    alphas = np.array([1 / 0.02, 1 / 0.05, 1 / 0.08, 1 / 0.2, 1 / 0.3, 1 / 0.4])  # Alphas for ES-MDA iterations.

    # EM survey design and scaling.
    em_scaling = 10e14  # Scaling factor for the EM observations.
    source_vector = [(400, 1400, 0, 0, 90), (1000, 1000, 0, 0, 90), (1000, 1800, 0, 0, 90), (400, 1000, 0, 0, 90),
                     (400, 1800, 0, 0, 90)]  # Source configuration (x, y, z, azimuth, elev).
    receivers = [(1400, 1400, -2050, 0, 90)]  # Receiver configuration (x, y, z, azimuth, elev).

    # Simulation time.
    time_step = 365 * 5  # Amount of time between observations
    training_time = 365 * 35  # Amount of time to collection assimilation data for.
    total_time = 365 * 50  # Total time to run the simulations for.

    time_range = np.arange(0, training_time, time_step)
    total_time_range = np.arange(0, total_time, time_step)

    # Production well data.
    columns_name_list = ['P1 : temperature (K)', 'P1 : BHP (bar)']
    datatypes_per_well = 2

    # Define the directories.
    curDir = os.getcwd()
    srcDir = f'{curDir}'

    # Prior ensemble models.
    parameter = np.zeros([nx * ny, Ne])
    for i in range(Ne):
        if parameter_type == 0:
            parameter[:, i] = np.genfromtxt('dap_well_like/poro' + str(i) + '.txt', skip_header=True, skip_footer=True)
        elif parameter_type == 1:
            parameter[:, i] = np.genfromtxt('dap_well_like/permx' + str(i) + '.txt', skip_header=True, skip_footer=True)
        else:
            print('Unknown parameter type, defaulting to porosity.')
            parameter[:, i] = np.genfromtxt('dap_well_like/poro' + str(i) + '.txt', skip_header=True, skip_footer=True)

    # Reference model. Here the 100th realization is taken as the reference model.
    if parameter_type == 0:
        parameter_true = np.genfromtxt('dap_well_like/poro' + str(100) + '.txt', skip_header=True, skip_footer=True)
    elif parameter_type == 1:
        parameter_true = np.genfromtxt('dap_well_like/permx' + str(100) + '.txt', skip_header=True, skip_footer=True)
    else:
        parameter_true = np.genfromtxt('dap_well_like/poro' + str(100) + '.txt', skip_header=True, skip_footer=True)

    # Run the reference model for the training time to get observation for the training period.
    # Create a directory for the reference solution files. The 0 and 1 are used to create the name for the results file.
    directory_name = create_directory(srcDir, f'data/simulations/true')
    dObs = run_realizations(parameter_true, time_range, time_step, receivers, RunEM, columns_name_list, directory_name,
                            0, save_temperature, use_gradients, parameter_type, source_vector, em_scaling)

    # Run the reference model for the entire time span to be able to compare it with the forecast of the ensemble.
    run_realizations(parameter_true, total_time_range, time_step, receivers, RunEM, columns_name_list, directory_name,
                     1, save_temperature, use_gradients, parameter_type, source_vector, em_scaling)

    # Uncertainty and noise.
    uncertainty_well_data = 0.001  # Noise and uncertainty for the production well observations.
    uncertainty_em_data = 0.01  # Noise and uncertainty for the EM observations.

    # Create the covariance matrix using the expected uncertainty.
    receiver_matrix = np.ones((np.size(time_range), len(receivers)*len(source_vector))) * uncertainty_em_data
    if only_EM:
        well_matrix = receiver_matrix
    else:
        well_matrix = np.ones((np.size(time_range), datatypes_per_well)) * uncertainty_well_data
        if RunEM > 0:
            well_matrix = np.hstack((well_matrix, receiver_matrix))

    if use_gradients:
        well_matrix = np.vstack((well_matrix, well_matrix[:-1, :]))

    # Create the observed data vector by adding noise to the reference model observations.
    dObs = dObs + dObs * well_matrix.T.flatten() * np.random.uniform(-1, 1, len(dObs))

    # Create the covariance matrix using the uncertainty of the observations and observation data.
    CeDiag = well_matrix.T.flatten() * dObs[:]

    # Simulate prior ensemble.
    directory_name = create_directory(srcDir, f'data/simulations/prior')
    results_dask = []
    for i in range(Ne):
        results_dask.append(dask.delayed(run_realizations)(parameter[:, i], total_time_range, time_step, receivers,
                                                           RunEM, columns_name_list, directory_name, i,
                                                           save_temperature, use_gradients, parameter_type,
                                                           source_vector, em_scaling))

    dask.array.compute(*results_dask)

    # ---------------------------------------------------- #
    # -----------------DATA ASSIMILATION------------------ #
    # ---------------------------------------------------- #

    # Determine parameters for the data assimilation
    NGrid = nx * ny
    NScalar = 0  # Scalar parameters in the problem like kro and krw are not considered.
    Nm = NGrid + NScalar
    Nd = len(dObs)
    csi = 0.99  # SVD truncation parameter for SVD.

    # Prepare the prior ensemble model vectors for data assimilation.
    mList = []
    for i in range(Ne):
        mList.append(parameter[:, i])

    MGridPrior = np.transpose(np.array(mList).reshape((Ne, NGrid)))

    # Define the error matrix for the analysis step.
    SDiag = np.sqrt(CeDiag)
    SInvDiag = np.power(SDiag, -1)
    INd = np.eye(Nd)

    MGrid = MGridPrior

    MObj = np.zeros([len(alphas), Ne])

    # Run
    start = time.time()

    L = 0
    D = np.zeros((Nd, Ne))
    for alpha in alphas:
        # 2. Forecast
        destDir = create_directory(srcDir, f'data/simulations/it{L}')

        # Generates the perturbed observations.
        z = np.random.normal(size=(Nd, Ne))
        DPObs = dObs[:, np.newaxis] + math.sqrt(alpha) * CeDiag[:, np.newaxis] * z

        # Run the simulations g(M).
        results_dask = []
        for i in range(Ne):
            results_dask.append(dask.delayed(run_realizations)(MGrid[:, i], time_range, time_step, receivers, RunEM,
                                                               columns_name_list, destDir, i,
                                                               save_temperature, use_gradients, parameter_type,
                                                               source_vector, em_scaling))

        computed_dask = dask.array.compute(*results_dask)

        for i in range(Ne):
            D[:, i] = computed_dask[i]

        if L == 0:
            DPrior = D

        DobsD = DPObs - D

        # Calculates DeltaD.
        meanMatrix = np.mean(D, axis=1)
        DeltaD = D - meanMatrix[:, np.newaxis]
        print(f'Finished runs of ES-MDA iteration: {L}')

        # Calculates CHat (12.10)
        CHat = SInvDiag[:, np.newaxis] * (DeltaD @ DeltaD.T) * SInvDiag[np.newaxis, :] + alpha * (Ne - 1) * INd

        # Calculates Gamma and X.
        U, SigmaDiag, Vt = np.linalg.svd(CHat)
        Nr = FindTruncationNumber(SigmaDiag, csi)

        GammaDiag = np.power(SigmaDiag[0:Nr], -1)
        X = SInvDiag[:, np.newaxis] * U[:, 0:Nr]

        # Calculates M^a.
        X1 = GammaDiag[:, np.newaxis] * X.T
        X8 = DeltaD.T @ X
        X9 = X8 @ X1

        MGrid = UpdateModel(MGrid, X9, DobsD)

        # Constrain the parameter field to be within the expected range.
        if parameter_type == 0:
            MGrid[MGrid > 0.40] = 0.40
            MGrid[MGrid < 0.05] = 0.05
        elif parameter_type == 1:
            MGrid[MGrid > 196449 * 0.40 ** 4.3762] = 196449 * 0.40 ** 4.3762
            MGrid[MGrid < 196449 * 0.05 ** 4.3762] = 196449 * 0.05 ** 4.3762
        else:
            MGrid[MGrid > 0.40] = 0.40
            MGrid[MGrid < 0.05] = 0.05
        pd.DataFrame(MGrid).to_pickle(f'{destDir}/MGrid_{L}.pkl')

        CeInv = np.power(CeDiag, -1)
        MObj[L, :] = calcDataMismatchObjectiveFunction(dObs[:, np.newaxis], D, CeInv)
        pd.DataFrame(MObj[L, :]).to_pickle(f'{destDir}/MObj_{L}.pkl')

        L += 1

    destDir = create_directory(srcDir, f'data/simulations/post')

    results_dask = []
    for i in range(Ne):
        results_dask.append(dask.delayed(run_realizations)(MGrid[:, i], total_time_range, time_step, receivers, RunEM,
                                                           columns_name_list, destDir, i,
                                                           save_temperature, use_gradients, parameter_type,
                                                           source_vector, em_scaling))

    computed_dask = dask.array.compute(*results_dask)

    end = time.time()
    elapsed = end - start
    print('Elapsed time of the ES-MDA: ', elapsed)
