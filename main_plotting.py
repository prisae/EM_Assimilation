import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Retrieve both the temperature and BHP vectors.
def get_temperature_bhp(directory, length_f, ne_ff):
    bhp_values = np.zeros((length_f, ne_ff))
    temperature_values = np.zeros((length_f, ne_ff))
    for realization in range(ne_ff):
        temporary = pd.read_pickle(directory + '/Simulation_' + str(realization) + '.pkl')
        temperature = temporary['P1 : temperature (K)']
        bhp = temporary['P1 : BHP (bar)']
        temperature_values[:, realization] = np.array(temperature)
        bhp_values[:, realization] = np.array(bhp)
    return temperature_values, bhp_values


# Plot the bhp and temperature data.
def plot_bhp_temperature(iteration_number_f, length_f, time_f, real_temperature_f, real_bhp_f, ne_f):
    temperature_matrix, bhp_matrix = get_temperature_bhp(f'data/simulations/it{iteration_number_f-1}', length_f, ne_f)

    for ii in range(ne_f):
        plt.plot(time_f, temperature_matrix[:, ii], color='b')

    mean_temperature_f = np.mean(temperature_matrix, axis=1)
    plt.plot(time_f, real_temperature_f, color='k', label='Real Observation')
    plt.plot(time_f, mean_temperature_f, color='y', linestyle='--', label='Mean Ensemble')
    plt.legend()
    plt.ylim([335, 350])
    plt.xlim([0, time_f[-1]])
    plt.title(f'Temperature at the production well at iteration {iteration_number_f}.')
    plt.ylabel('Temperature (K)', fontsize=12)
    plt.xlabel('Time (years)', fontsize=12)
    plt.show()

    for ii in range(ne):
        plt.plot(time_f, bhp_matrix[:, ii], color='b')

    mean_bhp_f = np.mean(bhp_matrix, axis=1)
    plt.plot(time_f, real_bhp_f, color='k', label='Real Observation')
    plt.plot(time_f, mean_bhp_f, color='y', linestyle='--', label='Mean Ensemble')
    plt.legend()
    plt.ylim([180, 192])
    plt.xlim([0, time_f[-1]])
    plt.title(f'BHP at the production well at iteration {iteration_number_f}.')
    plt.ylabel('BHP (Bar)', fontsize=12)
    plt.xlabel('Time (years)', fontsize=12)
    plt.show()


ne = 100  # Amount of ensemble members
iterations = 6  # Total amount of ES_MDA iteration
iteration_number = 4  # The specific iteration to plot.
use_gradients = True  # If gradient data was used for the observations.

# Get the reference solution for the training time duration.
training_real = pd.read_pickle('data/simulations/true/Simulation_0.pkl')
training_real_temperature = np.array(training_real['P1 : temperature (K)'])
training_real_bhp = np.array(training_real['P1 : BHP (bar)'])
training_time = np.array(training_real['time'])
training_time = training_time/365
training_length = len(training_real_temperature)

# Plot a specific iteration for comparison.
plot_bhp_temperature(iteration_number, training_length, training_time, training_real_temperature, training_real_bhp, ne)

# Plot the objective values for each iteration.
objectives = np.ones((ne, iterations))

for i in range(iterations):
    objectives[:, i] = np.array(pd.read_pickle(f'data/simulations/it{i}/MObj_{i}.pkl'))[:, 0]

plt.boxplot(objectives)
plt.title('Objective function for the Entire Ensemble.')
plt.ylabel('Objective Function')
plt.xlabel('Iterations')
plt.show()

# Get the reference solution for the entire simulation time.
real = pd.read_pickle('data/simulations/true/Simulation_1.pkl')
real_temperature = np.array(real['P1 : temperature (K)'])
real_bhp = np.array(real['P1 : BHP (bar)'])
real_time = np.array(real['time'])
real_time = real_time/365
length = len(real_temperature)

# Get both the posterior and prior predictions for the entire simulation time.
temperature_matrix_prior, bhp_matrix_prior = get_temperature_bhp('data/simulations/prior', length, ne)
temperature_matrix_post, bhp_matrix_post = get_temperature_bhp('data/simulations/post', length, ne)

# Plot both posterior and prior predictions for production well temperature the entire simulation time.
for i in range(ne):
    plt.plot(real_time, temperature_matrix_prior[:, i], color='grey')

for i in range(ne):
    plt.plot(real_time, temperature_matrix_post[:, i], color='b')

# Add the mean of posterior ensemble.
mean_temperature = np.mean(temperature_matrix_post, axis=1)
plt.plot(real_time, mean_temperature, color='y', linestyle='--', label='Mean Ensemble')

# Plot the prior, posterior, and reference model.
plt.plot(real_time, real_temperature, color='k', label='Real Observation')
plt.legend()
plt.axvspan(0, training_time[-1], facecolor='green', alpha=0.1)
plt.ylim([335, 350])
plt.xlim([0, real_time[-1]])
plt.title(f'Temperature at the production well (well).', fontsize=12)
plt.ylabel('Temperature (K)', fontsize=12)
plt.xlabel('Time (years)', fontsize=12)
plt.show()

# Plot both posterior and prior predictions for production well pressure the entire simulation time.
for i in range(ne):
    plt.plot(real_time, bhp_matrix_prior[:, i], color='grey')

for i in range(ne):
    plt.plot(real_time, bhp_matrix_post[:, i], color='b')

mean_bhp = np.mean(bhp_matrix_post, axis=1)
plt.plot(real_time, real_bhp, color='k', label='Real Observation')
plt.plot(real_time, mean_bhp, color='y', linestyle='--', label='Mean Ensemble')
plt.legend()
plt.axvspan(0, training_time[-1], facecolor='g', alpha=0.1)
plt.ylim([180, 192])
plt.xlim([0, real_time[-1]])
plt.title(f'BHP at the production well.')
plt.ylabel('BHP (Bar)')
plt.xlabel('Time (years)')
plt.show()

# Retrieve the posterior temperature profiles at the end of the simulation.
temperature_profiles_post = np.ones((140*140, ne))
for i in range(ne):
    temperature_profiles_post[:, i] = np.load(f'data/simulations/post/temperature_{i}.txt.npy')

# Plot the mean posterior temperature profile.
temperature_profile_post_mean = np.mean(temperature_profiles_post, axis=1)
plt.imshow(temperature_profile_post_mean.reshape(140, 140), extent=[0, 20*140, 0, 20*140])
plt.title('Mean of the posterior temperature field.')
plt.xlabel('y-axis (m)', fontsize=12)
plt.ylabel('x-axis (m)', fontsize=12)
plt.colorbar()
plt.show()

# Plot the reference model temperature profile.
temperature_profile_post_best = temperature_profiles_post[:, np.argmin(objectives[:, iterations-1])]
plt.imshow(temperature_profile_post_best.reshape(140, 140), extent=[0, 20*140, 0, 20*140])
plt.title('Best posterior temperature field.')
plt.xlabel('y-axis (m)', fontsize=12)
plt.ylabel('x-axis (m)', fontsize=12)
plt.colorbar()
plt.show()

# Plot the posterior temperature profile of the reservoir model with the lowest objective function.
temperature_profile_true = np.load(f'data/simulations/true/temperature_1.txt.npy')
plt.imshow(temperature_profile_true.reshape(140, 140), extent=[0, 20*140, 0, 20*140])
plt.title('True temperature field.')
plt.xlabel('y-axis (m)', fontsize=12)
plt.ylabel('x-axis (m)', fontsize=12)
plt.colorbar()
plt.show()

# Plot the difference between the reference and mean posterior temperature profile.
plt.imshow(temperature_profile_true.reshape(140, 140) - temperature_profile_post_mean.reshape(140, 140),
           extent=[0, 20*140, 0, 20*140], clim=[-30, 30])
plt.title('Difference ensemble mean and true temperature field (well).', fontsize=12)
plt.xlabel('Y-axis (m)', fontsize=12)
plt.ylabel('X-axis (m)', fontsize=12)
color_bar = plt.colorbar()
color_bar.set_label('Temperature difference (K)', fontsize=12)
plt.show()

# Plot the difference between the reference and best posterior temperature profile.
plt.imshow(temperature_profile_true.reshape(140, 140) - temperature_profile_post_best.reshape(140, 140),
           extent=[0, 20*140, 0, 20*140], clim=[-30, 30])
plt.title('Difference best and true temperature field (well).', fontsize=12)
plt.xlabel('Y-axis (m)', fontsize=12)
plt.ylabel('X-axis (m)', fontsize=12)
color_bar = plt.colorbar()
color_bar.set_label('Temperature difference (K)', fontsize=12)
plt.show()

# Create a time vector for the measurement data.
time_step = 5
total_time = 50
time_range = np.arange(0, training_time[-1], time_step)
total_time_range = np.arange(0, total_time, time_step)

# Load a single observation file for the dimensions of the prior and posterior observations.
dimensions_temp = np.array(np.load(f'data/simulations/prior/observations_{0}.txt.npy'))
observations = np.ones((np.shape(dimensions_temp)[0], np.shape(dimensions_temp)[1], ne))
observations_prior = np.ones((np.shape(dimensions_temp)[0], np.shape(dimensions_temp)[1], ne))

# Calculate the amount of observations, excluding gradient observations if present.
if use_gradients:
    observation_amount = int(np.ceil(((np.shape(dimensions_temp)[0])/2)))
else:
    observation_amount = int(np.shape(dimensions_temp)[0])

# Iterate through the measurement vectors. The first two are temperature and BHP and are skipped as they are shown
# before.
for i in range(2, np.shape(dimensions_temp)[1]):

    # Load prior observations.
    for j in range(ne):
        observations_prior[:, :, j] = np.array(np.load(f'data/simulations/prior/observations_{j}.txt.npy'))
        plt.plot(total_time_range, observations_prior[:observation_amount, i, j], color='gray')

    # Load posterior observations.
    for j in range(ne):
        observations[:, :, j] = np.array(np.load(f'data/simulations/post/observations_{j}.txt.npy'))
        plt.plot(total_time_range, observations[:observation_amount, i, j], color='blue')

    # Load the reference model observations.
    true_observation = np.array(np.load(f'data/simulations/true/observations_{1}.txt.npy'))

    # Create a mean vector of the posterior observations.
    observations_mean = np.mean(observations[:observation_amount, i, :], axis=1)

    # Plotting.
    plt.plot(total_time_range, true_observation[:observation_amount, i], color='k', label='Real Observation')
    plt.plot(total_time_range, observations_mean, color='y', linestyle='--', label='Mean Ensemble')
    plt.axvspan(0, training_time[-1], facecolor='g', alpha=0.1)
    plt.title(f'Observation {i + 1}.')
    plt.ylabel('|E_z|*10e14', fontsize=12)
    plt.xlabel('Time (years)', fontsize=12)
    plt.show()
