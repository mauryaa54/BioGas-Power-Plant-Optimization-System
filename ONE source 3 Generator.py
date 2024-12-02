
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Constants
C_t = 1.0  # Cost per unit of time, fixed operational cost
lambda_penalty = 0.001  # Penalty scaling factor
H_min = 660  # Minimum height in m³ (or 0.2 for normalized)
H_max = 6000  # Maximum height in m³ (or 0.9 for normalized)
H_low = 2000  # Lower boundary for no penalty region
H_high = 4760  # Upper boundary for no penalty region
deltaH_source = 50  # Gas input per cycle (15 minutes, scaled down from 200 m³/h)
starting_cost = 30  # Cost to start each generator
generator_capacity = [400, 400, 380]  # Capacities of the three generators in kW

# Power generation per m³ of gas at 100% load
kWh_per_m3 = 1.7  # 1.7 kWh of electricity generated per m³ of gas

# Read CSV file
data = pd.read_csv('Project_B_Sample_Data_copy.csv')

# Forward fill the initial height across all cycles
data['Height'] = data['Initial_Height'].ffill()

# Number of time intervals
num_intervals = len(data)

# Efficiency values provided by the manufacturer
load_factors = np.array([1.0, 0.75, 0.5]).reshape(-1, 1)  # 100%, 75%, 50% load
efficiencies = np.array([0.395, 0.382, 0.358])  # Corresponding efficiencies

# Fit a polynomial regression model
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(load_factors, efficiencies)

# Function to determine efficiency based on load factor
def calculate_efficiency(load_factor):
    if load_factor < 0.5:
        return 0  # Generator should not operate if load < 50%
    elif load_factor >= 1.0:
        return poly_model.predict([[1.0]])[0]  # Use 100% efficiency for load >= 1.0
    else:
        return poly_model.predict([[load_factor]])[0]

# Penalty function
def penalty_smooth(H, H_min, H_max, H_low, H_high, A):
    if H < H_low:
        return A * (H_low - H)**2
    elif H_low <= H <= H_high:
        return 0  # No penalty in the optimal range
    elif H > H_high:
        return A * (H - H_high)**2
    else:
        return 0 

# Fitness function to evaluate the total cost of a power generation profile
def fitness_function(profile):
    n_particles = profile.shape[0]
    total_costs = np.zeros(n_particles)
    
    for p in range(n_particles):
        total_cost = 0
        height = data.loc[0, 'Height']
        
        for i in range(num_intervals):
            market_price = data.at[i, 'Market_Price']  # €/MWh
            power_generated = 0
            total_gas_consumed = 0
            cycle_cost = C_t
            
            for gen in range(len(generator_capacity)):
                load_factor = profile[p, i] / 1.0  # Assuming load factor profile applies to all generators
                efficiency = calculate_efficiency(load_factor)
                generator_on = height > H_min and 0.5 <= load_factor <= 1.0
                
                if generator_on:
                    power_needed = load_factor * 0.25 * generator_capacity[gen]  # Power in kW for 15 minutes
                    gas_consumed = power_needed / (kWh_per_m3 * efficiency)  # Gas consumed in m³
                    total_gas_consumed += gas_consumed
                    power_generated += gas_consumed * kWh_per_m3  # Power generated in kWh
                    if i == 0 or profile[p, i - 1] == 0:
                        cycle_cost += starting_cost  # Apply starting cost if the generator is turned on

            height -= total_gas_consumed
            height += deltaH_source
            height = np.clip(height, 0, H_max)

            penalty = penalty_smooth(height, H_min, H_max, H_low, H_high, A=lambda_penalty)
            revenue = market_price * power_generated
            total_cost += penalty + cycle_cost - revenue
        
        total_costs[p] = total_cost
    return total_costs


# Define bounds for each time interval
lb = [0.2] * num_intervals
ub = [0.9] * num_intervals
bounds = (lb, ub)

# Use PSO to minimize the fitness function
options = {'c1': 1.5, 'c2': 2, 'w': 0.4}
optimizer = ps.single.GlobalBestPSO(n_particles=1000, dimensions=num_intervals, options=options, bounds=bounds)  # we can change the number of particles here to num_intervals*2
best_cost, optimal_profile = optimizer.optimize(fitness_function, iters=150)

# # Debug: Print shape of optimal_profile
# print(f"Shape of optimal_profile before conversion: {np.shape(optimal_profile)}")

# Ensure optimal_profile is treated as a numpy array and has the correct shape
optimal_profile = np.array(optimal_profile)

# # Debug: Print shape of optimal_profile after conversion
# print(f"Shape of optimal_profile after conversion: {optimal_profile.shape}")

# Check if the shape is correct, otherwise print an error message
if optimal_profile.shape[0] != num_intervals:
    print(f"Error: optimal_profile shape is incorrect. Expected {num_intervals}, got {optimal_profile.shape[0]}")
else:
    # Apply the optimal power generation profile to compute the final results
    power_generated_list = []
    penalty_list = []
    generator_on_list = []
    total_cost_list = []
    height_list = []
    load_factor_list = []
    individual_power_generated_list = []

    height = data.loc[0, 'Height']
    for i in range(len(data)):
        height_list.append(height)
        market_price = data.at[i, 'Market_Price']
        power_generation_rate = optimal_profile[i]
        total_gas_consumed = 0
        total_power_generated = 0
        cycle_cost = C_t
        generator_on_str = []
        individual_power_generated = []

        for gen in range(len(generator_capacity)):
            load_factor = power_generation_rate / 1.0
            efficiency = calculate_efficiency(load_factor)
            generator_on = height > H_min and 0.5 <= load_factor <= 1.0

            if generator_on:
                power_needed = load_factor * 0.25 * generator_capacity[gen]
                gas_consumed = power_needed / (kWh_per_m3 * efficiency)
                total_gas_consumed += gas_consumed
                power_generated = gas_consumed * kWh_per_m3
                total_power_generated += power_generated
                individual_power_generated.append(power_generated)
                if i == 0 or not generator_on_list[-1] == 'Yes':
                    cycle_cost += starting_cost
                generator_on_str.append('Yes')
            else:
                individual_power_generated.append(0)
                generator_on_str.append('No')

        individual_power_generated_list.append(', '.join(map(str, individual_power_generated)))

        penalty = penalty_smooth(height, H_min, H_max, H_low, H_high, A=lambda_penalty)
        revenue = market_price * total_power_generated
        total_cycle_cost = penalty + cycle_cost - revenue
        total_cost_list.append(total_cycle_cost)
        load_factor_list.append(load_factor)

        height -= total_gas_consumed
        height += deltaH_source
        height = np.clip(height, 0, H_max)

        

        data.at[i, 'Height'] = height
        power_generated_list.append(total_power_generated)
        penalty_list.append(penalty)
        generator_on_list.append(', '.join(generator_on_str))

    data['Height'] = height_list
    data['Power_Generated'] = power_generated_list
    data['Individual_Power_Generated'] = individual_power_generated_list
    data['Penalty'] = penalty_list
    data['Generator_On'] = generator_on_list
    data['Total_Cost'] = total_cost_list
    data['Load_Factor'] = load_factor_list

    # Print the sum of all total costs
    print(f"Sum of all total costs: {sum(total_cost_list)}")
    
    # Plot Cost vs Optimization Iterations
    plt.figure(figsize=(12, 8))
    plot_cost_history(cost_history=optimizer.cost_history)
    plt.title('Cost vs Optimization Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.savefig('Cost_vs_Optimization_Iterations.png')
    plt.show()
    
    # Visualization of Particle Convergence
    # plt.figure(figsize=(12, 8))
    # for pos in optimizer.pos_history:
    #     plt.plot(np.mean(pos, axis=1), marker='o', markersize=3, linestyle='-', alpha=0.5)
    
    # plt.title('Particle Positions Over Iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Particle Position (mean)')
    # plt.grid(True)
    # plt.savefig('Particle_Positions_Over_Iterations.png')
    # plt.show()
    # # Histogram of Particle Positions
    # plt.figure(figsize=(12, 8))
    # for i in range(0, len(optimizer.pos_history), len(optimizer.pos_history) // 5):
    #     plt.hist(optimizer.pos_history[i].flatten(), bins=50, alpha=0.5, label=f'Iteration {i}')
    # plt.title('Histogram of Particle Positions Across Iterations')
    # plt.xlabel('Particle Position')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # 2D Scatter Plot of Particles (for 2D problems)
    # plt.figure(figsize=(12, 8))
    # for i in range(len(optimizer.pos_history)):
    #     plt.scatter(optimizer.pos_history[i][:, 0], optimizer.pos_history[i][:, 1], alpha=0.3, label=f'Iteration {i}' if i % (len(optimizer.pos_history) // 5) == 0 else "")
    # plt.title('Particle Movement Across Iterations')
    # plt.xlabel('Position Dimension 1')
    # plt.ylabel('Position Dimension 2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # Particle Position Heatmap
    # particle_positions_mean = np.mean(np.array(optimizer.pos_history), axis=2)  # Mean position across particles
    # plt.figure(figsize=(12, 8))
    # plt.imshow(particle_positions_mean.T, aspect='auto', cmap='viridis')
    # plt.colorbar(label='Mean Particle Position')
    # plt.title('Heatmap of Particle Position Mean Across Iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Dimension')
    # plt.show()
    
    # Line Plot of Particle Spread Over Iterations
    particle_spread = [np.std(pos) for pos in optimizer.pos_history]
    plt.figure(figsize=(12, 8))
    plt.plot(particle_spread, marker='o', linestyle='-', alpha=0.7)
    plt.title('Particle Spread (Standard Deviation) Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Particle Spread (Std Dev)')
    plt.grid(True)
    plt.savefig('Particle_Spread_Over_Iterations.png')
    plt.show()

    # Visualization of Height and Total Cost
    plt.figure(figsize=(12, 8))
    plt.scatter(data['Height'], data['Penalty'], color='red')
    plt.title('Relationship between Height and Penality')
    plt.xlabel('Tank Height (units)')
    plt.ylabel('Penalty')
    plt.grid(True)
    plt.savefig('Relationship_between_Height_and_Penality.png')
    plt.show()

    # Displaying cycle details in tabular format
    print(tabulate(data[['Cycle', 'Market_Price', 'Height', 'Load_Factor', 'Power_Generated', 'Penalty', 'Generator_On', 'Individual_Power_Generated', 'Total_Cost']],
                   headers='keys', tablefmt='psql', showindex=False))