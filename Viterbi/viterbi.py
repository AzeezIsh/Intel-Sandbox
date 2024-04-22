# %%
# import numpy as np
import array_api_strict as np
# import array_api_compat.numpy as np
def viterbi_np(obs, states, start_p, trans_p, emit_p):
    num_states = len(states)
    num_obs = len(obs)
    
    # Convert state indices to numbers if necessary
    state_map = {state: idx for idx, state in enumerate(states)}
    
    # Convert parameters to numpy arrays
    start_prob = np.asarray([start_p[state] for state in states])
    trans_prob = np.asarray([[trans_p[prev][curr] for curr in states] for prev in states])
    emit_prob = np.asarray([[emit_p[state][ob] for ob in obs] for state in states])
    
    # Initialize the Viterbi table
    V = np.zeros((num_states, num_obs))
    path = np.zeros((num_states, num_obs), dtype=np.int16)
    
    # Initial state probabilities
    V[:, 0] = start_prob * emit_prob[:, 0]
    
    # Dynamic programming
    for t in range(1, num_obs):
        for curr_state in range(num_states):
            # Max transition probability from any previous state
            trans_probs = V[:, t-1] * trans_prob[:, curr_state]
            # Highest transition probability
            best_prev_state = np.argmax(trans_probs)
            # Curr state max probability
            V[curr_state, t] = trans_probs[best_prev_state] * emit_prob[curr_state, t]
            # Store path to backtrack
            path[curr_state, t] = best_prev_state
    
    # Backtrack to find the optimal path
    optimal_path = np.zeros(num_obs, dtype=np.int16)
    # Backtrack from state with the highest probability
    optimal_path[-1] = np.argmax(V[:, -1])
    # Optimal for loop
    for t in range(num_obs-2, -1, -1):
        optimal_path[t] = path[optimal_path[t+1], t+1]
    
    # Indices back to state names if necessary
    optimal_path_states = [states[idx] for idx in optimal_path]
    
    return V, optimal_path_states


# %%
# States and Observations
states = ["LowPower", "HighPerformance"]
obs = ["Idle", "Moderate", "Intensive"]

# Starting Probabilities
start_p = {"LowPower": 0.7, "HighPerformance": 0.3}

# Transition Probabilities
trans_p = {
    "LowPower": {"LowPower": 0.8, "HighPerformance": 0.2},
    "HighPerformance": {"LowPower": 0.3, "HighPerformance": 0.7}
}

# Emission Probabilities
emit_p = {
    "LowPower": {"Idle": 0.6, "Moderate": 0.3, "Intensive": 0.1},
    "HighPerformance": {"Idle": 0.1, "Moderate": 0.4, "Intensive": 0.5}
}

# Now let's use the viterbi algorithm to decode the most likely states for a given sequence of observations
V, path = viterbi_np(obs, states, start_p, trans_p, emit_p)
print("Viterbi matrix:\n", V)
print("Optimal path through the states:", path)


# %%
states = ["IntelDominant", "NvidiaDominant"]
start_p = {"IntelDominant": 0.5, "NvidiaDominant": 0.5}
trans_p = {
    "IntelDominant": {"IntelDominant": 0.6, "NvidiaDominant": 0.4},
    "NvidiaDominant": {"IntelDominant": 0.4, "NvidiaDominant": 0.6}
}
emit_p = {
    "IntelDominant": {
        "AIandDeepLearning": 0.1, "HighPerformanceComputing": 0.25, "GraphicIntensive": 0.05,
        "DataAnalytics": 0.2, "VirtualReality": 0.1, "EnergyEfficientComputing": 0.15,
        "ScientificModeling": 0.2, "CryptocurrencyMining": 0.05
    },
    "NvidiaDominant": {
        "AIandDeepLearning": 0.25, "HighPerformanceComputing": 0.1, "GraphicIntensive": 0.25,
        "DataAnalytics": 0.05, "VirtualReality": 0.15, "EnergyEfficientComputing": 0.05,
        "ScientificModeling": 0.05, "CryptocurrencyMining": 0.2
    }
}
obs = ["DataAnalytics", "HighPerformanceComputing", "AIandDeepLearning", 
       "GraphicIntensive", "VirtualReality", "ScientificModeling", 
       "EnergyEfficientComputing", "CryptocurrencyMining"]
V, path = viterbi_np(obs, states, start_p, trans_p, emit_p)
print("Viterbi matrix:\n", V)
print("Optimal path through the states:", path)


# %%
def viterbi_np_verbose(obs, states, start_p, trans_p, emit_p):
    num_states = len(states)
    num_obs = len(obs)
    
    state_map = {state: idx for idx, state in enumerate(states)}
    
    start_prob = np.asarray([start_p[state] for state in states])
    trans_prob = np.asarray([[trans_p[prev][curr] for curr in states] for prev in states])
    emit_prob = np.asarray([[emit_p[state][ob] for ob in obs] for state in states])
    
    V = np.zeros((num_states, num_obs))
    path = np.zeros((num_states, num_obs), dtype=np.int16)
    
    V[:, 0] = start_prob * emit_prob[:, 0]
    print(f"Initialization: {obs[0]}")
    for state in states:
        print(f"P({state}|{obs[0]}) = start_p[{state}] * emit_p[{state}][{obs[0]}] = {start_prob[state_map[state]]} * {emit_prob[state_map[state], 0]} = {V[state_map[state], 0]}")
    
    for t in range(1, num_obs):
        print(f"\nStep {t}: {obs[t]}")
        for curr_state in range(num_states):
            trans_probs = V[:, t-1] * trans_prob[:, curr_state]
            best_prev_state = np.argmax(trans_probs)
            V[curr_state, t] = trans_probs[best_prev_state] * emit_prob[curr_state, t]
            path[curr_state, t] = best_prev_state
            print(f"P({states[curr_state]}|{obs[t]}) = max(P(prev_state)*trans_p[prev_state][{states[curr_state]}])*emit_p[{states[curr_state]}][{obs[t]}]")
            for prev_state in range(num_states):
                print(f"  P({states[prev_state]})*trans_p[{states[prev_state]}][{states[curr_state]}] = {V[prev_state, t-1]} * {trans_prob[prev_state, curr_state]} = {V[prev_state, t-1] * trans_prob[prev_state, curr_state]}")
            print(f"Selected max: {trans_probs[best_prev_state]} * {emit_prob[curr_state, t]} = {V[curr_state, t]} from {states[best_prev_state]}")
    
    optimal_path = np.zeros(num_obs, dtype=np.int16)
    optimal_path[-1] = np.argmax(V[:, -1])
    for t in range(num_obs-2, -1, -1):
        optimal_path[t] = path[optimal_path[t+1], t+1]
    
    optimal_path_states = [states[idx] for idx in optimal_path]
    
    return V, optimal_path_states

V, path = viterbi_np_verbose(obs, states, start_p, trans_p, emit_p)
print("\nViterbi matrix:\n", V)
print("Optimal path through the states:", path)


# %%
states = ["IntelDominant", "NvidiaDominant"]
start_p = {"IntelDominant": 0.5, "NvidiaDominant": 0.5}
trans_p = {
    "IntelDominant": {"IntelDominant": 0.6, "NvidiaDominant": 0.4},
    "NvidiaDominant": {"IntelDominant": 0.4, "NvidiaDominant": 0.6}
}
emit_p = {
    "IntelDominant": {
        "AIandDeepLearning": 0.1, "HighPerformanceComputing": 0.25, "GraphicIntensive": 0.05,
        "DataAnalytics": 0.2, "VirtualReality": 0.1, "EnergyEfficientComputing": 0.15,
        "ScientificModeling": 0.2, "CryptocurrencyMining": 0.05
    },
    "NvidiaDominant": {
        "AIandDeepLearning": 0.25, "HighPerformanceComputing": 0.1, "GraphicIntensive": 0.25,
        "DataAnalytics": 0.05, "VirtualReality": 0.15, "EnergyEfficientComputing": 0.05,
        "ScientificModeling": 0.05, "CryptocurrencyMining": 0.2
    }
}
obs = ["DataAnalytics", "HighPerformanceComputing", "AIandDeepLearning", 
       "GraphicIntensive", "VirtualReality", "ScientificModeling", 
       "EnergyEfficientComputing", "CryptocurrencyMining"]
V, path = viterbi_np_verbose(obs, states, start_p, trans_p, emit_p)
print("Viterbi matrix:\n", V)
print("Optimal path through the states:", path)


