from sensor_placement.utils  import Tensorization, Matrixization, extract_subtensor, compute_info_gain, plot_oed_histogram
from sensor_placement.decomposition import  TuckerOEDSelection
import time
import torch

def run_oed_methods(X, X1, target_rank, Pi, methods, plot=False, OED_values=None, k=None, bins=60, Scientificnot=False, decimal_places=4, loc= 'upper right', show_title=True, greedy_approach=False):
    """
    Runs multiple TuckerOEDSelection methods, computes information gain, 
    and optionally plots the results.

    Parameters:
    - X: Original tensor
    - X1: Processed tensor
    - target_rank: Target core size
    - Pi: A permutation or ordering of the modes
    - methods: List of methods to run (e.g., methods =  ["IndSelect", "SeqSelect", "IterSelect"])
    - plot: Whether to plot the results (default is False)
    - OED_values: Additional OED values for comparison (used for plotting)
    - k: Number of bins in the histogram (used for plotting)
    - bins: Histogram bins (used for plotting)
    
    Returns:
    - indices_dict: Dictionary containing OED values for each method
    """
    indices_dict = {}

    for method in methods:
        # Run TuckerOEDSelection for the given method
        if method == "IterSelect":
            S_list = TuckerOEDSelection(X1.clone(), core_size=target_rank, method=method, n_iter_max=7, Pi=Pi, greedy_approach=greedy_approach)
        else:
            S_list = TuckerOEDSelection(X1.clone(), core_size=target_rank, method=method, Pi=Pi, greedy_approach=greedy_approach)

        # Print selected indices for debugging
        print(f"Method: {method}")
        for mode in Pi:
            print("Selected Indices for mode:", mode)
            print(S_list[mode])

        # Extract subtensor and compute unfolded matrix
        subtensor = extract_subtensor(X, S_list, Pi)
        unfolded = Matrixization(subtensor)

        # Compute information gain and store in dictionary
        OED_value = compute_info_gain(unfolded)
        print("EIG:", OED_value )
        shortened_method = method.replace("Select", "")  # Remove "Select"
        indices_dict[f'{shortened_method}'] = OED_value

    # Optional: Plot results
    if plot:
        plot_oed_histogram(indices_dict, OED_values, k=k, bins=bins, Scientificnot = Scientificnot, decimal_places=decimal_places, loc=loc, show_title=show_title)

    return indices_dict

def benchmark_oed_methods(X1, target_rank, Pi, methods=["IndSelect", "SeqSelect", "IterSelect"], greedy_approach=False, n_runs=3):
    """
    Benchmarks the time to compute S_list for each method in TuckerOEDSelection.

    Parameters:
    - X1: The input tensor for TuckerOEDSelection
    - target_rank: Target core size
    - Pi: A permutation or ordering of the modes
    - methods: List of methods to benchmark (default = all 3)
    - greedy_approach: Whether to use greedy version
    - n_runs: Number of times to run each method (default: 3)

    Returns:
    - timing_dict: Dictionary mapping method names to average computation time (in seconds)
    """
    timing_dict = {}
    X1_copy = X1.clone()  # avoid in-place side effects

    for method in methods:
        times = []
        for _ in range(n_runs):
            
            start = time.time()
            if method == "IterSelect":
                _ = TuckerOEDSelection(X1_copy, core_size=target_rank, method=method, n_iter_max=7, Pi=Pi, greedy_approach=greedy_approach)
            else:
                _ = TuckerOEDSelection(X1_copy, core_size=target_rank, method=method, Pi=Pi, greedy_approach=greedy_approach)
            end = time.time()
            times.append(end - start)
        
        avg_time = sum(times) / n_runs
        shortened = method.replace("Select", "")
        timing_dict[shortened] = avg_time
        print(f"Avg time for {method}: {avg_time:.4f} seconds")

    return timing_dict


