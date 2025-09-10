from typing import Tuple, List, Optional
import numpy as np
from scipy import linalg
from scipy.linalg import qr
from scipy.sparse.linalg import LinearOperator
import torch
from torch import _linalg_utils as _utils, Tensor
from torch.overrides import handle_torch_function, has_torch_function
import tensorly as tl
from tensorly.random import random_tucker
from tensorly import backend as T
from tensorly import unfold
from tensorly import fold
tl.set_backend('pytorch')

# Plotting and visualization
import matplotlib.pyplot as plt

from .utils import  Tensorization, compute_info_gain, GKS, extract_subtensor, plot_oed_histogram,  Matrixization, greedy2EIG

    

def process_modes(
    C: torch.Tensor,
    core_size: List[int],
    Pi: List[int],
    method: str,
    n_iter_max: int = 10,
    verbose: bool = False,
    tol: float = 1e-16,
) -> List[torch.Tensor]:
    """
    Processes all modes in the specified order during a Tucker-like decomposition method.

    Parameters
    ----------
    C : torch.Tensor
        The tensor being processed.
    core_size : List[int]
        Target sizes of the core tensor along each mode.
    Pi : List[int]
        The order in which the modes are processed.
    method : str
        The decomposition method to use ("IndSelect", "SeqSelect", or "HOOI").
    n_iter_max : int, optional (default: 100)
        Maximum number of iterations for iterative methods (e.g., HOOI).
    verbose : bool, optional (default: True)
        Whether to print convergence information during iterations.
    tol : float, optional (default: 1e-10)
        Tolerance for convergence in iterative methods.

    Returns
    -------
    S_list : List[torch.Tensor]
        List of selected indices for each mode, based on the specified method.
    """
    m = C.shape # m1 x ...x m_d x n  # core_size: k1 x ...x k_d 
   
    S_list = []

    if method == "IndSelect":
        # Higher-Order SVD: Select rows using GKS for each mode in Pi
        for mode in Pi:
            r = core_size[mode]
            Smode = GKS(tl.unfold(C, mode), k=r, cols=False)
            S_list.append(Smode)

    elif method == "SeqSelect":
        # Sequentially Truncated GKS: Update the tensor after processing each mode
        for mode in Pi:
            r = core_size[mode]
            shapeC = np.array(C.shape)  # Current shape of C

            Smode = GKS(tl.unfold(C, mode), k=r, cols=False)
            shapeC[mode] = r  # Update shape after truncation
            # Update C by folding the selected rows back into tensor format
            C = fold(tl.unfold(C, mode)[Smode, :], mode, shapeC) # TODO check 
            # print("Shape C", np.array(C.shape) )
            # print("mode",mode)
            # print("r",r)
            S_list.append(Smode)

    elif method == "IterSelect":
        # Higher-Order Orthogonal Iteration
        # Initialize set indices (k_j out of m_j )
        EIG_Slist = []  # Track information gain during iterations
        S_list = [np.random.choice(m[i], size=core_size[i], replace=False) for i in Pi]  # Another approach: S_list = [range(core_size[i]) for i in Pi] 
        # EIG initial selection 
        Csubtensor = extract_subtensor(C, S_list, Pi)
        unfoldedCst  =  Matrixization(Csubtensor) # same as  unfold(Y, mode=C.ndim - 1) # tensor to matrix 
        EIG = compute_info_gain(unfoldedCst) # Compute information gain for the unfolded tensor
        EIG_Slist.append(EIG)
        if verbose:
            print("Initial EIG:", EIG)
        
        for iteration in range(n_iter_max):

            for mode in Pi:
                r = core_size[mode]
                temp =  S_list.copy()
                S_list[mode] = np.arange(C.shape[mode])  # Select all rows for this mode
                Y = extract_subtensor(C, S_list, Pi)  # Subtensor 
                S_list[mode] = GKS(tl.unfold(Y, mode), k=r, cols=False) #TODO only update if improves EIG 
                #print("difference", np.linalg.norm(S_list[mode] -temp[mode] ))

            Csubtensor= extract_subtensor(C, S_list, Pi) # Extract subtensor with the updated indices
            # Unfold the tensor along the last mode
            unfoldedCst =  Matrixization(Csubtensor) # same as unfold(subtensor, mode=C.ndim - 1) # tensor to matrix 
            EIG = compute_info_gain(unfoldedCst) # Compute information gain for the unfolded tensor
            EIG_Slist.append(EIG)

            if verbose:
                print(
                        f"Iteration {iteration+1}: Current EIG={EIG} "
                    )
            # Check convergence
            if iteration > 0:
                if verbose:
                    print(
                      #  f"Iteration {iteration+1}: Current EIG={EIG_Slist[-1]}, "
                        f"Successive Variation={EIG_Slist[-2] - EIG_Slist[-1]}"
                    )
                if EIG_Slist[-1] - EIG_Slist[-2] < 0:
                    S_list = temp # use previous S
                    print(f"Warning: EIG decreased at iteration {iteration}. Stopping early.")
                    break    
                if tol and  (EIG_Slist[-1]- EIG_Slist[-2])/EIG_Slist[-2] <  tol:
                    if verbose:
                        print(f"Converged in {iteration} iterations.")
                    break
    else:
        raise ValueError(f"Method '{method}' is not supported. Please implement it.")

    return S_list



def TuckerOEDSelection( 
    tensor: torch.Tensor,
    core_size: List[int],
    method: str = "IndSelect",
    n_iter_max=100,
    Pi: Optional[List[int]] = None,
    greedy_approach: bool = False  # Added flag with default value False

) ->  List[List[int]]:
    """
        Sensor Selection based on Tucker-like formats 

        Parameters
        ----------
        tensor : torch.Tensor
            Arbitrarily dimensional tensor.
        core_size : list of int
            Target size of the core tensor along each mode.
        Pi : list of int, optional
            Order in which modes are processed. If None, the default order (0, 1, ..., N-1) is used.

        Returns
        -------
        C   : torch.Tensor
            Core tensor after truncation, same as core_size if dimensions are big enough for the truncated SVD.
        column_basis : List[torch.Tensor]
            List of (column) basis vectors for each mode.
             
        """
    inverse_Pi = None # inverse permutation

    # Set default processing order if none provided
    if Pi is None:
        Pi = range(len(tensor.shape)-1)  # Default order: 0, 1, ...
        
    else:  # Compute the inverse of the mode permutation (if Pi was provided)
        inverse_Pi = [0] * len(Pi)
        for i, pos in enumerate(Pi):
            inverse_Pi[pos] = i     

    C = tensor.clone()
    S_list = []

    # dictionary mapping methods to the corresponding names
    methods = {"IndSelect", "SeqSelect", "IterSelect"}
    # Check if the method is valid and then process the modes
    if method in methods:
        if greedy_approach:
            
            S_list = process_modes_greedy(C, core_size, Pi, method=method, n_iter_max=n_iter_max)
        else:
            S_list = process_modes(C, core_size, Pi, method=method, n_iter_max=n_iter_max)
    else:
        raise ValueError(f"Method {method} is not supported yet.")
  
     
    # Reordering of S_list     
    if inverse_Pi is None: # If Pi = 0 1 2 ...  
        return S_list
    
    else:  # Reorder singular values and vectors if a custom mode order was used. This can be done in a more efficient way. 
        S_list = [S_list[i] for i in inverse_Pi]     
        return S_list    




def ScketchThenTuckerOEDSelection(
    A: torch.Tensor,
    p: int,
    target_rank: List[int], 
    shapeX,
    method: str = "IndSelect",
    n_iter_max: int = 100,
    Pi: Optional[List[int]] = None,

) -> List[List[int]]:
    """
    Perform sketching on matrix A, tensorize the result, and call TuckerOEDSelection.

    Parameters:
    -----------
    A : torch.Tensor
        The input matrix to sketch.
    p : int
        Oversampling parameter for the sketching process.
    target_rank : List[int]
        The desired core size for the Tucker decomposition.
    method : str, optional
        The method to use in TuckerOEDSelection ("IndSelect", "SeqSelect", "IterSelect"), default is "IndSelect".
    n_iter_max : int, optional
        Maximum number of iterations for iterative methods, default is 100.
    Pi : List[int], optional
        Modes to consider for subsampling, default is None.


    Returns:
    --------
    List[List[int]]
        List of selected indices for each mode of the tensor.
    """


    # Step 1: Sketch the matrix A
    n =  A.shape[0]
    K = np.prod(target_rank)
    d2 = K + p
    mu, sigma = 0, 1/np.sqrt(d2) # mean and standard deviation
    sketchA = np.random.normal(mu, sigma, (d2, n)) @ A
    # Step 2: Tensorize the sketched matrix
    shape_sketchX  = list(shapeX)
    shape_sketchX[-1] = d2
    sketchX = Tensorization(sketchA ,shape= shape_sketchX)

    # Step 3: Call TuckerOEDSelection
    S_list = TuckerOEDSelection(tensor=sketchX , core_size=target_rank, method=method, n_iter_max=n_iter_max, Pi=Pi)

    return S_list
