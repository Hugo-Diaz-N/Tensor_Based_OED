import numpy as np
import scipy.sparse.linalg as spla
#from numba import njit
from typing import Tuple, List, Optional
from scipy.linalg import svdvals
from scipy.linalg import qr, svd, solve_triangular
import os
from brokenaxes import brokenaxes
from time import time
from scipy import linalg
from scipy.linalg import qr
from scipy.sparse.linalg import LinearOperator
import scipy.io
import scipy.sparse
import math
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


def load_tensor(file_name, sub_folder=None, sparse =False):
    """
    Loads a tensor from a file located in the current directory, a specified subfolder 
    of 'data', or the 'data' folder itself.

    Parameters:
        file_name (str): Name of the file to load (e.g., 'tensor.npy').
        sub_folder (str, optional): Name of a subfolder within 'data' to check first. Default is None.

    Returns:
        numpy.ndarray: The loaded tensor.

    Raises:
        FileNotFoundError: If the file is not found in any of the checked locations.
    """
    # Check in the current directory
    if os.path.isfile(file_name):
        print("File was found inside the working directory. ")
        return np.load(file_name)
    
    # Get the base 'data' directory (assumed to be at the same level as 'sensor_placement')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(os.path.dirname(current_dir), "data")

    # If a subfolder is specified, check there first
    if sub_folder:
        file_path_in_subfolder = os.path.join(data_folder, sub_folder, file_name)
        if os.path.isfile(file_path_in_subfolder):
            print("File was found inside the subfolder inside 'data'. ")
            if sparse:
                return scipy.sparse.load_npz(file_path_in_subfolder)
            return np.load(file_path_in_subfolder)
    
    # Check in the 'data' folder itself
    file_path_in_data = os.path.join(data_folder, file_name)
    if os.path.isfile(file_path_in_data):
        print("File was found inside 'data'. ")
        return np.load(file_path_in_data)

    # Raise an error if the file is not found
    raise FileNotFoundError(f"File '{file_name}' not found in the current directory, 'data/{sub_folder}' (if specified), or 'data'.")



def relative_error_t(T: torch.Tensor, T_N: torch.Tensor) -> float:
    """
    Compute the relative error between two tensors.
    
    Parameters
    ----------
    T : torch.Tensor
        Original tensor.
    T_N : torch.Tensor
        Reconstructed tensor.
        
    Returns
    -------
    float
        Relative error as ||T - T_N|| / ||T||. (Frobenius norm) 
    """
    return torch.norm(T - T_N).item() / torch.norm(T).item()


def create_random_TuckerTensor(
    tensor_size: Tuple[int, ...],
      core_size: Tuple[int, ...],
        dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """
    Generate a random Tucker tensor for testing.
    
    Parameters
    ----------
    tensor_size : Tuple[int, ...]
        Size of the tensor.
    core_size : Tuple[int, ...]
        Rank of the core tensor.
    dtype : torch.dtype
        Data type of the tensor.
        
    Returns
    -------
    torch.Tensor
        Randomly generated tensor.
    """
    return random_tucker(tensor_size, rank=core_size, full=True, dtype=dtype)



def compute_info_gain(A):
  """
  Computes logdet( I + AA^T)

  Args:
      A (numpy.ndarray): The matrix to compute the OED.

  Returns:
      float: The log determinant of (eye(A.shape[0]) + A.T @ A) or using the SVD.
  """
#   B = A.T @ A
#   _, log_det_k_columns = np.linalg.slogdet(np.eye(B.shape[0]) + B)
  
  log_det_k_columns = np.sum(np.log(1 + svdvals(A)**2)) 
  return log_det_k_columns




def greedyEIG(A, k): # greedy 
    """
    Finds k columns from matrix A that maximize the compute_info_gain function.
    
    Args:
        A (numpy.ndarray): Input matrix of shape (m, n).
        k (int): Number of columns to select.

    Returns:
        list: Indices of the selected columns.
        numpy.ndarray: Selected columns as a matrix.
    """
    m, n = A.shape
    selected_indices = []
    selected_columns = np.empty((m, 0))

    for _ in range(k):# greedy select k columns 
        best_gain = -np.inf
        best_column_index = -1 # TODO check this, numpy considers this the last column
        for i in range(n): # 0 to n-1 
            if i in selected_indices:
                continue
            candidate_columns = np.hstack((selected_columns, A[:, [i]]))
            gain = compute_info_gain(candidate_columns)
            if gain > best_gain:
                best_gain = gain
                best_column_index = i

        selected_indices.append(best_column_index)
        selected_columns = np.hstack((selected_columns, A[:, [best_column_index]]))

    return np.array(selected_indices), selected_columns, best_gain




def greedy2EIG(A, k, mode, shapeX): # greedy 
    """
    Finds k columns from matrix A that maximize the compute_info_gain function.
    
    Args:
        A (numpy.ndarray): Input matrix of shape (m, n). A = X_(last_dim)
        k (int): Number of columns to select.
        # m: mode 

    Returns:
        list: Indices of the selected columns.
        numpy.ndarray: Selected columns as a matrix.
    """
    m, n = A.shape
    selected_indices = []
    selected_columns = np.empty((m, 0))
    
    shapeX[mode] = k

    for s in range(k):# greedy select k columns 
        best_gain = -np.inf
        best_column_index = -1 # TODO check this, numpy considers this the last column
        for i in range(n): # 0 to n-1 
            if i in selected_indices:
                continue
            candidate_columns = np.hstack((A[:, [i]], selected_columns))
            Ncols = candidate_columns.shape[1]
            shapeX[mode] = Ncols
            X_refolded = fold(tl.tensor(candidate_columns.T), mode=mode, shape=shapeX)
            
            gain = compute_info_gain(Matrixization(X_refolded ))
            if gain > best_gain:
                best_gain = gain
                best_column_index = i

        selected_indices.append(best_column_index)
        selected_columns = np.hstack(( A[:, [best_column_index]], selected_columns,))

    return np.array(selected_indices), selected_columns, best_gain


def plot_oed_histogram(indices_dict, oed_values, k=0, bins=200, show_title=True, decimal_places=2, fontsize=35, save=False, Scientificnot = False, loc= 'upper right'):
    """
    Plots a well-formatted histogram of OED values, highlighting values from different methods.

    Args:
        indices_dict (dict): A dictionary where keys are method names (for legend)
                             and values are specific OED values.
        oed_values (list or array): The OED values to visualize.
        k (int, optional): Parameter for title customization. Defaults to 0.
        bins (int, optional): Number of bins for the histogram. Defaults to 300.
        show_title (bool, optional): Whether to display the title. Defaults to True.
        decimal_places (int, optional): Decimal places for displaying OED values. Defaults to 2.
        fontsize (int, optional): Font size for plot elements. Defaults to 30.
    """
    plt.figure(figsize=(14, 8))  # Set appropriate figure size

    #colors = ['brown','black', 'red', 'purple', 'blue', 'green']
    colormap = plt.cm.get_cmap('tab10',10)  # 'tab20' is a colormap with 20 distinct colors
    colors = [colormap(i) for i in range(colormap.N)]


    linestyles = ['-', ':', '--', '-.', '-']
    for idx, (method, value) in enumerate(indices_dict.items()):
        color = colors[idx % len(colors)]  # Cycle through colors
        linestyle = linestyles[idx % len(linestyles)]  # Cycle through line styles
        # Format the label based on Scientificnot flag
        if Scientificnot:
            formatted_value = f"{value:.{decimal_places}e}"  # Scientific notation
        else:
            formatted_value = f"{value:.{decimal_places}f}"  # Floating-point notation

        plt.axvline(
            x=value,
            color=color,
            linestyle=linestyle,
            linewidth=10,
            alpha=0.8,  # Set the opacity (0 to 1)
            label=rf'$\phi_{{EIG}}: {method}$' + rf'$\mathbf{{:{formatted_value}}}$',
        )

    # Plot OED values with density normalization
    plt.hist(oed_values, bins=bins, color='black', edgecolor='black', label='EIG Values', density=True, histtype='stepfilled', facecolor='grey', alpha=0.65)

    # Set x-axis limits between min and max of values
    #plt.xlim(min(oed_values), max(max(oed_values), *indices_dict.values()) + 1)

    # Customize plot elements for clarity
    plt.xlabel('EIG Value', fontsize=fontsize)
    plt.ylabel('Frequency Density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks([],fontsize=fontsize)
    if show_title:
        plt.title(f'Random Design vs Near Optimal Selections -- $k=$ {k}', fontsize=fontsize)
    plt.legend(loc=loc, shadow=True, fontsize=3 * fontsize // 4)  # Position legend for best visibility
    plt.tight_layout()
    if save:
        plt.savefig('histo_k_' + str(k) + '_2.eps', format='eps')
    plt.show()


def plot_histogram(indices_dict, oed_values, k=0, bins='auto', show_title=True, decimal_places=2, fontsize=30, save=False, loc= 'upper right'):
    """
    Plots a well-formatted histogram of OED values, highlighting values from different methods.

    Args:
        indices_dict (dict): A dictionary where keys are method names (for legend)
                             and values are specific OED values.
        oed_values (list or array): The OED values to visualize.
        k (int, optional): Parameter for title customization. Defaults to 0.
        bins (int, optional): Number of bins for the histogram. Defaults to 300.
        show_title (bool, optional): Whether to display the title. Defaults to True.
        decimal_places (int, optional): Decimal places for displaying OED values. Defaults to 2.
        fontsize (int, optional): Font size for plot elements. Defaults to 30.
    """
    plt.figure(figsize=(14, 8))  # Set appropriate figure size

    #colors = ['brown','black', 'red', 'purple', 'blue', 'green']
    colormap = plt.cm.get_cmap('tab10',10)  # 'tab20' is a colormap with 20 distinct colors
    colors = [colormap(i) for i in range(colormap.N)]


    linestyles = ['-', ':', '--', '-.', '-']
    for idx, (method, value) in enumerate(indices_dict.items()):
        color = colors[idx % len(colors)]  # Cycle through colors
        linestyle = linestyles[idx % len(linestyles)]  # Cycle through line styles
        plt.axvline(
            x=value,
            color=color,
            linestyle=linestyle,
            linewidth=10,
            alpha=0.8,  # Set the opacity (0 to 1)
#            label=f'$\phi_{{EIG}}: {method}$' + r'$\mathbf{' + ':' + f'{value: .{decimal_places}f}' + r'}$',
            label=rf'$\phi_{{EIG}}: {method}$' + r'$\mathbf{' + ':' + f'{float(value): .{decimal_places}f}' + r'}$',

        )

    # Plot OED values with density normalization
    plt.hist(oed_values, bins=bins, color='black', edgecolor='black', label='EIG Values', density=True, histtype='stepfilled', facecolor='grey', alpha=0.35)

    # Set x-axis limits between min and max of values
    plt.xlim(max(min(oed_values)-1,0), max(max(oed_values), *indices_dict.values()))

    # Customize plot elements for clarity
    plt.xlabel('Relative Errors', fontsize=fontsize)
    plt.ylabel('Frequency Density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks([],fontsize=fontsize)
    if show_title:
        plt.title(f'Random Design vs Near Optimal sensor selections -- $k=$ {k}', fontsize=fontsize)
    plt.legend(loc=loc, shadow=True, fontsize=3 * fontsize // 4)  # Position legend for best visibility
    plt.tight_layout()
    if save:
        plt.savefig('histo_k_' + str(k) + '_2.eps', format='eps')
    plt.show()


def plot_histogram_relError(indices_dict, oed_values, k=0, bins='auto', show_title=True, decimal_places=2, fontsize=30, save=False, loc='upper left'):
    """
    Plots a well-formatted histogram of OED values, highlighting values from different methods.

    Args:
        indices_dict (dict): A dictionary where keys are method names (for legend)
                             and values are specific OED values.
        oed_values (list or array): The OED values to visualize.
        k (int, optional): Parameter for title customization. Defaults to 0.
        bins (int, optional): Number of bins for the histogram. Defaults to 300.
        show_title (bool, optional): Whether to display the title. Defaults to True.
        decimal_places (int, optional): Decimal places for displaying OED values. Defaults to 2.
        fontsize (int, optional): Font size for plot elements. Defaults to 30.
    """
    plt.figure(figsize=(14, 8))  # Set appropriate figure size

    #colors = ['brown','black', 'red', 'purple', 'blue', 'green']
    colormap = plt.cm.get_cmap('tab10',10)  # 'tab20' is a colormap with 20 distinct colors
    colors = [colormap(i) for i in range(colormap.N)]


    linestyles = ['-', ':', '--', '-.', '-']
    for idx, (method, value) in enumerate(indices_dict.items()):
        color = colors[idx % len(colors)]  # Cycle through colors
        linestyle = linestyles[idx % len(linestyles)]  # Cycle through line styles
        plt.axvline(
            x=value,
            color=color,
            linestyle=linestyle,
            linewidth=10,
            alpha=0.8,  # Set the opacity (0 to 1)
#            label=f'$\phi_{{EIG}}: {method}$' + r'$\mathbf{' + ':' + f'{value: .{decimal_places}f}' + r'}$',
            label=rf'${method}$' + r'$\mathbf{' + ':' + f'{float(value): .{decimal_places}f}' + r'}$',

        )

    # Plot OED values with density normalization
    plt.hist(oed_values, bins=bins, color='black', edgecolor='black', label='Relative Errors', density=True, histtype='stepfilled', facecolor='grey', alpha=0.65)

    # Set x-axis limits between min and max of values
    plt.xlim(min(oed_values)/3, max(max(oed_values), *indices_dict.values()))

    # Customize plot elements for clarity
    plt.xlabel('Relative Errors', fontsize=fontsize)
    plt.ylabel('Frequency Density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks([],fontsize=fontsize)
    if show_title:
        plt.title(f'Random Design vs Near Optimal sensor selections -- $k=$ {k}', fontsize=fontsize)
    plt.legend(loc=loc, shadow=True, fontsize=3 * fontsize // 4)  # Position legend for best visibility
    plt.tight_layout()
    if save:
        plt.savefig('histo_k_' + str(k) + '_2.eps', format='eps')
    plt.show()



def GKS(X, k, cols=True):
    """
    Perform the Golub-Klema-Stewart (GKS) method using a truncated SVD and 
    QR decomposition with column pivoting.

    Parameters
    ----------
    X : ndarray or tensor
        The input matrix or tensor to decompose.
    k : int
        The number of leading components to extract.
    cols : bool, optional (default: True)
        If True, operates on the columns of `X`. If False, operates on the rows.

    Returns
    -------
    S : ndarray
        The indices of the first `k` pivoted columns or rows, depending on `cols`.
    """
    # Input validation
    assert isinstance(k, int) and k > 0, "k must be a positive integer."

    # Perform truncated SVD
    if cols:
        _, _, V = tl.tenalg.truncated_svd(X, n_eigenvecs=k)
        # QR decomposition with pivoting on the right singular vectors
        _, _, P = linalg.qr(V.mH, pivoting=True)
    else:  # rows
        U, _, _ = tl.tenalg.truncated_svd(X, n_eigenvecs=k)
        # QR decomposition with pivoting on the left singular vectors
        _, _, P = linalg.qr(U.mH, pivoting=True)

    # Select the first k pivoted indices
    S = P[:k]

    return S


def RGKS(X, k, cols=True): # TODO randomized version 
    """
    Perform the Golub-Klema-Stewart (GKS) method using a truncated SVD and 
    QR decomposition with column pivoting.

    Parameters
    ----------
    X : ndarray or tensor
        The input matrix or tensor to decompose.
    k : int
        The number of leading components to extract.
    cols : bool, optional (default: True)
        If True, operates on the columns of `X`. If False, operates on the rows.

    Returns
    -------
    S : ndarray
        The indices of the first `k` pivoted columns or rows, depending on `cols`.
    """
    # Input validation
    assert isinstance(k, int) and k > 0, "k must be a positive integer."

    # Perform truncated SVD
    if cols:
        _, _, V = tl.tenalg.truncated_svd(X, n_eigenvecs=k)
        # QR decomposition with pivoting on the right singular vectors
        _, _, P = linalg.qr(V.mH, pivoting=True)
    else:  # rows
        U, _, _ = tl.tenalg.truncated_svd(X, n_eigenvecs=k)
        # QR decomposition with pivoting on the left singular vectors
        _, _, P = linalg.qr(U.mH, pivoting=True)

    # Select the first k pivoted indices
    S = P[:k]

    return S



#@njit
def extract_subtensor(X, S_list, Pi):
    """
    Extracts a subtensor from a tensor X based on provided indices S_list and Phi.
    The modes specified in Phi are indexed by S_list in the same order.
    All other modes default to selecting the full range.

    Args:
        X (np.ndarray): Input tensor of shape (m1, m2, m3, ...).
        S_list (list of np.ndarray): List of index arrays for the modes in Phi.
        Phi (list of int): List of modes corresponding to S_list.

    Returns:
        np.ndarray: Extracted subtensor.
    """
    # Get the shape of the tensor
    shape = X.shape
    ndim = X.ndim

    # Initialize indices for all dimensions
    complete_indices = [np.arange(shape[mode]) for mode in range(ndim)]

    # Update indices for the specified modes
    for i in range(len(Pi)):
        complete_indices[Pi[i]] = np.sort(S_list[i])

    # Extract subtensor using multi-dimensional indexing
    return X[np.ix_(*complete_indices)]



def Matrixization(tensor, mode=None):
    """
    Unfolds (matrixizes) a tensor along a specified mode using TensorLy.

    Parameters
    ----------
    tensor : ndarray or Tensor
        The input tensor to be unfolded.
    mode : int, optional
        The mode along which to unfold the tensor. Defaults to the last mode.

    Returns
    -------
    unfolded : ndarray
        The unfolded tensor as a matrix.
    """
    if mode is None:
        mode = tl.ndim(tensor) - 1  # Default to the last mode
    return tl.unfold(tensor, mode)


def matrixize_numpy_array(np_array, mode=None):
    """
    Converts a NumPy array to a TensorLy tensor, applies matrixization (unfolding) 
    using the provided Matrixization function, and converts it back to a NumPy array.

    Parameters
    ----------
    np_array : ndarray
        The input NumPy array to be matrixized.
    mode : int, optional
        The mode along which to unfold the tensor. Defaults to the last mode.

    Returns
    -------
    unfolded_np_array : ndarray
        The unfolded tensor as a NumPy array.
    """
    # Convert NumPy array to TensorLy tensor
    tensor = tl.tensor(np_array)

    # Apply Matrixization (unfold the tensor using the given function)
    unfolded_tensor = Matrixization(tensor, mode)

    # Convert the unfolded tensor back to NumPy array
    unfolded_np_array = tl.to_numpy(unfolded_tensor)

    return unfolded_np_array


def Tensorization(unfolded, mode=None, shape=None):
    """
    Folds a matrix back into a tensor along the specified mode using TensorLy.
    By default, assumes the unfolding was performed along the last mode.

    Parameters
    ----------
    unfolded : ndarray
        The unfolded matrix to be folded back into a tensor.
    mode : int, optional
        The mode along which the tensor was unfolded. Defaults to the last mode.
    shape : tuple, optional
        The original shape of the tensor. Defaults to None, but must be provided.

    Returns
    -------
    folded : ndarray
        The reconstructed tensor.
    """
    if shape is None:
        raise ValueError("The 'shape' parameter must be provided to fold the tensor.")
    
    if mode is None:
        mode = len(shape) - 1  # Default to the last mode
    
    return tl.fold(tl.tensor(unfolded), mode, shape)



def solve_subtensor_system(
    Ft: np.ndarray, 
    btt: np.ndarray, 
    S_list: list, 
    Pi, 
    sigma2: float, 
    fact: float, 
    K: np.ndarray, 
    M: np.ndarray, 
    target_rank: tuple, 
    Mx: np.ndarray, 
    xt: np.ndarray, 
    maxiter: int = 300, 
    rtol: float = 1e-6
) -> tuple[np.ndarray, float]:
    """
    Solves a linear system derived from a subtensor extraction and transformation.

    Parameters
    ----------
    Ft : np.ndarray
        Full tensor to extract from.
    btt : np.ndarray
        Right-hand side tensor.
    S_list : list
        Subset indices for extraction.
    Pi : any
        Permutation or indexing object.
    sigma2 : float
        Scalar parameter for system definition.
    fact : float
        Factor scaling term.
    K : np.ndarray
        Matrix used in system construction.
    M : np.ndarray
        Matrix used in preconditioning and system.
    target_rank : tuple
        Target rank for reshaping.
    Mx : np.ndarray
        Preconditioner for conjugate gradient.
    xt : np.ndarray
        True solution for error calculation.
    maxiter : int, optional
        Maximum iterations for conjugate gradient (default: 300).
    rtol : float, optional
        Relative tolerance for conjugate gradient (default: 1e-6).

    Returns
    -------
    xs : np.ndarray
        Solution vector.
    rerr : float
        Relative error norm. If CG fails, returns -1.0.
    """

    # Extract and reshape Fs
    Fs = extract_subtensor(Ft, S_list, Pi)
    Fs = matrixize_numpy_array(Fs).T  # Equivalent to Matrixization(tl.tensor(Fs)).T

    # Extract and reshape bs
    bs = extract_subtensor(btt, S_list, Pi)
    bs = np.reshape(bs, (np.prod(target_rank), 1), order='C')  # Data generated with MATLAB

    # Define linear operator
    def Asv(v: np.ndarray) -> np.ndarray:
        return (Fs.T @ (Fs @ v)) / sigma2 + (fact / sigma2) * (K @ spla.spsolve(M, K @ v))

    AS = LinearOperator(M.shape, matvec=Asv, dtype=M.dtype)

    # Compute right-hand side
    Sb = (Fs.T @ bs) / sigma2
    Sb = Sb.ravel()

    # Solve using conjugate gradient
    xs, flag = spla.cg(AS, Sb, rtol=rtol, maxiter=maxiter, M=Mx)

    if flag == 0:
        # Compute relative error
        rerr = np.linalg.norm(xs - xt) / np.linalg.norm(xt)
    else:
        rerr = -1.0
        print(f"Warning: CG did not converge (flag={flag}). Consider increasing maxiter or adjusting parameters.")

    return xs, rerr


def greedyEIG(A, k): # greedy 
    """
    Finds k columns from matrix A that maximize the compute_info_gain function.
    
    Args:
        A (numpy.ndarray): Input matrix of shape (m, n).
        k (int): Number of columns to select.

    Returns:
        list: Indices of the selected columns.
        numpy.ndarray: Selected columns as a matrix.
    """
    m, n = A.shape
    selected_indices = []
    selected_columns = np.empty((m, 0))

    for _ in range(k):# greedy select k columns 
        best_gain = -np.inf
        best_column_index = -1 # TODO check this, numpy considers this the last column
        for i in range(n): # 0 to n-1 
            if i in selected_indices:
                continue
            candidate_columns = np.hstack((selected_columns, A[:, [i]]))
            gain = compute_info_gain(candidate_columns)
            if gain > best_gain:
                best_gain = gain
                best_column_index = i

        selected_indices.append(best_column_index)
        selected_columns = np.hstack((selected_columns, A[:, [best_column_index]]))

    return np.array(selected_indices), selected_columns, best_gain

def create_original_columns(Pi_I, nt, ns, k):
    """
    Create original columns using broadcasting.

    Args:
        Pi_I (numpy.ndarray): Array containing the selected columns.
        nt (int): Number of time steps.
        ns (int): Number of columns.
        k (int): Number of selected columns.

    Returns:
        numpy.ndarray: Array of original columns.
    """
    cols_original = np.zeros((k*nt), dtype = int) # number of selected sensors * number of snapshots

    for i in range(nt):
        cols_original[i*k:(i+1)*k] = Pi_I + i*ns
    
    return cols_original


### This are tailored functions for paper's plots and slides (remove)

# def plot_histogram4DVAR(indices_dict, oed_values, k=0, bins='auto', show_title=True, decimal_places=2, fontsize=30, save=False, loc= 'upper right'):
#     """
#     Plots a well-formatted histogram of OED values, highlighting values from different methods.

#     Args:
#         indices_dict (dict): A dictionary where keys are method names (for legend)
#                              and values are specific OED values.
#         oed_values (list or array): The OED values to visualize.
#         k (int, optional): Parameter for title customization. Defaults to 0.
#         bins (int, optional): Number of bins for the histogram. Defaults to 300.
#         show_title (bool, optional): Whether to display the title. Defaults to True.
#         decimal_places (int, optional): Decimal places for displaying OED values. Defaults to 2.
#         fontsize (int, optional): Font size for plot elements. Defaults to 30.
#     """
#     plt.figure(figsize=(14, 8))  # Set appropriate figure size

#     #colors = ['brown','black', 'red', 'purple', 'blue', 'green']
#     #colormap = plt.cm.get_cmap('tab10',10)  # 'tab20' is a colormap with 20 distinct colors
#     #colors = [colormap(i) for i in range(colormap.N - 1, -1, -1)] # range(colormap.N)
#     colors = ['brown','black', 'red', 'purple', 'blue', 'green']

#     linestyles = ['-', ':', '--', '-.', '-']
#     for idx, (method, value) in enumerate(indices_dict.items()):
#         color = colors[idx % len(colors)]  # Cycle through colors
#         linestyle = linestyles[idx % len(linestyles)]  # Cycle through line styles
#         plt.axvline(
#             x=value,
#             color=color,
#             linestyle=linestyle,
#             linewidth=10,
#             alpha=0.8,  # Set the opacity (0 to 1)
# #            label=f'$\phi_{{EIG}}: {method}$' + r'$\mathbf{' + ':' + f'{value: .{decimal_places}f}' + r'}$',
#             label=rf'$\phi_{{EIG}}: {method}$' + r'$\mathbf{' + ':' + f'{float(value): .{decimal_places}f}' + r'}$',

#         )

#     # Plot OED values with density normalization
#     plt.hist(oed_values, bins=bins, color='black', edgecolor='black', label='EIG Values', density=True, histtype='stepfilled', facecolor='grey', alpha=0.65)

#     # Set x-axis limits between min and max of values
#     plt.xlim(300,302)


#     # Customize plot elements for clarity
#     plt.xlabel('EIG Value', fontsize=fontsize)
#     plt.ylabel('Frequency Density', fontsize=fontsize)
#     plt.xticks(np.array([300,301,302]),fontsize=fontsize)
#     plt.yticks([],fontsize=fontsize)
#     if show_title:
#         plt.title(f'Random Design vs Near Optimal sensor selections -- $k=$ {k}', fontsize=fontsize)
#     plt.legend(loc=loc, shadow=True, fontsize=3 * fontsize // 4)  # Position legend for best visibility
#     plt.tight_layout()
#     if save:
#         plt.savefig('histo_k_' + str(k) + '_2.eps', format='eps')
#     plt.show()


# def plot_histogram4DVAR(indices_dict, oed_values, k=0, bins='auto', show_title=True, decimal_places=2, fontsize=30, save=False, loc='upper right'):
#     """
#     Plots a well-formatted histogram of OED values, highlighting values from different methods.

#     Args:
#         indices_dict (dict): A dictionary where keys are method names and values are OED values.
#         oed_values (list or array): The OED values to visualize.
#         k (int): Parameter for title customization.
#         bins (int or str): Number of bins or bin strategy.
#         show_title (bool): Whether to show title.
#         decimal_places (int): Precision for float display in legend.
#         fontsize (int): Font size for labels.
#         save (bool): If True, saves the figure.
#         loc (str): Legend location.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     plt.figure(figsize=(14, 8))

#     # Fixed lists for styling
#     colors = ['red', 'black', 'teal', 'purple', 'blue', 'green']
#     linestyles = ['-', '-', '-.', ':', '-']
#     linewidths = [15, 8, 10, 12, 15, 10]  # Prescribed line widths
#     linewidths = [5, 4, 5, 5, 5, 5] 

#     for idx, (method, value) in enumerate(indices_dict.items()):
#         color = colors[idx % len(colors)]
#         linestyle = linestyles[idx % len(linestyles)]
#         linewidth = linewidths[idx % len(linewidths)]
#         plt.axvline(
#             x=value,
#             color=color,
#             linestyle=linestyle,
#             linewidth=linewidth,
#             alpha=0.8,
#             label=rf'$\phi_{{EIG}}: {method}$' + r'$\mathbf{' + ':' + f'{float(value): .{decimal_places}f}' + r'}$'
#         )

#     # Histogram of OED values
#     plt.hist(
#         oed_values,
#         bins=bins,
#         color='black',
#         edgecolor='black',
#         label='EIG Values',
#         density=True,
#         histtype='stepfilled',
#         facecolor='grey',
#         alpha=0.65
#     )

#     plt.xlim(234, 248)
#     plt.xlabel('EIG Value', fontsize=fontsize)
#     plt.ylabel('Frequency Density', fontsize=fontsize)
#     plt.xticks(np.array([177, 242, 248]), fontsize=fontsize)
#     plt.yticks([], fontsize=fontsize)

#     if show_title:
#         plt.title(f'Random Design vs Near Optimal sensor selections -- $k=$ {k}', fontsize=fontsize)

#     plt.legend(loc=loc, shadow=True, fontsize=3 * fontsize // 4)
#     plt.tight_layout()
#     if save:
#         plt.savefig(f'histo_k_{k}_2.eps', format='eps')
#     plt.show()


# broken axis # k=5
def plot_histogram4DVAR(indices_dict, oed_values, k=0, bins='auto', show_title=True,
                         decimal_places=2, fontsize=30, save=False, loc='upper right'):

    xlims = [(178, 187), (234, 247)]
    fig = plt.figure(figsize=(14, 8))
    bax = brokenaxes(xlims=xlims, hspace=0.5, fig=fig)

    # Vertical lines for selected values
    colors = ['red', 'black', 'teal', 'purple', 'blue', 'green']
    linestyles = ['-', '-', '-.', ':', '-']
    linewidths = [15, 8, 10, 12, 15, 10]

    for idx, (method, value) in enumerate(indices_dict.items()):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        linewidth = linewidths[idx % len(linewidths)]
        bax.axvline(
            x=value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8,
            label=rf'$\phi_{{EIG}}: {method}$' + r'$\mathbf{' + ':' + f'{float(value): .{decimal_places}f}' + r'}$'
        )

    # Plot histogram
    bax.hist(oed_values, bins=bins, color='grey', edgecolor='black',
             label='EIG Values', density=True, alpha=0.65,
             histtype='stepfilled', facecolor='grey')

    bax.set_xlabel('EIG Value', fontsize=fontsize)
    bax.set_ylabel('Frequency Density', fontsize=fontsize)

    # Custom xticks for each axis
    bax.axs[0].set_xticks([ 180,  185])
    bax.axs[1].set_xticks([ 242,  246])
    
    bax.axs[0].tick_params(axis='x', labelsize=fontsize)
    bax.axs[1].tick_params(axis='x', labelsize=fontsize)

    # Optionally hide y-axis ticks
    for ax in bax.axs:
        ax.set_yticks([])
        ax.tick_params(axis='y', labelsize=fontsize)

    # Manually draw a unified top and right border across both axes
    fig.canvas.draw()

    left_ax_pos = bax.axs[0].get_position()
    right_ax_pos = bax.axs[1].get_position()

    # Top line across both axes
    fig.lines.append(plt.Line2D(
        [left_ax_pos.x0, right_ax_pos.x1],
        [left_ax_pos.y1, right_ax_pos.y1],
        transform=fig.transFigure,
        color='black',
        linewidth=1.5
    ))

    # Right vertical line on far-right
    fig.lines.append(plt.Line2D(
        [right_ax_pos.x1, right_ax_pos.x1],
        [right_ax_pos.y0, right_ax_pos.y1],
        transform=fig.transFigure,
        color='black',
        linewidth=1.5
    ))    

    if show_title:
        bax.set_title(f'Random Design vs Near Optimal sensor selections -- $k=$ {k}', fontsize=fontsize)

    bax.legend(loc=loc, shadow=True, fontsize=3 * fontsize // 4)

    if save:
        fig.savefig(f'histo_k_{k}_broken.eps', format='eps')

    plt.show()


def plot_histogram4DVAR(indices_dict, oed_values, k=0, bins='auto', show_title=True,
                         decimal_places=2, fontsize=30, save=False, loc='upper right'):

    xlims = [(300, 302)]  # Only one interval now
    fig = plt.figure(figsize=(14, 8))
    bax = brokenaxes(xlims=xlims, hspace=0.5, fig=fig)  # small \

    # Vertical lines for selected methods
    colors = ['red', 'black', 'teal', 'purple', 'blue', 'green']
    linestyles = ['-', '-', '-.', ':', '-']
    linewidths = [15, 8, 10, 12, 10, 10]

    for idx, (method, value) in enumerate(indices_dict.items()):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        linewidth = linewidths[idx % len(linewidths)]
        bax.axvline(
            x=value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8,
            label=rf'$\phi_{{EIG}}: {method}$' + r'$\mathbf{' + ':' + f'{float(value): .{decimal_places}f}' + r'}$'
        )

    # Plot histogram
    bax.hist(oed_values, bins=bins, color='grey', edgecolor='black',
             label='EIG Values', density=True, alpha=0.65,
             histtype='stepfilled', facecolor='grey')

    # Set labels
    bax.set_xlabel('EIG Value', fontsize=fontsize)
    bax.set_ylabel('Frequency Density', fontsize=fontsize)

    # Set xticks
    bax.axs[0].set_xticks([300, 302])
    bax.axs[0].tick_params(axis='x', labelsize=fontsize)

    # Hide y-axis ticks
    for ax in bax.axs:
        ax.set_yticks([])
        ax.tick_params(axis='y', labelsize=fontsize)

    # Draw borders manually
    fig.canvas.draw()
    ax_pos = bax.axs[0].get_position()

    # Top line
    fig.lines.append(plt.Line2D(
        [ax_pos.x0, ax_pos.x1],
        [ax_pos.y1, ax_pos.y1],
        transform=fig.transFigure,
        color='black',
        linewidth=1.5
    ))

    # Right vertical line
    fig.lines.append(plt.Line2D(
        [ax_pos.x1, ax_pos.x1],
        [ax_pos.y0, ax_pos.y1],
        transform=fig.transFigure,
        color='black',
        linewidth=1.5
    ))

    if show_title:
        bax.set_title(f'Random Design vs Near Optimal sensor selections -- $k=$ {k}', fontsize=fontsize)

    bax.legend(loc=loc, shadow=True, fontsize=3 * fontsize // 4)

    if save:
        fig.savefig(f'histo_k_{k}_broken.eps', format='eps')

    plt.show()
