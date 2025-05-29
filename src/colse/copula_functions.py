import numpy as np
import torch
from colse.copula_types import CopulaTypes, ArchCopulaTypes
from scipy.stats import kendalltau



copula_type_mapper = {
    CopulaTypes.GUMBEL: ArchCopulaTypes.GUMBEL,
    CopulaTypes.FRANK: ArchCopulaTypes.FRANK,
    CopulaTypes.CLAYTON: ArchCopulaTypes.CLAYTON,
}


def gumbel_copula(u, v, theta):
    part1 = (-np.log(u)) ** theta
    part2 = (-np.log(v)) ** theta
    copula_value = np.exp(-((part1 + part2) ** (1 / theta)))
    return copula_value

def rotated_copula_90(u, v, theta, function):
    copula_value = v - function(1-u, v, theta)
    return copula_value

def rotated_copula_180(u, v, theta, function):
    copula_value = u + v - 1 + function(1-u, 1-v, theta)
    return copula_value

def rotated_copula_270(u, v, theta, function):
    copula_value = u - function(u, 1-v, theta)
    return copula_value

def get_gumbel_copula(copula_type: ArchCopulaTypes, cdf1, cdf2, theta):
    match copula_type:
        case ArchCopulaTypes.GUMBEL:
            return gumbel_copula(cdf1, cdf2, theta)
        case ArchCopulaTypes.GUMBEL_90:
            return rotated_copula_90(cdf1, cdf2, theta, gumbel_copula)
        case ArchCopulaTypes.GUMBEL_180:
            return rotated_copula_180(cdf1, cdf2, theta, gumbel_copula)
        case ArchCopulaTypes.GUMBEL_270:
            return rotated_copula_270(cdf1, cdf2, theta, gumbel_copula)

def get_clayton_copula(copula_type: ArchCopulaTypes, cdf1, cdf2, theta):
    match copula_type:
        case ArchCopulaTypes.CLAYTON:
            return clayton_copula(cdf1, cdf2, theta)
        case ArchCopulaTypes.CLAYTON_90:
            return rotated_copula_90(cdf1, cdf2, theta, clayton_copula)
        case ArchCopulaTypes.CLAYTON_180:
            return rotated_copula_180(cdf1, cdf2, theta, clayton_copula)
        case ArchCopulaTypes.CLAYTON_270:
            return rotated_copula_270(cdf1, cdf2, theta, clayton_copula)
        

def gumbel_grad(u, v, theta, copula=None):
    if copula is None:
        copula = get_copula(CopulaTypes.GUMBEL, u, v, theta)
    
    part_1 = copula * (1/v) * (-np.log(v)) ** (theta - 1) 
    part_2 = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1/theta - 1)

    return part_1 * part_2


def gumbel_copula_torch(u, v, theta):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    epsilon = 1e-5  # Small epsilon value for numerical stability
    u = torch.clamp(u, min=epsilon, max=1 - epsilon)
    v = torch.clamp(v, min=epsilon, max=1 - epsilon)
    part1 = (-torch.log(u)) ** theta
    part2 = (-torch.log(v)) ** theta
    output = -((part1 + part2) ** (1.0 / theta))
    capped_input = torch.clamp(output, max=100.0)  # Cap large values
    copula_value = torch.exp(capped_input)
    return copula_value

def clayton_copula_torch(u, v, theta):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    copula_value = torch.maximum(u ** (-theta) + v ** (-theta) - 1, torch.zeros(1)) ** (-1 / theta)
    return copula_value


def frank_copula_torch(u, v, theta):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32)
    part1 = torch.exp(-theta * u) - 1
    part2 = torch.exp(-theta * v) - 1
    part3 = torch.exp(-theta) - 1
    copula_value = -1 / theta * torch.log(1 + (part1 * part2) / part3)
    return copula_value


def get_copula_torch(copula_type: CopulaTypes, cdf1, cdf2, theta):
    match copula_type:
        case CopulaTypes.CLAYTON:
            return clayton_copula_torch(cdf1, cdf2, theta)
        case CopulaTypes.GUMBEL:
            return gumbel_copula_torch(cdf1, cdf2, theta)
        case CopulaTypes.FRANK:
            return frank_copula_torch(cdf1, cdf2, theta)


def clayton_copula(u, v, theta):
    u = np.asarray(u)
    v = np.asarray(v)
    copula_value = np.maximum(u ** (-theta) + v ** (-theta) - 1, 0) ** (-1 / theta)
    return copula_value


def frank_copula(u, v, theta):
    u = np.asarray(u)
    v = np.asarray(v)
    part1 = np.exp(-theta * u) - 1
    part2 = np.exp(-theta * v) - 1
    part3 = np.exp(-theta) - 1
    copula_value = -1 / theta * np.log(1 + (part1 * part2) / part3)
    return copula_value


def get_copula_torch(copula_type: CopulaTypes, cdf1, cdf2, theta, tolerance=1e-5):
    if isinstance(theta, float) or (isinstance(theta, torch.Tensor) and theta.shape == torch.Size([])):
        batch_size = cdf1.shape[0]
        theta = torch.tensor([theta] * batch_size, dtype=torch.float32)
    # Handle cases where cdf1 or cdf2 are below tolerance
    mask_low = (cdf1 < tolerance) | (cdf2 < tolerance)
    
    # Handle cases where both cdf1 and cdf2 are above 1 - tolerance
    mask_high_both = (cdf1 > (1 - tolerance)) & (cdf2 > (1 - tolerance))
    
    # Handle cases where only cdf1 or only cdf2 is above 1 - tolerance
    mask_high_cdf1 = (cdf1 > (1 - tolerance)) & ~(cdf2 > (1 - tolerance))
    mask_high_cdf2 = (cdf2 > (1 - tolerance)) & ~(cdf1 > (1 - tolerance))

    # Initialize the result tensor with zeros where necessary
    result = torch.where(
        mask_low, 
        torch.tensor(0.0, dtype=torch.float32, requires_grad=True), 
        torch.zeros_like(cdf1, dtype=torch.float32, requires_grad=True)
    )

    result = torch.where(
        mask_high_both, 
        torch.tensor(1.0, dtype=torch.float32, requires_grad=True), 
        result
    )

    result = torch.where(
        mask_high_cdf1, 
        cdf2.to(torch.float32).requires_grad_(), 
        result
    )

    result = torch.where(
        mask_high_cdf2, 
        cdf1.to(torch.float32).requires_grad_(), 
        result
    )
    
    # Handle the remaining elements for copula computations
    remaining_mask = ~(mask_low | mask_high_both | mask_high_cdf1 | mask_high_cdf2)
    
    copula_result = torch.zeros_like(result, dtype=torch.float32)
    if remaining_mask.any():
        if copula_type == CopulaTypes.CLAYTON:
            copula_values = clayton_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
        elif copula_type == CopulaTypes.GUMBEL:
            copula_values = gumbel_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
        elif copula_type == CopulaTypes.FRANK:
            copula_values = frank_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
        
        # Create a result tensor that holds copula values in the remaining positions
        copula_result[remaining_mask] = copula_values

    # Use torch.where to apply the copula values in the appropriate positions
    result = torch.where(remaining_mask, copula_result, result)

    return result

""" This is changed with the dynamic training loop
if remaining_mask.any():
    if copula_type == CopulaTypes.CLAYTON:
        copula_values = clayton_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
    elif copula_type == CopulaTypes.GUMBEL:
        copula_values = gumbel_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
    elif copula_type == CopulaTypes.FRANK:
        copula_values = frank_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
    
    # Create a result tensor that holds copula values in the remaining positions
    copula_result = torch.zeros_like(result, dtype=torch.float32)
    copula_result[remaining_mask] = copula_values

    # Use torch.where to apply the copula values in the appropriate positions
    result = torch.where(remaining_mask, copula_result, result)
"""

def get_copula(copula_type: CopulaTypes | ArchCopulaTypes, cdf1, cdf2, theta, tolerance=1e-5):
    if cdf1 < tolerance or cdf2 < tolerance:
        return 0
    if cdf1 > 1 - tolerance and cdf2 > 1 - tolerance:
        return 1 
    elif cdf1 > 1 - tolerance:
        return cdf2
    elif cdf2 > 1 - tolerance:
        return cdf1 

    if isinstance(copula_type, CopulaTypes):
        copula_type = copula_type_mapper[copula_type]
    
    if copula_type.is_gumbel_type():
        return get_gumbel_copula(copula_type, cdf1, cdf2, theta)
    elif copula_type.is_clayton_type():
        return get_clayton_copula(copula_type, cdf1, cdf2, theta)
    else:
        return frank_copula(cdf1, cdf2, theta)

def get_copula_parallel(copula_type, cdf1, cdf2, theta, tolerance=1e-5):
    cdf1 = np.array(cdf1)
    cdf2 = np.array(cdf2)

    # Handle tolerance conditions
    result = np.zeros_like(cdf1)
    mask_tolerance = (cdf1 >= tolerance) & (cdf2 >= tolerance)
    mask_upper = (cdf1 > 1 - tolerance) & (cdf2 > 1 - tolerance)
    mask_cdf1_upper = (cdf1 > 1 - tolerance) & ~mask_upper
    mask_cdf2_upper = (cdf2 > 1 - tolerance) & ~mask_upper

    result[mask_upper] = 1
    result[mask_cdf1_upper] = cdf2[mask_cdf1_upper]
    result[mask_cdf2_upper] = cdf1[mask_cdf2_upper]

    # Mask for computing copula values
    mask_compute = mask_tolerance & ~(mask_upper | mask_cdf1_upper | mask_cdf2_upper)

    if isinstance(copula_type, CopulaTypes):
        copula_type = copula_type_mapper[copula_type]

    # Compute the copula values
    if copula_type.is_gumbel_type():
        result[mask_compute] = get_gumbel_copula(copula_type, cdf1[mask_compute], cdf2[mask_compute], theta)
    elif copula_type.is_clayton_type():
        result[mask_compute] = get_clayton_copula(copula_type, cdf1[mask_compute], cdf2[mask_compute], theta)
    else:
        result[mask_compute] = frank_copula(cdf1[mask_compute], cdf2[mask_compute], theta)

    return result

def get_theta(*args):
    copula_type: CopulaTypes | ArchCopulaTypes
    copula_type, data_1, data_2 = args[0]
    if isinstance(copula_type, CopulaTypes):
        copula_type = copula_type_mapper[copula_type]
        
    match copula_type:
        case CopulaTypes.CLAYTON | ArchCopulaTypes.CLAYTON:
            return _get_theta_for_clayton_copula(data_1, data_2)
        case CopulaTypes.GUMBEL | ArchCopulaTypes.GUMBEL | ArchCopulaTypes.GUMBEL_90 | ArchCopulaTypes.GUMBEL_180 | ArchCopulaTypes.GUMBEL_270:
            return _get_theta_gumbell(data_1, data_2)
        case CopulaTypes.FRANK | ArchCopulaTypes.FRANK:
            return _get_theta_frank(data_1, data_2)


def get_theta_from_tau(copula_type: CopulaTypes, tau):
    match copula_type:
        case CopulaTypes.CLAYTON:
            return ((2 * tau) / (1 - tau)) if tau != 1 else 1000
        case CopulaTypes.GUMBEL:
            return (1 / (1 - tau)) if tau != 1 else 1000


def _get_theta_for_clayton_copula(data_1, data_2):
    tau, _ = kendalltau(data_1, data_2)
    print(f"tau: {tau}")
    theta = ((2 * tau) / (1 - tau)) if tau != 1 else 1000
    return theta


def _get_theta_gumbell(data_1, data_2):
    tau, _ = kendalltau(data_1, data_2)
    if tau is np.nan:
        print(f"data_1: {data_1}")
        print(f"data_2: {data_2}")
        return False
    theta = (1 / (1 - tau)) if tau != 1 else 1000
    return min(theta, 1000)

def _get_theta_frank(data_1, data_2):
    kendall_tau, _ = kendalltau(data_1, data_2)
    if kendall_tau == 1:
        return float('inf')  # Perfect positive dependence
    elif kendall_tau == -1:
        return float('-inf')  # Perfect negative dependence
    else:
        theta = - (3 * kendall_tau) / (2 * (1 - kendall_tau))
        return theta


if __name__ == "__main__":
    
    u = np.array([0.5, 0.7])
    v = np.array([0.5, 0.7])
    theta = np.array([2.0, 1.5])


    tu = torch.tensor(u, dtype=torch.float32)
    tv = torch.tensor(v, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)
    # theta = 2.0
    ret = get_copula_torch(CopulaTypes.GUMBEL, tu, tv, theta)
    print(ret)