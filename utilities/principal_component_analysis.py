import numpy as np
import pandas as pd


def principal_component_analysis(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a matrix, returns the eigenvalues vector and the eigenvectors matrix.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sorting from greatest to lowest the eigenvalues and the eigenvectors
    sort_indices = eigenvalues.argsort()[::-1]

    return eigenvalues[sort_indices], eigenvectors[:, sort_indices]


def align_eigenvectors_to_previous(
    current_eig_vecs: pd.DataFrame,
    previous_eig_vecs: pd.DataFrame | None,
) -> pd.DataFrame:
    aligned = current_eig_vecs.copy()

    if previous_eig_vecs is None:
        if aligned.iloc[:, 0].sum() < 0:
            aligned.iloc[:, 0] *= -1
        return aligned

    common_assets = aligned.index.intersection(previous_eig_vecs.index)
    components_num = min(
        len(common_assets),
        aligned.shape[1],
        previous_eig_vecs.shape[1],
    )
    for pc in range(components_num):
        similarity = float(
            aligned.loc[common_assets, pc].dot(previous_eig_vecs.loc[common_assets, pc])
        )
        if similarity < 0:
            aligned.iloc[:, pc] *= -1

    return aligned


def pca_denoise_covariance(
    cov_matrix: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Denoise a covariance matrix by keeping k signal eigenvalues and averaging the rest.

    Args:
        cov_matrix: Sample covariance matrix (N x N).
        k: Number of principal components to retain as signal.

    Returns:
        Denoised covariance matrix and its eigenvalues (sorted descending).
    """
    eig_vals, eig_vecs = principal_component_analysis(cov_matrix)
    noise_avg = np.mean(eig_vals[k:])
    denoised_eig_vals = np.concatenate(
        [eig_vals[:k], np.full(len(eig_vals) - k, noise_avg)]
    )
    denoised_cov = (eig_vecs * denoised_eig_vals) @ eig_vecs.T
    return denoised_cov, denoised_eig_vals
