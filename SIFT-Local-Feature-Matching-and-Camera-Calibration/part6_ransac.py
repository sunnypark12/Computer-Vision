import math

import numpy as np
from vision.part5_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    if ind_prob_correct == 0:
        return float('inf') 
    num_samples = np.log(1 - prob_success) / np.log(1 - (ind_prob_correct ** sample_size))


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(np.ceil(num_samples))


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    prob_success = 0.99
    sample_size = 8 
    ind_prob_correct = 0.5 
    threshold = 0.1 

    n_iters = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct)

    max_inliers = []
    best_F = None

    for _ in range(n_iters):
        sample_indices = np.random.choice(matches_a.shape[0], sample_size, replace=False)
        sampled_points_a = matches_a[sample_indices]
        sampled_points_b = matches_b[sample_indices]

        F = estimate_fundamental_matrix(sampled_points_a, sampled_points_b)

        inliers = []
        for i in range(matches_a.shape[0]):
            pt_a = np.append(matches_a[i], 1) 
            pt_b = np.append(matches_b[i], 1)

            error = np.abs(pt_b.T @ F @ pt_a)

            if error < threshold:
                inliers.append(i)

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_F = F

    inliers_a = matches_a[max_inliers]
    inliers_b = matches_b[max_inliers]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
