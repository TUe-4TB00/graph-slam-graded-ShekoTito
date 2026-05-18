import numpy as np
import copy
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):

    best_pose = None    
    best_landmark = None 
    best_score = float('inf') # start with worst possible score 
    sum_of_marginals = 0
    
    # Try all combinations (4 poses x 2 landmarks)
    for pose_label, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            # Copy so each trial is fully independent
            graph_trial = copy.deepcopy(graph)
            estimate_trial = copy.deepcopy(initial_estimate)

            # Add X(5) at this candidate position
            graph_trial, estimate_trial = add_pose(graph_trial, estimate_trial, pose_5)

            # First optimization -> to get good landmark positions
            result_trial = optimize(graph_trial, estimate_trial)

            # Add measurement from X(5) to chosen landmark
            graph_trial = add_landmark_measurement(graph_trial, result_trial, pose_5, landmark)

            # Second optimization with the new measurement -> to get updated covariances
            result_trial = optimize(graph_trial, estimate_trial)

            # Compute marginals
            marginals_trial = gtsam.Marginals(graph_trial, result_trial)
            cov_l1 = marginals_trial.marginalCovariance(L(1))
            cov_l2 = marginals_trial.marginalCovariance(L(2))

            # Use trace to find the winner (gives pose d as best)
            score = np.trace(cov_l1) + np.trace(cov_l2)

            print(f"Pose {pose_label}, Landmark {landmark}: score = {score:.6f}")

            # To keep track of the best combination 
            if score < best_score: 
                best_score = score
                best_pose = pose_label
                best_landmark = landmark
                # sum_of_marginals = score
                sum_of_marginals = cov_l1.sum() + cov_l2.sum()
    
    print(f"\nBest pose: {best_pose}, Best landmark: L({best_landmark}), score: {best_score:.6f}")
    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    
    best_pose = None       
    best_landmark = None   
    best_score = float('inf') # Same as above, start with worst possible score
    sum_of_errors = 0

    for pose_label, pose_5 in pose_options.items():
        for landmark in [1, 2]:

            # Independent copy for each trial
            graph_trial = copy.deepcopy(graph)
            estimate_trial = copy.deepcopy(initial_estimate)

            graph_trial, estimate_trial = add_pose(graph_trial, estimate_trial, pose_5)
            result_trial = optimize(graph_trial, estimate_trial)

            graph_trial = add_landmark_measurement(graph_trial, result_trial, pose_5, landmark)
            result_trial = optimize(graph_trial, estimate_trial)

            # Compute marginals
            marginals_trial = gtsam.Marginals(graph_trial, result_trial)
            cov_l1 = marginals_trial.marginalCovariance(L(1))
            cov_l2 = marginals_trial.marginalCovariance(L(2))

            #Rank using .sum() (tried trace but didn't work)
            # This metric picks pose b, L1 as the winner
            score = cov_l1.sum() + cov_l2.sum()

            print(f"Pose {pose_label}, Landmark {landmark}: score = {score:.6f}")

            if score < best_score: 
                best_score = score
                best_pose = pose_label
                best_landmark = landmark
                sum_of_errors = score


                # Return value: actual coordinate errors of X(1), X(2), X(3)
                x1 = result_trial.atPose2(X(1))
                x2 = result_trial.atPose2(X(2))
                x3 = result_trial.atPose2(X(3))
                sum_of_errors = (
                    abs(x1.x()) + abs(x1.y()) + abs(x1.theta()) + # Skip x for X(2) and X(3) —> their x is the true value
                    abs(x2.y()) + abs(x2.theta()) +
                    abs(x3.y()) + abs(x3.theta())
                )


    print(f"\nBest pose: {best_pose}, Best landmark: L({best_landmark}), score: {best_score:.6f}")
    return best_pose, best_landmark, sum_of_errors 