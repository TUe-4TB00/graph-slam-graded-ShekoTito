import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

from helperfunctions import add_landmark_measurement_from_global

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    
    pose_x4 = result.atPose2(X(4))
    landmark_12 = result.atPoint2(L(2))

    distance, rotation = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(4),
        pose=pose_x4,
        landmark_key=L(2),
        landmark_point=landmark_12,
        measurement_noise=MEASUREMENT_NOISE,
        add_factor=False
    )

    rotation_degrees = math.degrees(rotation)

    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromDegrees(rotation_degrees), distance, MEASUREMENT_NOISE))
    return graph