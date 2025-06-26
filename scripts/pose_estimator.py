import cv2
import numpy as np
import yaml # For loading camera parameters if stored in YAML

# Placeholder for camera parameters (will be loaded or passed as arguments)
# These would ideally be loaded from the YAML-like structure you provided
# For now, let's define them directly or assume they are passed to the main function.

# K_left, D_left, K_right, D_right, R, T, img_size will be needed.

def load_camera_parameters(yaml_string_data):
    """Loads camera parameters from a YAML string."""
    params = yaml.safe_load(yaml_string_data)
    K_left = np.array(params['K_left'])
    D_left = np.array(params['D_left'][0]) # Assuming D_left is a list containing one list
    K_right = np.array(params['K_right'])
    D_right = np.array(params['D_right'][0]) # Assuming D_right is a list containing one list
    R = np.array(params['R'])
    T = np.array(params['T']).flatten() # Ensure T is a 1D array (3,)
    img_size = tuple(params['img_size'])
    return K_left, D_left, K_right, D_right, R, T, img_size

def find_mask_centroid(mask_array, mask_identifier="mask"):
    """Finds the centroid of the largest contour in a given mask NumPy array."""
    if mask_array is None:
        raise ValueError(f"Input mask_array for '{mask_identifier}' is None.")
    if not isinstance(mask_array, np.ndarray):
        raise TypeError(f"Input mask_array for '{mask_identifier}' must be a NumPy array.")
    if mask_array.ndim != 2: # Expecting grayscale mask
        raise ValueError(f"Input mask_array for '{mask_identifier}' must be 2D. Shape is {mask_array.shape}")


    # Ensure mask is binary (0 or 255) if it's not already.
    # SAM processor should output this way, but good to be robust.
    # If mask contains other values, threshold it.
    if not np.all(np.isin(mask_array, [0, 255])):
        print(f"Warning: Mask '{mask_identifier}' not purely binary (0, 255). Thresholding at 128.")
        _, mask_array = cv2.threshold(mask_array, 128, 255, cv2.THRESH_BINARY)
        mask_array = mask_array.astype(np.uint8)


    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Warning: No contours found in mask '{mask_identifier}'")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        print(f"Warning: m00 is 0, cannot compute centroid for mask '{mask_identifier}'")
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return np.array([[cx, cy]], dtype=np.float32) # Return as (1, 1, 2) array for undistortPoints

def triangulate_object_pose(left_mask_array, right_mask_array, cam_params_yaml_string):
    """
    Calculates the 3D pose of an object given its masks (as NumPy arrays)
    from left and right stereo images.

    Args:
        left_mask_array (np.array): NumPy array of the mask from the left camera.
        right_mask_array (np.array): NumPy array of the mask from the right camera.
        cam_params_yaml_string (str): String containing camera parameters in YAML format.

    Returns:
        np.array: The 3D coordinates (X, Y, Z) of the object, or None if error.
    """
    K_left, D_left, K_right, D_right, R_stereo, T_stereo, img_size = \
        load_camera_parameters(cam_params_yaml_string)

    # 1. Find centroids of masks
    left_point_distorted_coords = find_mask_centroid(left_mask_array, "left_mask")
    right_point_distorted_coords = find_mask_centroid(right_mask_array, "right_mask")

    if left_point_distorted_coords is None or right_point_distorted_coords is None:
        print("Error: Could not find centroid in one or both mask arrays.")
        return None

    # 2. Get rectification transforms and projection matrices
    # R_stereo is rotation from left to right cam, T_stereo is translation from left to right cam
    # img_size should be (width, height)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, D_left,
        K_right, D_right,
        img_size, R_stereo, T_stereo,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1
    )

    # 3. Undistort points and transform them to the rectified coordinate system.
    # The points should be in the format expected by undistortPoints: (N, 1, 2) or (1, N, 2)
    # find_mask_centroid returns them as np.array([[cx, cy]], dtype=np.float32) which is (1,1,2)

    # Correct usage of undistortPoints for rectified points:
    # Pass R=R1 (or R2) and P=P1 (or P2)
    # This transforms points from the original distorted image to the rectified image
    left_point_rectified = cv2.undistortPoints(left_point_distorted_coords, K_left, D_left, R=R1, P=P1)
    right_point_rectified = cv2.undistortPoints(right_point_distorted_coords, K_right, D_right, R=R2, P=P2)

    if left_point_rectified is None or right_point_rectified is None:
        print("Error: Point undistortion and rectification failed.")
        return None

    # 4. Triangulate points
    # points4D_hom will be 4xN array. triangulatePoints expects 2xN arrays for points.
    # Reshape rectified points from (1,1,2) to (2,1)
    points4D_hom = cv2.triangulatePoints(P1, P2, left_point_rectified.reshape(2, -1), right_point_rectified.reshape(2, -1))

    # 5. Convert to Euclidean coordinates
    points3D = points4D_hom[:3] / points4D_hom[3]

    return points3D.T[0] # Return as a 1D array [X, Y, Z]

if __name__ == '__main__':
    camera_params_yaml_string = """
D_left:
- - [-0.09232185240241979, 0.1773924651423729, -0.0001600137512772027, -0.0005613759595648238, -0.012369108999031563]
D_right:
- - [-0.09095426235446184, 0.07881024095930775, 0.00018193006933585844, 0.0003402274251635849, 0.3568843208121935]
E:
- [0.025017913725526475, -7.957540190920693, 0.05301274794484878]
- [-8.791296510807303, -0.12936080399566893, 94.31391181138339]
- [0.36474543877417664, -94.38726668851889, -0.10065522851688417]
F:
- [-2.4060494130760035e-09, 7.6532669002052e-07, -0.00042081925778631065]
- [8.455451326427749e-07, 1.2442313467772342e-08, -0.02246333938993815]
- [-0.0005479741201890524, 0.021398612656402506, 1.0]
K_left:
- [2410.6192871723683, 0.0, 697.4417431766733]
- [0.0, 2410.538433663819, 535.9894603671755]
- [0.0, 0.0, 1.0]
K_right:
- [2419.600162255308, 0.0, 730.7136330233421]
- [0.0, 2419.4307210170527, 549.7706037794145]
- [0.0, 0.0, 1.0]
R:
- [0.984357025335072, 0.004268138412977445, 0.1761335563371616]
- [-0.004506182837047281, 0.999989394379795, 0.0009515483014976251]
- [-0.17612762699170031, -0.001730353264119462, 0.9843658186303778]
T: # Units of T will determine units of output 3D point
- [-94.38800504096402]
- [0.0615469468955383]
- [7.9575180873748455]
img_size: [1440, 1080] # width, height
"""
    # Create dummy mask files for testing
    dummy_left_mask_path = "dummy_left_mask.png"
    dummy_right_mask_path = "dummy_right_mask.png"

    # Ensure scripts directory exists if it's run from root
    import os
    if not os.path.exists("scripts"):
        os.makedirs("scripts", exist_ok=True)

    # Use paths relative to where the script is, if this script is in scripts/
    # For simplicity, assume paths are relative to CWD when script is run.

    # Create simple dummy masks (e.g., a white square on black background)
    # Image size from parameters: 1080 height, 1440 width
    h, w = 1080, 1440
    dummy_mask_img_left = np.zeros((h, w), dtype=np.uint8)
    # Define a bounding box for the left image's mask
    # Let's put it somewhere reasonable, e.g. x from 600-800, y from 400-600
    cv2.rectangle(dummy_mask_img_left, (600, 400), (800, 600), 255, -1)
    cv2.imwrite(dummy_left_mask_path, dummy_mask_img_left)

    dummy_mask_img_right = np.zeros((h, w), dtype=np.uint8)
    # Define a bounding box for the right image's mask, shifted for disparity
    # e.g., shifted left by 50 pixels: x from 550-750, y from 400-600
    cv2.rectangle(dummy_mask_img_right, (550, 400), (750, 600), 255, -1)
    # cv2.imwrite(dummy_right_mask_path, dummy_mask_img_right) # Not saving, using array directly

    try:
        # For testing, load the saved dummy masks into arrays if needed, or directly use the arrays
        # pose_3d = triangulate_object_pose(cv2.imread(dummy_left_mask_path, cv2.IMREAD_GRAYSCALE),
        #                                   cv2.imread(dummy_right_mask_path, cv2.IMREAD_GRAYSCALE),
        #                                   camera_params_yaml_string)
        # Or directly use the created arrays:
        pose_3d = triangulate_object_pose(dummy_mask_img_left, dummy_mask_img_right, camera_params_yaml_string)

        if pose_3d is not None:
            print(f"Calculated 3D Pose (X, Y, Z) in left camera frame: {pose_3d}")
            # Original ROS node output was: Z, -X, -Y.
            # This implies original X was right, Y was down, Z was forward.
            # And desired output was X_ros = Z_cv, Y_ros = -X_cv, Z_ros = -Y_cv (if assuming Z up for ROS)
            # Or, if ROS also X right, Y up, Z forward: X_ros=X_cv, Y_ros=-Y_cv, Z_ros=Z_cv
            # The prompt's PoseStamped had x=Z, y=-X, z=-Y. This means:
            # ROS X-axis = OpenCV Z-axis (depth)
            # ROS Y-axis = OpenCV -X-axis (left)
            # ROS Z-axis = OpenCV -Y-axis (up)
            print(f"Adjusted for original ROS node's PoseStamped (X=Z, Y=-X, Z=-Y): [{pose_3d[2]:.3f}, {-pose_3d[0]:.3f}, {-pose_3d[1]:.3f}]")

    except Exception as e:
        print(f"An error occurred during triangulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_left_mask_path):
            os.remove(dummy_left_mask_path)
        if os.path.exists(dummy_right_mask_path):
            os.remove(dummy_right_mask_path)
