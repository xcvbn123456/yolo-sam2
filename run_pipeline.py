import subprocess
import json
import os
import numpy as np
import sys
from pathlib import Path # Ensure Path is imported

# Add scripts directory to sys.path to allow direct import of modules
# This assumes run_pipeline.py is in the project root and scripts/ is a subdirectory.
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Add project root to sys.path to help with 'sam2' imports if 'sam2' is a top-level dir
project_root_path = os.path.dirname(os.path.abspath(__file__))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)


try:
    from pose_estimator import triangulate_object_pose, camera_params_yaml_string as default_cam_params
    from sam_processor import initialize_sam_predictor, generate_sam_mask
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print(f"Ensure scripts directory '{scripts_dir}' is accessible and contains pose_estimator.py and sam_processor.py.")
    print(f"Also ensure 'sam2' package is importable (check PYTHONPATH or if it's in project root: {project_root_path}).")
    sys.exit(1)

import torch # Import torch after sys.path modifications, for sam_processor

# --- Configuration Section ---

# Image Paths (User should verify these paths)
LEFT_IMAGE_PATH = "data/left_image.png" # Placeholder
RIGHT_IMAGE_PATH = "data/right_image.png" # Placeholder
# Actual paths from user were:
# LEFT_IMAGE_PATH = "/home/lyl/cqy_graduation/calib/images/0/left.png"
# RIGHT_IMAGE_PATH = "/home/lyl/cqy_graduation/calib/images/1/right.png"


# YOLO Configuration
YOLO_WEIGHTS_PATH = "/home/lyl/yolov9/runs/train/exp9/weights/best.pt" # User-provided
YOLO_SCRIPT_PATH = "yolov9/detect.py" # Relative to project root
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45
YOLO_PROJECT_DIR = "runs/yolo_detections" # Output directory for YOLO

# SAM2 Configuration
SAM_CHECKPOINT_PATH = "/home/super/catkin_ws/src/screw_ros/SAM222/checkpoints/sam2_hiera_tiny.pt" # From sam_bigsure.py
SAM_MODEL_CONFIG = "sam2_hiera_t.yaml" # From sam_bigsure.py (build_sam2 needs to find this)

# Output Directories
MASKS_OUTPUT_DIR = "output_masks" # For saving masks for debugging
os.makedirs(YOLO_PROJECT_DIR, exist_ok=True)
os.makedirs(MASKS_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LEFT_IMAGE_PATH), exist_ok=True) # Ensure data dir exists for dummy images
os.makedirs(os.path.dirname(RIGHT_IMAGE_PATH), exist_ok=True)


# Camera Parameters (from pose_estimator.py, or load from a file)
CAMERA_PARAMETERS_YAML = default_cam_params


def run_yolo_detection(image_path, yolo_weights, yolo_script, project_dir, base_name):
    """Runs YOLO detection and returns the path to the JSON output."""
    print(f"Running YOLO on {image_path}...")
    exp_name = f"{base_name}_yolo_out"

    cmd = [
        sys.executable, yolo_script,
        "--weights", yolo_weights,
        "--source", image_path,
        "--conf-thres", str(YOLO_CONF_THRES),
        "--iou-thres", str(YOLO_IOU_THRES),
        "--save-json",
        "--project", project_dir,
        "--name", exp_name,
        "--exist-ok",
        "--nosave"
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120) # Added timeout
        # print("YOLO Output STDOUT:", process.stdout) # Can be very verbose
        if process.stderr:
            print("YOLO Output STDERR:", process.stderr)
    except subprocess.TimeoutExpired:
        print(f"YOLO process timed out for {image_path}.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running YOLO for {image_path}: {e.returncode}")
        print("YOLO Error STDOUT:", e.stdout)
        print("YOLO Error STDERR:", e.stderr)
        return None

    json_output_path = Path(project_dir) / exp_name / "labels_json" / f"{Path(image_path).stem}.json"

    if not json_output_path.exists():
        print(f"YOLO JSON output not found at {json_output_path}")
        label_dir = Path(project_dir) / exp_name / "labels_json"
        if label_dir.exists():
            json_files = list(label_dir.glob('*.json'))
            if json_files:
                json_output_path = json_files[0]
                print(f"Found JSON file: {json_output_path}")
            else:
                print(f"No JSON files found in {label_dir}")
                return None
        else:
            print(f"Label directory {label_dir} does not exist.")
            return None

    return str(json_output_path)


def get_bbox_from_yolo_json(json_path, pick_highest_conf=True):
    """Parses YOLO JSON output and returns the first or highest confidence bounding box."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        detections = data.get("detections", [])
        if not detections:
            print(f"No detections found in {json_path}")
            return None

        if pick_highest_conf:
            detections.sort(key=lambda d: d.get("confidence", 0), reverse=True)

        chosen_det = detections[0]
        bbox_data = chosen_det.get("bbox")
        return [bbox_data['xmin'], bbox_data['ymin'], bbox_data['xmax'], bbox_data['ymax']]

    except Exception as e:
        print(f"Error parsing YOLO JSON {json_path}: {e}")
        return None

def get_point_prompt_from_bbox(bbox_xyxy):
    """Converts a bounding box [xmin, ymin, xmax, ymax] to a center point prompt."""
    if bbox_xyxy is None: return None, None
    xmin, ymin, xmax, ymax = bbox_xyxy
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    point_coords = np.array([[cx, cy]])
    point_label = np.array([1])
    return point_coords, point_label

def create_dummy_images_if_not_exist(left_path, right_path):
    """Creates dummy PNG images if the specified paths don't exist."""
    import cv2

    for img_path in [left_path, right_path]:
        img_dir = os.path.dirname(img_path)
        if img_dir and not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)

        if not os.path.exists(img_path):
            print(f"Image not found at {img_path}. Creating a dummy image.")
            dummy_img_arr = np.zeros((600, 800, 3), dtype=np.uint8)
            color = (0, 255, 0) if "left" in img_path.lower() else (0, 0, 255)
            text = "LEFT DUMMY" if "left" in img_path.lower() else "RIGHT DUMMY"
            cv2.putText(dummy_img_arr, text, (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            cv2.imwrite(img_path, dummy_img_arr)
            print(f"Dummy image created at {img_path}")


def main_pipeline():
    print("Starting 3D Object Pose Estimation Pipeline...")
    create_dummy_images_if_not_exist(LEFT_IMAGE_PATH, RIGHT_IMAGE_PATH)

    print("Initializing SAM2 predictor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam_predictor = None
    try:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device=="cuda")):
             sam_predictor = initialize_sam_predictor(SAM_MODEL_CONFIG, SAM_CHECKPOINT_PATH, device=device)
    except Exception as e:
        print(f"Failed to initialize SAM predictor: {e}")
        import traceback
        traceback.print_exc()
        if not os.path.exists(SAM_CHECKPOINT_PATH): print(f"SAM Checkpoint not found: {SAM_CHECKPOINT_PATH}")
        print("Ensure SAM checkpoint and model config paths are correct and files are accessible.")
        return

    if sam_predictor is None: # Should be caught by exception, but as a safeguard
        print("SAM predictor could not be initialized. Exiting.")
        return
    print("SAM2 predictor initialized successfully.")

    processed_data = []
    for img_idx, image_path in enumerate([LEFT_IMAGE_PATH, RIGHT_IMAGE_PATH]):
        img_label = "left" if img_idx == 0 else "right"
        print(f"\n--- Processing {img_label} image: {image_path} ---")

        yolo_json_output = run_yolo_detection(
            image_path, YOLO_WEIGHTS_PATH, YOLO_SCRIPT_PATH,
            YOLO_PROJECT_DIR, Path(image_path).stem
        )
        if not yolo_json_output:
            print(f"YOLO detection failed for {image_path}.")
            processed_data.append({"status": "yolo_failed", "mask_array": None})
            continue

        bbox = get_bbox_from_yolo_json(yolo_json_output)
        if not bbox:
            print(f"No bounding box in YOLO output for {image_path}.")
            processed_data.append({"status": "no_bbox", "mask_array": None})
            continue

        point_coords, point_label = get_point_prompt_from_bbox(bbox)
        if point_coords is None:
            processed_data.append({"status": "no_prompt", "mask_array": None})
            continue

        # Define where to save the mask image (for debugging, can be None if not needed)
        mask_debug_save_path = os.path.join(MASKS_OUTPUT_DIR, f"{Path(image_path).stem}_mask.png")

        # generate_sam_mask now returns (bool, mask_array)
        # output_mask_path (mask_debug_save_path) is for optional saving within generate_sam_mask
        sam_success, returned_mask_array = generate_sam_mask(
            sam_predictor,
            image_path,
            point_coords,
            point_label,
            mask_debug_save_path, # Pass path for saving, or None to not save from sam_processor
            device=device
        )

        if not sam_success or returned_mask_array is None:
            print(f"SAM2 mask generation failed for {image_path}.")
            processed_data.append({"status": "sam_failed", "mask_array": None}) # mask_array is None
            continue

        # We have the mask_array directly
        print(f"SAM2 mask generated for {img_label}, array shape: {returned_mask_array.shape}. Debug mask saved to: {mask_debug_save_path if mask_debug_save_path else 'No path'}")
        processed_data.append({"status": "success", "mask_array": returned_mask_array})

    # Check results before triangulation
    if len(processed_data) != 2 or \
       processed_data[0].get("status") != "success" or processed_data[0].get("mask_array") is None or \
       processed_data[1].get("status") != "success" or processed_data[1].get("mask_array") is None:
        print("\nPipeline did not successfully generate mask arrays for both images. Cannot triangulate.")
        if len(processed_data) > 0: print(f"Left image status: {processed_data[0].get('status')}")
        if len(processed_data) > 1: print(f"Right image status: {processed_data[1].get('status')}")
        return

    left_mask_array = processed_data[0]["mask_array"]
    right_mask_array = processed_data[1]["mask_array"]
    # No need to print mask paths as we are using arrays directly
    print(f"\nLeft and right mask arrays generated successfully and will be used for triangulation.")

    # 3. Perform Stereo Triangulation
    print("\n--- Performing Stereo Triangulation ---")
    try:
        # Call triangulate_object_pose with the NumPy arrays for masks
        pose_3d = triangulate_object_pose(left_mask_array, right_mask_array, CAMERA_PARAMETERS_YAML)
        if pose_3d is not None:
            print(f"\nSuccessfully calculated 3D Pose!")
            print(f"Coordinates (X, Y, Z) in left camera frame: {pose_3d}")
            ros_X, ros_Y, ros_Z = pose_3d[2], -pose_3d[0], -pose_3d[1]
            print(f"Adjusted for ROS-like output (X_ros=Z_cv, Y_ros=-X_cv, Z_ros=-Y_cv):")
            print(f"  X: {ros_X:.3f}\n  Y: {ros_Y:.3f}\n  Z: {ros_Z:.3f}")
        else:
            print("Triangulation failed or returned no result.")
    except Exception as e:
        print(f"An error occurred during triangulation: {e}")
        import traceback
        traceback.print_exc()

    print("\nPipeline finished.")

if __name__ == "__main__":
    if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
        torch.multiprocessing.set_start_method('spawn', force=True)

    if not os.path.exists(YOLO_SCRIPT_PATH):
        print(f"CRITICAL: YOLO script not found: {os.path.abspath(YOLO_SCRIPT_PATH)}")
        sys.exit(1)
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        print(f"WARNING: YOLO weights not found: {YOLO_WEIGHTS_PATH}. Detection will likely fail.")

    main_pipeline()
