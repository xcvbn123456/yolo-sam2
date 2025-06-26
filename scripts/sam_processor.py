import torch
import numpy as np
from PIL import Image
import cv2 # For saving mask
import os

# Assuming sam2 is installed and importable.
# If sam2 package is in the parent directory of 'scripts',
# and 'scripts' is not a package itself, direct import might fail.
# We might need to adjust sys.path in run_pipeline.py if 'sam2' is not in PYTHONPATH
# or not installed in a way that makes it discoverable.
# Based on project structure (sam2/sam2/__init__.py), imports should be:
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor


def initialize_sam_predictor(model_cfg_path, sam_checkpoint_path, device="cuda"):
    """Initializes and returns the SAM2ImagePredictor."""
    # Autocast and TF32 settings are often global or managed by a higher-level context.
    # If running on Ampere GPUs or newer, these are good defaults.
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # print("SAM Processor: Enabling TF32 for CUDA matmul and cuDNN.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build_sam2 and SAM2ImagePredictor might manage their own autocast or expect it to be set.
    # The original script had a global autocast. For a library function, it's trickier.
    # Let's assume for now that if autocast is needed during model loading, it should be
    # active when this function is called.
    sam2_model = build_sam2(model_cfg_path, sam_checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    # print(f"SAM Processor: Predictor initialized on {device}.")
    return predictor

def generate_sam_mask(
    predictor,
    image_path,
    input_point_coords, # e.g., np.array([[x, y]])
    input_point_label,  # e.g., np.array([1])
    output_mask_path, # Can be None if only array output is needed
    multimask_output=True,
    device="cuda" # Added device for autocast context
):
    """
    Generates a segmentation mask for an object in an image using SAM2.
    Optionally saves the mask to a file and returns the mask as a NumPy array.

    Returns:
        tuple: (bool, np.array or None): Success status and the binary mask array (H, W) or None.
    """
    try:
        image = Image.open(image_path)
        image_np = np.array(image.convert("RGB"))

        # Use autocast for the prediction part, as per original script's practice
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device=="cuda")):
            predictor.set_image(image_np)
            masks, scores, _ = predictor.predict(
                point_coords=input_point_coords,
                point_labels=input_point_label,
                multimask_output=multimask_output,
            )

        if masks is None or len(masks) == 0:
            print(f"SAM Processor: SAM2 did not return any masks for {image_path} with point {input_point_coords}")
            return False, None

        sorted_ind = np.argsort(scores)[::-1]
        best_mask = masks[sorted_ind[0]]  # This is a boolean mask (H, W)
        binary_mask_array = best_mask.astype(np.uint8) * 255 # Convert to uint8 binary image (0 or 255)

        if output_mask_path:
            output_dir = os.path.dirname(output_mask_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            cv2.imwrite(output_mask_path, binary_mask_array)
            # print(f"SAM Processor: Mask saved to {output_mask_path}")

        return True, binary_mask_array

    except Exception as e:
        print(f"SAM Processor: Error in generate_sam_mask for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == '__main__':
    print("SAM Processor: Running example usage...")
    # These paths must be correctly set for the example to run.
    # The user provided these paths earlier, assuming they are accessible from where this script runs.
    # SAM2 Checkpoint: /home/super/catkin_ws/src/screw_ros/SAM222/checkpoints/sam2_hiera_tiny.pt
    # SAM2 Model Config: sam2_hiera_t.yaml (build_sam2 needs to find this, e.g. in sam2/configs or via full path)

    # For the example, let's assume these paths are correct and files exist.
    # IMPORTANT: User needs to ensure these paths are valid in their execution environment.
    sam_checkpoint_path = "/home/super/catkin_ws/src/screw_ros/SAM222/checkpoints/sam2_hiera_tiny.pt" # User-confirmed path from sam_bigsure.py

    # For model_cfg, build_sam2 might look in a relative path like `sam2/configs/`
    # If 'sam2_hiera_t.yaml' is a key for a config file within the sam2 library's structure,
    # just the filename might be okay. Otherwise, a full or correct relative path is needed.
    # Let's assume it's a key that build_sam2 can resolve.
    # If not, this will fail: sam2_model = build_sam2(model_cfg_path, ...)
    # A safer bet for testing might be to provide a full path if known, or ensure CWD is sam2/
    model_config_name = "sam2_hiera_t.yaml" # User-confirmed name from sam_bigsure.py

    # Example image - this needs to be a valid path to an image.
    # The path below is from the original sam_bigsure.py script.
    # Replace with a readily available image for testing if that one isn't present.
    # For robustness, let's try to create a dummy image if a known one isn't available.
    test_image_path = "scripts/sam_test_image.jpg"
    test_output_mask_path = "scripts/sam_test_output_mask.png"

    # Create a dummy image for the example if the hardcoded one is not there
    if not os.path.exists(test_image_path):
        print(f"SAM Processor: Test image {test_image_path} not found, creating a dummy image.")
        dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "TEST IMAGE", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.imwrite(test_image_path, dummy_img)

    # Example point prompt (center of the dummy image)
    img_h, img_w = cv2.imread(test_image_path).shape[:2]
    example_point = np.array([[img_w // 2, img_h // 2]])
    example_label = np.array([1])

    current_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM Processor: Using device: {current_device}")

    try:
        # The global autocast context was used in the original script.
        # For modularity, it's often better to limit its scope, e.g., within predict.
        # However, to mimic original behavior for the example:
        with torch.autocast(device_type=current_device, dtype=torch.bfloat16, enabled=(current_device=="cuda")):
            print(f"SAM Processor: Initializing SAM2 predictor with config '{model_config_name}' and checkpoint '{sam_checkpoint_path}'...")
            # Check if checkpoint exists
            if not os.path.exists(sam_checkpoint_path):
                print(f"SAM Processor: FATAL ERROR - SAM Checkpoint not found at {sam_checkpoint_path}")
                print("Please ensure the path is correct and the file exists.")
            # build_sam2 will try to find model_config_name, e.g., in sam2.configs.MODEL_CFGS
            # If it's a direct file path, it should also work.
            else:
                sam_predictor = initialize_sam_predictor(model_config_name, sam_checkpoint_path, device=current_device)
                print("SAM Processor: SAM2 predictor initialized.")

                print(f"SAM Processor: Generating mask for {test_image_path}...")
                success, mask_array = generate_sam_mask(
                    sam_predictor,
                    test_image_path,
                    example_point,
                    example_label,
                    test_output_mask_path, # Still save for example
                    device=current_device
                )
                if success and mask_array is not None:
                    print(f"SAM Processor: Mask generation example complete. Mask saved to {test_output_mask_path}. Mask array shape: {mask_array.shape}")
                elif success and mask_array is None: # Should not happen if success is True
                     print("SAM Processor: Mask generation reported success but no mask array returned.")
                else:
                    print("SAM Processor: Mask generation example failed.")
    except Exception as e:
        print(f"SAM Processor: An error occurred in SAM example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_image_path) and "dummy_image" in test_image_path : # Clean up if it was a dummy
             os.remove(test_image_path)
        # Keep output mask for inspection if example was run.
        pass
