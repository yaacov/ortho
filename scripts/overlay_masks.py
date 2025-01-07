import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

class UndoManager:
    def __init__(self, max_steps=10):
        self.undo_stack = []
        self.max_steps = max_steps
    
    def push_state(self, image):
        """Save current state for undo"""
        self.undo_stack.append(image.copy())
        if len(self.undo_stack) > self.max_steps:
            self.undo_stack.pop(0)
    
    def undo(self):
        """Restore previous state"""
        if len(self.undo_stack) > 0:
            return self.undo_stack.pop()
        return None

def overlay_images(image, mask, alpha=0.3, undo_manager=None):
    """Overlay mask on image with transparency."""
    if undo_manager:
        undo_manager.push_state(image)
        
    # Create a yellow mask overlay
    mask_overlay = np.zeros_like(image)
    mask_overlay[:,:,0] = 255  # Red channel
    mask_overlay[:,:,1] = 255  # Green channel
    
    # Convert binary mask to boolean mask
    bool_mask = mask > 0.5
    
    # Create the overlay
    result = image.copy()
    result[bool_mask] = (1 - alpha) * image[bool_mask] + alpha * mask_overlay[bool_mask]
    
    return result

def process_image_pair(train_path, val_path, output_path):
    """Process a pair of images (original and mask) and save the overlay."""
    # Load original image and mask
    image = plt.imread(train_path)
    mask = plt.imread(val_path)
    
    # Convert grayscale image to RGB if needed
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    # Convert RGBA to RGB if needed
    elif image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    
    # Convert mask to grayscale if it's RGB/RGBA
    if len(mask.shape) == 3:
        if mask.shape[-1] == 4:
            mask = color.rgba2rgb(mask)
        mask = color.rgb2gray(mask)
    
    # Normalize image to 0-255 if needed
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    
    # Initialize undo manager
    undo_manager = UndoManager()
    
    # Create overlay with undo support
    result = overlay_images(image, mask, undo_manager=undo_manager)
    
    # Handle Ctrl-Z (you'll need to integrate this with your UI framework)
    def handle_undo(event):
        if event.key == 'ctrl+z':
            prev_state = undo_manager.undo()
            if prev_state is not None:
                return prev_state
    
    # Save result
    plt.imsave(output_path, result)
    print(f"Saved overlay to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Overlay masks on original images.")
    parser.add_argument(
        "--input_dir",
        default="data/processed/img-line-masks",
        help="Path to input directory containing 'train' and 'val' subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/img-line-overlays",
        help="Path to save the overlayed images",
    )

    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get paths
    train_dir = os.path.join(args.input_dir, "inputs")
    val_dir = os.path.join(args.input_dir, "targets")
    
    # Get list of files in train directory
    train_files = sorted(os.listdir(train_dir))
    
    for filename in train_files:
        train_path = os.path.join(train_dir, filename)
        val_path = os.path.join(val_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        # Check if corresponding mask exists
        if os.path.exists(val_path):
            try:
                process_image_pair(train_path, val_path, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        else:
            print(f"No matching mask found for {filename}")

if __name__ == "__main__":
    main()
