import os
import nibabel as nib
import numpy as np
import scipy.io as sio
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def prepare_dataset(t2_dir, mask_dir, output_dir, target_shape=(160, 160), test_split=0.2, val_split=0.1, visualize=True):
    """
    Prepares T2 and mask data for 2D U-Net while preserving subject integrity. 
    
    Parameters:
    -----------
    t2_dir : str
        Directory containing skull-stripped T2 NIfTI files
    mask_dir : str
        Directory containing ground truth mask .mat files
    output_dir : str
        Root directory for organized dataset
    target_shape : tuple
        Target dimensions (width, height) for all slices
    test_split : float
        Proportion of data for testing
    val_split : float
        Proportion of training data for validation
    visualize : bool
        Whether to create visualizations to verify pairing
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'masks'), exist_ok=True)
    
    # Create visualization directory if needed
    if visualize:
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Get all subject IDs (assuming filenames have subject IDs as prefixes)
    all_t2_files = sorted([f for f in os.listdir(t2_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    all_mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.mat')])
    
    # Extract subject IDs and create a mapping
    subject_pairs = []
    
    for t2_file in all_t2_files:
        # Assuming format like "subject001_t2.nii.gz"
        subject_id = t2_file.split('_')[0]
        
        # Find corresponding mask file
        matching_mask = None
        for mask_file in all_mask_files:
            if mask_file.startswith(subject_id):
                matching_mask = mask_file
                break
        
        if matching_mask:
            subject_pairs.append((t2_file, matching_mask))
        else:
            print(f"Warning: No matching mask found for {t2_file}")
    
    # Split subjects into train, val, test while preserving subject integrity
    train_val_pairs, test_pairs = train_test_split(subject_pairs, test_size=test_split, random_state=42)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_split/(1-test_split), random_state=42)
    
    # Process each split
    process_split(train_pairs, t2_dir, mask_dir, os.path.join(output_dir, 'train'), target_shape, visualize, output_dir)
    process_split(val_pairs, t2_dir, mask_dir, os.path.join(output_dir, 'val'), target_shape, visualize, output_dir)
    process_split(test_pairs, t2_dir, mask_dir, os.path.join(output_dir, 'test'), target_shape, visualize, output_dir)
    
    print(f"Dataset prepared: {len(train_pairs)} training, {len(val_pairs)} validation, {len(test_pairs)} test subjects")
    
    # Create a verification report if visualizations were generated
    if visualize:
        create_verification_report(output_dir)
        verify_2d_slice_pairing(output_dir)

def resize_volume_xy(volume, target_shape, is_mask=False):
    """
    Resizes a 3D volume in the x,y dimensions while keeping z dimension unchanged.
    
    Parameters:
    -----------
    volume : ndarray
        3D volume to resize
    target_shape : tuple
        Target shape for xy dimensions (width, height)
    is_mask : bool
        Whether the volume is a binary mask
    
    Returns:
    --------
    resized_volume : ndarray
        Resized volume with dimensions (target_shape[0], target_shape[1], original_z)
    """
    # Get original z dimension
    original_z = volume.shape[2]
    
    # Calculate zoom factors: x and y will change, z stays the same
    factors = [target_shape[0] / volume.shape[0], 
               target_shape[1] / volume.shape[1], 
               1.0]  # No change in z dimension
    
    # Use nearest neighbor interpolation for masks to preserve binary values
    # Use linear interpolation for image data
    order = 0 if is_mask else 1
    
    # Resize the volume
    resized = zoom(volume, factors, order=order)
    
    # Ensure mask remains binary after interpolation
    if is_mask:
        resized = (resized > 0.5).astype(np.float32)
    
    print(f"Resized from {volume.shape} to {resized.shape}")
    
    return resized

def process_split(subject_pairs, t2_dir, mask_dir, output_dir, target_shape, visualize, root_output_dir):
    """Process a list of subject pairs and convert them to 2D slices with per-subject normalization"""
    for t2_file, mask_file in subject_pairs:
        # Load the NIfTI file
        t2_path = os.path.join(t2_dir, t2_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Get subject ID for naming
        subject_id = t2_file.split('_')[0]
        
        # Load T2 volume
        t2_nifti = nib.load(t2_path)
        t2_data = t2_nifti.get_fdata()
        
        # Load mask from .mat file
        mask_data = sio.loadmat(mask_path)
        # Adjust the key below based on your .mat file structure
        mask_array = mask_data['mask']  # or the appropriate key in your .mat file
        
        # Ensure mask and T2 have same dimensions
        if t2_data.shape != mask_array.shape:
            print(f"Error: Shape mismatch for subject {subject_id}. T2: {t2_data.shape}, Mask: {mask_array.shape}")
            continue
        
        # Resize to target dimensions (160 x 160 x original_slices)
        print(f"Subject {subject_id} - Original shape: {t2_data.shape}")
        resized_t2 = resize_volume_xy(t2_data, target_shape)
        resized_mask = resize_volume_xy(mask_array, target_shape, is_mask=True)
        print(f"Subject {subject_id} - Resized shape: {resized_t2.shape}")
        
        # Per-subject normalization
        # Find non-zero voxels (brain tissue)
        non_zero_mask = resized_t2 > 0
        if np.sum(non_zero_mask) > 0:  # Make sure there's actual data
            # Only normalize non-zero voxels (brain tissue)
            brain_values = resized_t2[non_zero_mask]
            mean_val = np.mean(brain_values)
            std_val = np.std(brain_values)
            
            if std_val > 0:  # Avoid division by zero
                # Z-score normalization for the entire volume (but only affecting brain tissue)
                normalized_t2 = np.zeros_like(resized_t2)
                normalized_t2[non_zero_mask] = (resized_t2[non_zero_mask] - mean_val) / std_val
            else:
                # If std is zero, just use the original data
                normalized_t2 = resized_t2
                print(f"Warning: Zero standard deviation for subject {subject_id}, skipping normalization")
        else:
            # If there's no data, use the original
            normalized_t2 = resized_t2
            print(f"Warning: No brain tissue found for subject {subject_id}")
        
        # Visualize a few slices to verify alignment, resizing, and normalization
        if visualize:
            visualize_subject_pairing(subject_id, resized_t2, normalized_t2, resized_mask, root_output_dir)
        
        # Extract 2D slices along the axial plane (adjust axis as needed)
        for slice_idx in range(normalized_t2.shape[2]):  # Assuming axial slices (z-axis)
            t2_slice = normalized_t2[:, :, slice_idx]
            mask_slice = resized_mask[:, :, slice_idx]
            
            # Skip empty slices (optional)
            if np.sum(t2_slice) < 100 or np.sum(mask_slice) < 10:
                continue
            
            # No additional slice-wise normalization needed since we did per-subject normalization
            
            # Save slices with subject ID and slice index in filename to maintain traceability
            slice_name = f"{subject_id}_slice_{slice_idx:03d}"
            
            # Save as numpy arrays (can be changed to PNG or other formats)
            np.save(os.path.join(output_dir, 'images', f"{slice_name}.npy"), t2_slice)  # Save normalized data
            np.save(os.path.join(output_dir, 'masks', f"{slice_name}.npy"), mask_slice)

def visualize_subject_pairing(subject_id, original_t2, normalized_t2, mask_array, output_dir):
    """Visualize a few key slices to verify T2 and mask alignment and normalization effects"""
    # Create output directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select slices with significant mask content
    non_empty_slices = []
    for i in range(mask_array.shape[2]):
        if np.sum(mask_array[:,:,i]) > 50:
            non_empty_slices.append(i)
    
    # Select up to 3 slices (beginning, middle, end of non-empty slices)
    if len(non_empty_slices) > 0:
        if len(non_empty_slices) >= 3:
            viz_slices = [non_empty_slices[0], 
                         non_empty_slices[len(non_empty_slices)//2], 
                         non_empty_slices[-1]]
        else:
            viz_slices = non_empty_slices
        
        # Create figure with rows for each slice and 4 columns
        # (original T2, normalized T2, mask, overlay)
        fig, axes = plt.subplots(len(viz_slices), 4, figsize=(20, 5*len(viz_slices)))
        if len(viz_slices) == 1:
            axes = axes.reshape(1, -1)  # Ensure axes is 2D for single slice case
        
        for i, slice_idx in enumerate(viz_slices):
            # Get original T2 slice and normalize for display
            orig_t2_slice = original_t2[:, :, slice_idx]
            orig_t2_norm = (orig_t2_slice - np.min(orig_t2_slice)) / (np.max(orig_t2_slice) - np.min(orig_t2_slice) + 1e-8)
            
            # Get normalized T2 slice and rescale for display
            norm_t2_slice = normalized_t2[:, :, slice_idx]
            # Rescale to [0,1] range for visualization
            norm_t2_vis = (norm_t2_slice - np.min(norm_t2_slice)) / (np.max(norm_t2_slice) - np.min(norm_t2_slice) + 1e-8)
            
            # Get mask slice
            mask_slice = mask_array[:, :, slice_idx]
            
            # Plot original T2
            axes[i, 0].imshow(orig_t2_norm, cmap='gray')
            axes[i, 0].set_title(f'Original T2 - Subject {subject_id}, Slice {slice_idx}')
            axes[i, 0].axis('off')
            
            # Plot normalized T2
            axes[i, 1].imshow(norm_t2_vis, cmap='gray')
            axes[i, 1].set_title(f'Normalized T2 - Subject {subject_id}, Slice {slice_idx}')
            axes[i, 1].axis('off')
            
            # Plot mask
            axes[i, 2].imshow(mask_slice, cmap='viridis')
            axes[i, 2].set_title(f'Mask - Subject {subject_id}, Slice {slice_idx}')
            axes[i, 2].axis('off')
            
            # Create overlay of normalized T2 with mask
            overlay = np.zeros((norm_t2_vis.shape[0], norm_t2_vis.shape[1], 3))
            overlay[:, :, 0] = np.where(mask_slice > 0, 1, 0)  # Red channel for mask
            overlay[:, :, 1] = np.where(mask_slice > 0, 0, norm_t2_vis)  # Green channel for T2
            overlay[:, :, 2] = np.where(mask_slice > 0, 0, norm_t2_vis)  # Blue channel for T2
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Overlay - Subject {subject_id}, Slice {slice_idx}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{subject_id}_verification.png'))
        plt.close(fig)

def create_verification_report(output_dir):
    """Create a verification HTML report for easy viewing of all visualizations"""
    vis_dir = os.path.join(output_dir, 'visualizations')
    vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png') and 'slice' not in f])
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Pairing and Normalization Verification</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            h1 { color: #333; }
            .subject { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .subject h2 { margin-top: 0; }
            img { max-width: 100%; border: 1px solid #eee; }
            .explanation { color: #555; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>T2-Mask Pairing and Normalization Verification Report</h1>
        <div class="explanation">
            <p>This report shows the alignment between T2 images and their corresponding masks for each subject.</p>
            <p>For each subject, you can see:</p>
            <ul>
                <li><strong>Original T2:</strong> The resized T2-weighted image (160×160) before normalization</li>
                <li><strong>Normalized T2:</strong> The T2 image after per-subject normalization</li>
                <li><strong>Mask:</strong> The ground truth segmentation mask (also resized to 160×160)</li>
                <li><strong>Overlay:</strong> Normalized T2 with mask overlay (red) to verify alignment</li>
            </ul>
        </div>
    """
    
    for vis_file in vis_files:
        subject_id = vis_file.split('_')[0]
        html_content += f"""
        <div class="subject">
            <h2>Subject: {subject_id}</h2>
            <img src="{vis_file}" alt="Verification for {subject_id}" />
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(os.path.join(vis_dir, 'verification_report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Verification report created at {os.path.join(vis_dir, 'verification_report.html')}")

def verify_2d_slice_pairing(output_dir, num_samples=5):
    """Verify that 2D slices are properly paired after processing"""
    # Sample a few slices from each split to verify
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(output_dir, split, 'images')
        masks_dir = os.path.join(output_dir, split, 'masks')
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            continue
            
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        if len(image_files) == 0:
            continue
            
        # Sample up to num_samples files
        sample_indices = np.linspace(0, len(image_files) - 1, min(num_samples, len(image_files)), dtype=int)
        sample_files = [image_files[i] for i in sample_indices]
        
        fig, axes = plt.subplots(len(sample_files), 3, figsize=(15, 5*len(sample_files)))
        if len(sample_files) == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_file in enumerate(sample_files):
            # Get corresponding mask
            mask_file = img_file
            
            # Load files
            img = np.load(os.path.join(images_dir, img_file))
            mask = np.load(os.path.join(masks_dir, mask_file))
            
            # Rescale image for visualization
            img_vis = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            
            # Plot
            axes[i, 0].imshow(img_vis, cmap='gray')
            axes[i, 0].set_title(f'Image: {img_file}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='viridis')
            axes[i, 1].set_title(f'Mask: {mask_file}')
            axes[i, 1].axis('off')
            
            # Create overlay
            overlay = np.zeros((img.shape[0], img.shape[1], 3))
            overlay[:, :, 0] = np.where(mask > 0, 1, 0)  # Red channel for mask
            overlay[:, :, 1] = np.where(mask > 0, 0, img_vis)  # Green channel for image
            overlay[:, :, 2] = np.where(mask > 0, 0, img_vis)  # Blue channel for image
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{split}_2d_verification.png'))
        plt.close(fig)
    
    print("2D slice pairing verification complete")

# Example usage
if __name__ == "__main__":
    prepare_dataset(
        t2_dir="/home/smooi/Desktop/toast/data/toast_pipe_data/mask/t2stack",
        mask_dir="/home/smooi/Desktop/toast/data/toast_pipe_data/mask",
        output_dir="/home/smooi/Desktop/toast/data/toast_pipe_data",
        target_shape=(160, 160),  # New target dimensions for width and height
        visualize=True
    )