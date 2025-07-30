### This script will prepare the data for training and testing the model
# Author: Hailin Liang
# Date: 2025/07/01 

# This script will serve the following purposes:
# 1. Load the data from the specified directory.
# 2. Preprocess the data (e.g., normalization, resizing).
# 3. Save the preprocessed data to a specified output directory.
# 4. Handle both training and testing datasets.
# 5. Ensure the data is in a format suitable for model training (e.g., PyTorch tensors).
# when the mode is set to 'train', it will split the data into val, test, train and put them into respective loaders 
# 6. When the mode is set to 'pure_infer', it will load the data from the specified directory and return correct data loaders 

### This script will prepare the data for training and testing the model
# Author: Hailin Liang
# Date: 2025/07/01 
# Fixed version to handle original shapes properly

# importing 
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import os
import glob
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple, List, Dict, Optional, Union
import argparse


class ManualPreprocessor:
    """Manual preprocessing using SciPy for full control"""
    
    def __init__(self, target_spacing=(1.5, 1.5, 1.5), target_size=(128, 128, 64)):
        self.target_spacing = target_spacing
        self.target_size = target_size
    
    def load_nifti(self, filepath):
        """Load NIfTI file and extract data + metadata"""
        nii = nib.load(filepath)
        data = nii.get_fdata().astype(np.float32)
        
        # Get voxel spacing
        header = nii.header
        spacing = header.get_zooms()[:3]  # Get x, y, z spacing
        
        return data, spacing, nii.affine, header
    
    def resample_volume(self, volume, original_spacing, target_spacing, order=1):
        """
        Resample volume to target spacing using scipy
        order: 0=nearest (for labels), 1=linear (for images), 3=cubic
        """
        # Calculate zoom factors
        zoom_factors = [orig/target for orig, target in zip(original_spacing, target_spacing)]
        
        # Resample
        resampled = ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)
        
        return resampled
    
    def resize_volume(self, volume, target_size, order=1):
        """Resize volume to exact target size"""
        current_size = volume.shape
        zoom_factors = [target/current for target, current in zip(target_size, current_size)]
        
        resized = ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)
        
        return resized
    
    def normalize_intensity(self, volume, method='zscore', clip_percentiles=(1, 99)):
        """Normalize volume intensity"""
        if method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            mean = np.mean(volume[volume > 0])  # Only non-zero voxels
            std = np.std(volume[volume > 0])
            normalized = (volume - mean) / (std + 1e-8)
            
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            vmin, vmax = np.percentile(volume[volume > 0], clip_percentiles)
            normalized = np.clip(volume, vmin, vmax)
            normalized = (normalized - vmin) / (vmax - vmin + 1e-8)
            
        elif method == 'percentile':
            # Percentile-based clipping and normalization
            vmin, vmax = np.percentile(volume, clip_percentiles)
            normalized = np.clip(volume, vmin, vmax)
            normalized = (normalized - vmin) / (vmax - vmin + 1e-8)
            
        return normalized
    
    def process_pair(self, image_path, label_path):
        """Process image-label pair with guaranteed alignment"""
        
        # Load both files
        image_data, image_spacing, image_affine, image_header = self.load_nifti(image_path)
        label_data, label_spacing, label_affine, label_header = self.load_nifti(label_path)
        
        print(f"Original image shape: {image_data.shape}, spacing: {image_spacing}")
        print(f"Original label shape: {label_data.shape}, spacing: {label_spacing}")
        
        # Store original shape for later restoration
        original_shape = image_data.shape
        
        # Step 1: Resample image to target spacing
        image_resampled = self.resample_volume(image_data, image_spacing, self.target_spacing, order=1)
        
        # Step 2: Resample label to match image's new spacing and size exactly
        # First resample label to target spacing
        label_resampled = self.resample_volume(label_data, label_spacing, self.target_spacing, order=0)
        
        # Then resize label to exactly match image size
        if label_resampled.shape != image_resampled.shape:
            label_resampled = self.resize_volume(label_resampled, image_resampled.shape, order=0)
        
        print(f"After spacing - Image: {image_resampled.shape}, Label: {label_resampled.shape}")
        
        # Step 3: Resize both to target size (optional)
        if self.target_size:
            image_final = self.resize_volume(image_resampled, self.target_size, order=1)
            label_final = self.resize_volume(label_resampled, self.target_size, order=0)
        else:
            image_final = image_resampled
            label_final = label_resampled
        
        # Step 4: Normalize image intensity
        image_normalized = self.normalize_intensity(image_final, method='zscore')
        
        # Step 5: Ensure label is binary
        label_binary = (label_final > 0.5).astype(np.float32)
        
        print(f"Final - Image: {image_final.shape}, Label: {label_final.shape}")
        print(f"Label coverage: {(label_binary > 0).sum() / label_binary.size * 100:.2f}%")
        
        return image_normalized, label_binary, original_shape


class MedicalDataset(Dataset):
    """Custom dataset that properly handles subject names and original shapes"""
    
    def __init__(self, images, labels, subject_names, original_shapes):
        self.images = images
        self.labels = labels
        self.subject_names = subject_names
        # Ensure original shapes are stored as proper tuples
        self.original_shapes = []
        for shape in original_shapes:
            if isinstance(shape, (tuple, list)) and len(shape) == 3:
                self.original_shapes.append(tuple(shape))
            elif hasattr(shape, 'shape') and len(shape.shape) == 1 and len(shape) == 3:
                # Handle numpy arrays or tensors
                self.original_shapes.append(tuple(shape.tolist() if hasattr(shape, 'tolist') else shape))
            else:
                print(f"⚠️  Warning: Invalid shape {shape}, using default (160, 160, 80)")
                self.original_shapes.append((160, 160, 80))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
            'subject_name': self.subject_names[idx],
            'original_shape': self.original_shapes[idx]  # Return as tuple
        }


def custom_collate_fn(batch):
    """Custom collate function to properly handle batching of metadata"""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    subject_names = [item['subject_name'] for item in batch]
    original_shapes = [item['original_shape'] for item in batch]
    
    return {
        'image': images,
        'label': labels,
        'subject_name': subject_names,
        'original_shape': original_shapes
    }


def inspect_mask_files(mask_dir: str) -> None:
    """Utility function to inspect mask files and their contents"""
    print(" INSPECTING MASK FILES")
    print("=" * 50)
    
    nii_files = glob.glob(os.path.join(mask_dir, "*.nii*"))
    mat_files = glob.glob(os.path.join(mask_dir, "*.mat"))
    
    print(f"Found {len(nii_files)} NIfTI files and {len(mat_files)} MAT files")
    
    # Inspect a few NIfTI files
    if nii_files:
        print(f"\n Sample NIfTI files:")
        for i, nii_file in enumerate(nii_files[:3]):
            try:
                nii = nib.load(nii_file)
                data = nii.get_fdata()
                print(f"  {os.path.basename(nii_file)}: shape={data.shape}, "
                      f"range=[{data.min():.2f}, {data.max():.2f}], "
                      f"non-zero={np.count_nonzero(data)}")
            except Exception as e:
                print(f"  {os.path.basename(nii_file)}: Error loading - {e}")
    
    # Inspect a few MAT files
    if mat_files:
        print(f"\n Sample MAT files:")
        for i, mat_file in enumerate(mat_files[:3]):
            try:
                data_dict = loadmat(mat_file)
                arrays = [(k, v.shape, v.dtype) for k, v in data_dict.items() 
                         if isinstance(v, np.ndarray) and not k.startswith('__')]
                print(f"  {os.path.basename(mat_file)}:")
                for name, shape, dtype in arrays:
                    print(f"    - {name}: shape={shape}, dtype={dtype}")
            except Exception as e:
                print(f"  {os.path.basename(mat_file)}: Error loading - {e}")


def mat_to_nii(mat_path: str, ref_path: str) -> str:
    """Convert .mat file to temporary .nii.gz using reference NIfTI file for affine/header"""
    try:
        ref = nib.load(ref_path)
        data_dict = loadmat(mat_path)
        
        # Find the 3D array in the .mat file
        candidate_arrays = []
        for key, value in data_dict.items():
            if (isinstance(value, np.ndarray) and 
                value.ndim == 3 and 
                not key.startswith('__')):
                candidate_arrays.append((key, value))
        
        if not candidate_arrays:
            raise ValueError(f"No 3D arrays found in {mat_path}")
        
        # Use the largest 3D array (most likely to be the mask)
        if len(candidate_arrays) == 1:
            arr = candidate_arrays[0][1]
            print(f"  Using array '{candidate_arrays[0][0]}' from {os.path.basename(mat_path)}")
        else:
            arr = max(candidate_arrays, key=lambda x: x[1].size)[1]
            selected_key = max(candidate_arrays, key=lambda x: x[1].size)[0]
            print(f"  Multiple arrays found, using largest '{selected_key}' from {os.path.basename(mat_path)}")
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_filename = os.path.basename(mat_path).replace(".mat", ".nii.gz")
        tmp_path = os.path.join(temp_dir, temp_filename)
        
        # Convert to proper data type and save
        arr_float32 = arr.astype(np.float32)
        nii_img = nib.Nifti1Image(arr_float32, ref.affine, ref.header)
        nib.save(nii_img, tmp_path)
        
        print(f"  Converted {os.path.basename(mat_path)} -> {temp_filename} (shape: {arr.shape})")
        return tmp_path
        
    except Exception as e:
        print(f" Error converting {mat_path}: {str(e)}")
        raise


def load_dataset_files(t2_dir: str, mask_dir: str) -> List[Dict[str, str]]:
    """Load and organize dataset file paths with robust mask file handling"""
    t2_files = sorted(glob.glob(os.path.join(t2_dir, "*.nii*")))
    
    # Get all possible mask files
    nii_mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii*")))
    mat_mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.mat")))
    
    print(f"Found {len(t2_files)} T2 files")
    print(f"Found {len(nii_mask_files)} NIfTI mask files")
    print(f"Found {len(mat_mask_files)} MAT mask files")
    
    # Create a mapping for each T2 file to find its corresponding mask
    dataset = []
    
    for t2_file in t2_files:
        t2_basename = os.path.basename(t2_file)
        subject_id = t2_basename.split('.')[0]  # Remove extension
        
        # Try to find corresponding mask file
        mask_file = None
        mask_type = None
        
        # First try to find exact match in NIfTI masks
        for nii_mask in nii_mask_files:
            mask_basename = os.path.basename(nii_mask)
            mask_subject_id = mask_basename.split('.')[0]
            
            if subject_id == mask_subject_id or subject_id in mask_basename:
                mask_file = nii_mask
                mask_type = "nii"
                break
        
        # If no NIfTI mask found, try MAT files
        if mask_file is None:
            for mat_mask in mat_mask_files:
                mask_basename = os.path.basename(mat_mask)
                mask_subject_id = mask_basename.split('.')[0]
                
                if subject_id == mask_subject_id or subject_id in mask_basename:
                    mask_file = mat_mask
                    mask_type = "mat"
                    break
        
        # If still no mask found, try more flexible matching
        if mask_file is None:
            for mask_list, mtype in [(nii_mask_files, "nii"), (mat_mask_files, "mat")]:
                for mask in mask_list:
                    mask_basename = os.path.basename(mask).lower()
                    subject_lower = subject_id.lower()
                    
                    if subject_lower in mask_basename or any(part in mask_basename for part in subject_lower.split('_')):
                        mask_file = mask
                        mask_type = mtype
                        break
                if mask_file:
                    break
        
        if mask_file:
            dataset.append({
                "image": t2_file,
                "label": mask_file,
                "subject_id": subject_id,
                "mask_type": mask_type
            })
            print(f"✓ Paired: {subject_id} -> {os.path.basename(mask_file)} ({mask_type})")
        else:
            print(f" No mask found for: {subject_id}")
    
    print(f"\n Summary:")
    print(f"  Successfully paired: {len(dataset)} image-mask pairs")
    print(f"  Missing masks: {len(t2_files) - len(dataset)}")
    
    # Count mask types
    nii_count = sum(1 for item in dataset if item.get('mask_type') == 'nii')
    mat_count = sum(1 for item in dataset if item.get('mask_type') == 'mat')
    print(f"  NIfTI masks: {nii_count}")
    print(f"  MAT masks: {mat_count}")
    
    return dataset


def preprocess_split(file_list: List[Dict], preprocessor: ManualPreprocessor) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple]]:
    """Preprocess a split of the dataset with enhanced mask handling"""
    imgs, lbls, subject_names, original_shapes = [], [], [], []
    
    for i, rec in enumerate(file_list):
        print(f"\nProcessing {i+1}/{len(file_list)}: {rec['subject_id']}")
        
        label_path = rec["label"]
        mask_type = rec.get("mask_type", "unknown")
        
        # Handle .mat files
        if label_path.endswith(".mat"):
            print(f"  Converting MAT file: {os.path.basename(label_path)}")
            try:
                label_path = mat_to_nii(label_path, rec["image"])
            except Exception as e:
                print(f" Failed to convert MAT file: {e}")
                continue
        else:
            print(f"  Using NIfTI file: {os.path.basename(label_path)}")
        
        try:
            # Process the image-label pair
            img, lbl, orig_shape = preprocessor.process_pair(rec["image"], label_path)
            
            # Validate the processed data
            if img.shape != lbl.shape:
                print(f"Shape mismatch for {rec['subject_id']}: img={img.shape}, lbl={lbl.shape}")
                continue
                
            # Check if label has any positive values
            label_coverage = (lbl > 0.5).sum() / lbl.size * 100
            if label_coverage == 0:
                print(f"  Warning: {rec['subject_id']} has no positive labels (empty mask)")
            elif label_coverage > 90:
                print(f" Warning: {rec['subject_id']} has {label_coverage:.1f}% label coverage (possible issue)")
            
            imgs.append(img)
            lbls.append(lbl.astype(np.float32))
            subject_names.append(rec["subject_id"])
            # Ensure original shape is stored as proper tuple
            original_shapes.append(tuple(orig_shape))
                
            print(f" Successfully processed {rec['subject_id']} (coverage: {label_coverage:.2f}%)")
            
        except Exception as e:
            print(f" Error processing {rec['subject_id']}: {str(e)}")
            continue
    
    if not imgs:
        raise ValueError("No images were successfully processed!")
    
    print(f"\n📊 Preprocessing complete:")
    print(f"  Successfully processed: {len(imgs)} out of {len(file_list)} files")
    print(f"  Final shapes: {imgs[0].shape}")
    
    return np.stack(imgs), np.stack(lbls), subject_names, original_shapes


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    """Ensure array has channel dimension for MONAI (N, C, D, H, W)"""
    if x.ndim == 4:
        return x[:, None, ...]  # Insert channel dim at position 1
    elif x.ndim == 5:
        return x  # Already has channel dimension
    else:
        raise ValueError(f"Unsupported array shape {x.shape}, expected 4D or 5D array.")


def create_data_loaders(
    mode: str,
    t2_dir: str,
    mask_dir: str,
    train_txt: str,
    val_txt: str,
    test_txt: str,
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    target_size: Tuple[int, int, int]      = (128, 128, 64),
    batch_size: int                        = 4,
    num_workers: int                       = 8,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from fixed split files.
    """
    # 1) Read base‐IDs from each split file
    def file_parser(path: str):
        with open(path, 'r') as f:
            return [ln.strip() for ln in f if ln.strip()]

    train_ids = file_parser(train_txt)
    val_ids   = file_parser(val_txt)
    test_ids  = file_parser(test_txt)

    # 2) Load all image‐mask pairs
    pre       = ManualPreprocessor(target_spacing, target_size)
    all_recs  = load_dataset_files(t2_dir, mask_dir)

    # 3) Match any record whose subject_id startswith one of the base‐IDs
    def match_sid(sid: str, id_list: list):
        return any(sid.startswith(i) for i in id_list)

    splits = {'train': [], 'val': [], 'test': []}
    for rec in all_recs:
        sid = rec['subject_id']
        if   match_sid(sid, train_ids): splits['train'].append(rec)
        elif match_sid(sid,   val_ids): splits['val'].append(rec)
        elif match_sid(sid,  test_ids): splits['test'].append(rec)
        else:
            print(f"{sid} not in any split, skipping")

    # 4) Preprocess each split and build its DataLoader
    loaders = {}
    for split in ('train','val','test'):
        recs = splits[split]
        imgs, lbls, names, shapes = preprocess_split(recs, pre)

        # channel‐first + to‐tensor
        X = ensure_channel_first(imgs);  Y = ensure_channel_first(lbls)
        Xt = torch.from_numpy(X).float();   Yt = torch.from_numpy(Y).float()

        ds = MedicalDataset(Xt, Yt, names, shapes)
        shuffle = (split == 'train')
        bs      = batch_size if shuffle else 1
        nw      = num_workers if shuffle else max(1, num_workers // 2)

        loaders[split] = DataLoader(
            ds,
            batch_size   = bs,
            shuffle      = shuffle,
            num_workers  = nw,
            pin_memory   = True,
            collate_fn   = custom_collate_fn,
        )

    return loaders


def save_preprocessed_data(data_loaders: Dict[str, DataLoader], output_dir: str):
    """Save preprocessed data to disk with proper shape handling"""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, loader in data_loaders.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Extract data from loader
        images, labels, names, shapes = [], [], [], []
        
        for batch in loader:
            # Extract tensors and metadata
            batch_images = batch['image']
            batch_labels = batch['label']
            batch_names = batch['subject_name']
            batch_shapes = batch['original_shape']
            
            # Handle images and labels
            if batch_images.dim() == 5:  # Batched: (B, C, D, H, W)
                for i in range(batch_images.size(0)):
                    images.append(batch_images[i])
                    labels.append(batch_labels[i])
            else:  # Single sample: (C, D, H, W)
                images.append(batch_images)
                labels.append(batch_labels)
            
            # Handle names and shapes (already properly handled by custom_collate_fn)
            names.extend(batch_names)
            shapes.extend(batch_shapes)
        
        # Stack tensors
        images = torch.stack(images)
        labels = torch.stack(labels)
        
        # Save tensors
        torch.save(images, os.path.join(split_dir, 'images.pt'))
        torch.save(labels, os.path.join(split_dir, 'labels.pt'))
        
        # Save metadata
        metadata = {
            'subject_names': names,
            'original_shapes': shapes  # These are now proper 3-tuples
        }
        torch.save(metadata, os.path.join(split_dir, 'metadata.pt'))
        
        print(f"Saved {split_name} data: {images.shape[0]} samples to {split_dir}")
        print(f"  Sample shapes: {shapes[:3]}...")  # Show first 3 shapes for verification


## Need improvement below 
def main():
    parser = argparse.ArgumentParser(description='Prepare medical imaging data for training/inference')
    parser.add_argument('--mode', type=str, choices=['train', 'pure_infer', 'inspect'], required=True,
                       help='Mode: train (split data), pure_infer (all data for inference), or inspect (examine mask files)')
    parser.add_argument('--t2_dir', type=str, required=True,
                       help='Directory containing T2 images')
    parser.add_argument('--mask_dir', type=str, required=True,
                       help='Directory containing masks')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_data',
                       help='Output directory for preprocessed data')
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.5, 1.5, 1.5],
                       help='Target voxel spacing (x, y, z)')
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 64],
                       help='Target volume size (x, y, z)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of workers for data loading')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='Proportion for test set')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Proportion for validation set')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save_data', action='store_true',
                       help='Save preprocessed data to disk')
    
    args = parser.parse_args()
    
    # Inspect mode - just examine the mask files
    if args.mode == 'inspect':
        inspect_mask_files(args.mask_dir)
        return None
    
    # Create data loaders
    data_loaders = create_data_loaders(
    mode=args.mode,
    t2_dir=args.t2_dir,
    mask_dir=args.mask_dir,
    train_txt=args.train_txt,
    val_txt=args.val_txt,
    test_txt=args.test_txt,
    target_spacing=tuple(args.target_spacing),
    target_size=tuple(args.target_size),
    batch_size=args.batch_size,
    num_workers=args.num_workers
)
    # Print information about created loaders
    for split_name, loader in data_loaders.items():
        print(f"{split_name.capitalize()} loader: {len(loader)} batches, {len(loader.dataset)} samples")
        
        # Show example batch
        example_batch = next(iter(loader))
        print(f"  - Image shape: {example_batch['image'].shape}")
        print(f"  - Label shape: {example_batch['label'].shape}")
        print(f"  - Subject names: {example_batch['subject_name'][:3]}...")
        print(f"  - Original shapes: {example_batch['original_shape'][:3]}...")
    
    # Save preprocessed data if requested
    if args.save_data:
        save_preprocessed_data(data_loaders, args.output_dir)
    
    return data_loaders


if __name__ == "__main__":
    # Enable CUDA optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    data_loaders = main()