"""
dataset.py

Major Implementations: TODO

Author: Hailin Liang 
Date: 2025/03-07
"""

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



class Dataset():
    """
    Custom dataset to load study subject's nifti image and corresponding manual binary mask and stiffness probability matrix.
    """
    
    def __init__(self, nifit_dir, mask_dir, stiff_prob_dir):
        """
        Args:
            nifti_dir (str): Directory containing subject's brain image in nifti format.
            mask_dir (str): Directory containing manual masks in nifti format.
            stiff_prob_dir (str): Directory containing the probability matrix generated through strict defined threshold.
        """
        self.nifti_dir = nifit_dir
        self.mask_dir = mask_dir
        self.stiff_prob_dir = stiff_prob_dir
        
        
    def _sort(nifti_filename):
        """
        Create datasets separately for GE and SIEMENS to allow for even number of dataset from both devices to feed into validation and training. 
        
        Naming convention for all dataset: 
        [Device]_[StudyID]_[SUBJECTID]_[data_type].nifti / [Device]_[StudyID]_[SUBJECTID]_[data_type].mat
        E.g. 
        1. Subject's Brain Image Nifti: GE_EA_1046A_brain.nifti 
        2. Subject's Manual Mask Nifti: GE_EA_1046A_mask.nifti 
        3. Subject's NLI Prob Matrix: GE_EA_1046A_stiff.mat
        
        """