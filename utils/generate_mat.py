### This module is used for generating predictions from 

import os
import torch
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from torch.utils.data import DataLoader

def save_predictions_as_mat(
    model: torch.nn.Module,
    test_loader: DataLoader,
    output_dir: str,
    device: str = 'cpu',
    num_samples: int = 1
):
    """
    Runs a model on a test set and saves the binary prediction masks as .mat files.

    This function can handle both deterministic (num_samples=1) and probabilistic
    (num_samples > 1) models.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test set. It should yield batches with
                     'image' and 'subject_name'.
        output_dir: The directory where .mat files will be saved.
        device: The device to run the model on ('cuda' or 'cpu').
        num_samples: The number of prediction samples to generate per subject.
                     Set to 1 for deterministic models.
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the model to evaluation mode and move to the specified device
    model.to(device)
    model.eval()
    
    # this will trigger the probabilistic unet if the number of samples > 1 
    is_probabilistic = num_samples > 1
    mode_str = "Probabilistic" if is_probabilistic else "Deterministic"
    
    print(f" Running {mode_str} model to generate prediction masks...")
    print(f"   Saving .mat files to: {output_dir}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing Subjects"):
            image = batch['image'].to(device)
            # Access the subject name from the batch
            subject_name = batch['subject_name'][0]

            # The 'if' statement below is the trigger for the two examples.
            if is_probabilistic:
                # --- This is Example 2 ---
                # Loop to generate multiple samples for a probabilistic model.
                for i in range(num_samples):
                    logits = model(image)
                    pred_mask = (torch.sigmoid(logits) > 0.5).float()
                    pred_np = pred_mask.cpu().numpy().squeeze()
                    
                    # Create the filename in the format: {subjectname}_V1.mat, {subjectname}_V2.mat, etc.
                    filename = f"{subject_name}_V{i+1}.mat"
                    mat_path = os.path.join(output_dir, filename)
                    sio.savemat(mat_path, {"prediction_mask": pred_np})
            else:
                # --- This is Example 1 ---
                # Generate a single prediction for a deterministic model.
                logits = model(image)
                pred_mask = (torch.sigmoid(logits) > 0.5).float()
                pred_np = pred_mask.cpu().numpy().squeeze()

                # Create the filename in the format: {subjectname}_D.mat
                filename = f"{subject_name}_D.mat"
                mat_path = os.path.join(output_dir, filename)
                sio.savemat(mat_path, {"prediction_mask": pred_np})

    print(f"\n✅ Prediction generation complete.")

    print(f"\n✅ Prediction generation complete.")
    print(f"   All .mat files have been saved in '{output_dir}'.")


# --- Example Usage ---
# To use this script, you would uncomment and fill out the following section.
# if __name__ == '__main__':
#     from your_project.model import YourModelClass  # CHANGE: Import your model
#     from your_project.dataset import YourDatasetClass # CHANGE: Import your dataset
#
#     # 1. Set parameters
#     MODEL_PATH = "path/to/your/best_model.pt"
#     DATA_DIR = "path/to/your/test_data"
#     OUTPUT_DIR = "mat_predictions"
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     NUM_SAMPLES = 6 # Set to 1 for deterministic, >1 for probabilistic
#
#     # 2. Load your trained model
#     # model = YourModelClass(...) # CHANGE: Instantiate your model architecture
#     # model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     # print(f"Model loaded from {MODEL_PATH}")
#
#     # 3. Create your test DataLoader
#     # test_dataset = YourDatasetClass(data_dir=DATA_DIR, mode='test')
#     # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     # print(f"Loaded {len(test_dataset)} test subjects.")
#
#     # 4. Run the prediction and saving process
#     # save_predictions_as_mat(
#     #     model=model,
#     #     test_loader=test_loader,
#     #     output_dir=OUTPUT_DIR,
#     #     device=DEVICE,
#     #     num_samples=NUM_SAMPLES
#     # )