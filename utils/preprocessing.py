import random
import os
import json
import cv2
import time
import pyvips
def split_dataset(dataset, train_ratio=0.7 , val_ratio = 0.2):
    
    class_indices = {}
    train_indices = []
    test_indices = []
    val_indices = []

    # Organize indices by class
    for idx, (_, label) in enumerate(dataset):
     
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Split each class
    for label, indices in class_indices.items():
        random.shuffle(indices)  # Shuffle the indices
        train_size = int(len(indices) * train_ratio)
        val_size = int(len(indices) * val_ratio)
        train_indices.extend(indices[:train_size])
        test_indices.extend(indices[train_size:train_size+val_size])
        val_indices.extend(indices[train_size+val_size:])

    return train_indices, test_indices , val_indices

def get_next_experiment_number(base_path="experiments"):
    # Get the list of existing experiment folders
    existing_experiments = [d for d in os.listdir(base_path) if d.startswith("experiment")]
    
    # Extract the number from the folder names, e.g., 'experiment_1' -> 1
    if existing_experiments:
        experiment_numbers = [int(exp.split('_')[1]) for exp in existing_experiments]
        next_number = max(experiment_numbers) + 1
    else:
        next_number = 1
    
    return next_number

def create_experiment_folder(base_path="experiments", config_data=None):
    # Ensure the base directory exists
    os.makedirs(base_path, exist_ok=True)

    # Get the next available experiment number
    experiment_number = get_next_experiment_number(base_path)

    # Create a new experiment folder name like 'experiment_1'
    experiment_folder = os.path.join(base_path, f"experiment_{experiment_number}")

    # Create subdirectories for the experiment
    subdirs = ["checkpoints", "logs", "data" , "best"]
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_folder, subdir), exist_ok=True)

    # Create a config file if data is provided
    if config_data is not None:
        config_path = os.path.join(experiment_folder, "config.json")
        with open(config_path, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)

    print(f"Experiment folder created at: {experiment_folder}")
    return experiment_folder

def rezise_image(image , lib_ = "vipps"):
    desired_width = 1280
    desired_height = 720
    st_ = time.time()
    if lib_ == "vipps":
        image = pyvips.Image.new_from_array(image)
        image = image.resize(desired_width / image.width, vscale = desired_height / image.height)
        image = image.numpy()
    elif lib_ == "cv2":
        image = cv2.resize(image , (1280 , 720))
    en_ = time.time()
    print("Processing Timelapse :{:.2f}".format(en_ - st_))
    return image
