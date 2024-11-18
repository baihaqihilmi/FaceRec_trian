# config.py

import os
import argparse
import torch
import os.path as osp
class Config:
    # Hardware
    ROOT_DIR = "/media/baihaqi/Data_Temp/EdgeFace/edgeface_exo_2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "/media/baihaqi/Data_Temp/EdgeFace/edgeface_exo_2/models/checkpoints"

    DATA_DIR = osp.join(ROOT_DIR, "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VALID_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    SAVE_DIR = os.getenv("SAVE_DIR", "saved_modils")
    LOG_DIR = os.getenv("LOG_DIR", "logs")

    # Data Processing
    IMAGE_SIZE = (112,112)

    TRAIN_RATIO = float(0.7)
    TEST_RATIO = float(0.2)
    VAL_RATIO = float(0.1)

    TRAIN_BATCH_SIZE = int(32)
    VAL_BATCH_SIZE = int(32)
    NUM_WORKERS = int( 4)
    # Training Paramters
    OPTIMIZER = str("SGD"
)
    # Model Hyperparameters
    NUM_CLASSES = len(os.listdir(DATA_DIR))
    MODEL_NAME = "edgeface_xs_gamma_06"
    CHECKPOINT_MAME = "edgeface_xs_gamma_06.pt"
    CHECKPOINT =  osp.join(CHECKPOINT_DIR, CHECKPOINT_MAME)
    PRETRAINED = bool(1)
    
    # Training Hyperparameters
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-3))
    MOMENTUM = float(os.getenv("MOMENTUM", 0.9))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 1000))
    LR_STEP_SIZE = int(os.getenv("LR_STEP_SIZE", 7))
    LR_GAMMA = float(os.getenv("LR_GAMMA", 0.1))
    # Loss Hyperparamater
    LOSS = os.getenv("LOSS", "cross_entropy")
    SCALE = float(os.getenv("SCALE", 1.0))
    MARGIN = float(os.getenv("MARGIN", 0.5))
    EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", 512))
    # Checkpointing
    CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 100))
    FINE_TUNING = bool(int(os.getenv("FINE_TUNE", 0)))
    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def parse_args():
        """Parses arguments to override default configurations."""
        parser = argparse.ArgumentParser(description="Training Configuration")
        # Hardware
        parser.add_argument("--device", default=Config.DEVICE, type=str, help="Device to use for training")
        # Paths
        parser.add_argument("--data_dir", default=Config.DATA_DIR, type=str, help="Root directory for data")
        parser.add_argument("--save_dir", default=Config.SAVE_DIR, type=str, help="Directory to save models")
        parser.add_argument("--log_dir", default=Config.LOG_DIR, type=str, help="Directory for logs")

        # Data Processin

        parser.add_argument("--train batch_size", default=Config.TRAIN_BATCH_SIZE, type=int, help="Batch size for training")
        parser.add_argument("--val_batch_size", default=Config.VAL_BATCH_SIZE, type=int, help="Batch size for validation")
        parser.add_argument("--num_workers", default=Config.NUM_WORKERS, type=int, help="Number of workers for data loading")

        parser.add_argument("--train_ratio", default=Config.TRAIN_RATIO, type=float, help="Ratio of training data")
        parser.add_argument("--test_ratio", default=Config.TEST_RATIO, type=float, help="Ratio of test data")
        parser.add_argument("--val_ratio", default=Config.VAL_RATIO, type=float, help="Ratio of validation data")

        parser.add_argument("--image_size", nargs=2, default=Config.IMAGE_SIZE, type=int, help="Image height and width")
        # Optimizer 
        parser.add_argument("--optimizer", default=Config.OPTIMIZER, type=str, help="Batch size for training")


        # Model Hyperparameters 
        parser.add_argument("--num_classes", default=Config.NUM_CLASSES, type=int, help="Number of output classes")
        parser.add_argument("--model_name", default=Config.MODEL_NAME, type=str, help="Model architecture name")
        parser.add_argument("--pretrained", default=Config.PRETRAINED, type=bool, help="Use pretrained model weights")
        parser.add_argument("--checkpoint", default=Config.CHECKPOINT, type=str, help="Path to checkpoint file")
        # Training Hyperparameters
        parser.add_argument("--learning_rate", default=Config.LEARNING_RATE, type=float, help="Initial learning rate")
        parser.add_argument("--momentum", default=Config.MOMENTUM, type=float, help="Momentum for optimizer")
        parser.add_argument("--weight_decay", default=Config.WEIGHT_DECAY, type=float, help="Weight decay for optimizer")
        parser.add_argument("--num_epochs", default=Config.NUM_EPOCHS, type=int, help="Number of training epochs")
        parser.add_argument("--lr_step_size", default=Config.LR_STEP_SIZE, type=int, help="LR scheduler step size")
        parser.add_argument("--lr_gamma", default=Config.LR_GAMMA, type=float, help="LR scheduler decay factor")
        ## Loss Hyperparameters 
        parser.add_argument("--loss", default=Config.LOSS, type=str, help="Loss function to use")
        parser.add_argument("--scale", default=Config.SCALE, type=float, help="Scale Hyperparamets")
        parser.add_argument("--margin", default=Config.MARGIN, type=float, help="Margin Hyperparamets")
        parser.add_argument("--embedding_size", default=Config.EMBEDDING_SIZE, type=float, help="Interclass Filtering Threshold")
        # Checkpointing
        parser.add_argument("--checkpoint_interval", default=Config.CHECKPOINT_INTERVAL, type=int, help="Checkpoint save interval")
        parser.add_argument("--fine_tuning", default=Config.FINE_TUNING, type=bool, help="Enable fine-tuning")
        args = parser.parse_args()
        return args
    @staticmethod
    def update_config_from_args(config,  args):
        """Update Config attributes with command-line arguments if provided."""
        for key, value in vars(args).items():
            if value is not None:  # Only override if argument is provided
                # Convert argparse keys to uppercase to match Config attributes
                config_key = key.upper()
                if hasattr(config, config_key):
                    setattr(config, config_key, value)

    
    
# Example usage in training script:
# from config import Config
# config = Config.parse_args()
