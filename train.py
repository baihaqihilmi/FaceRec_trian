import torch
from torch import nn
from PIL import Image
from utils.get_configs import Config
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils.preprocessing import split_dataset , create_experiment_folder
from utils.earlt_stopping import EarlyStopping
from tqdm import tqdm
from backbones import get_model 
from torch.utils.data import DataLoader
from losses import get_loss
from optimizer import get_optimizer 
import wandb
from torch.utils.data import Subset
import os
import argparse
import torchmetrics
import time
import json
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
def train(cfg , model , criterion, optimizer, train_dataset, val_dataset):

    ##INitiate experiments
    
    exp_folder =  create_experiment_folder(base_path="experiments", config_data=cfg.__dict__)
    checkpoint_intervals = cfg.CHECKPOINT_INTERVAL

    ##
    early_stopping = EarlyStopping(patience=20 , min_delta=0.001)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    # Initialize metrics
    precision_metric = torchmetrics.Precision(num_classes=cfg.NUM_CLASSES, average='macro' , task="multiclass" ).to(cfg.DEVICE)
    recall_metric = torchmetrics.Recall(num_classes=cfg.NUM_CLASSES, average='macro' , task="multiclass").to(cfg.DEVICE)
    
    start_time = time.time()

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_corr = 0
        train_samples = 0
        precision_metric.reset()
        recall_metric.reset()
        print("-------_TRAINING BEGIN------------")
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{cfg.NUM_EPOCHS}', unit='batch') as pbar:
            for images, labels in train_loader:
                images = images.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)

                # Forward pass
                optimizer.zero_grad()
                emb = model(images)

                # Calculate loss
                raw_logits , loss = criterion(emb, labels )
                probs = F.softmax(raw_logits, dim=1)
                predicted = torch.argmax(probs, dim=1)
                ## Calculate accuracy
                correct = (predicted == labels).sum().item() 
                
                ## Update metrics
                train_loss += loss.item() * len(labels)
                train_corr += correct
                train_samples += len(labels) 
                # Backward pass 9 Exploding Gradient)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), 0.8)
                optimizer.step()

                precision_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
                pbar.set_postfix(loss=loss.item())  # Update the progress bar with loss
                pbar.update(1)  # Increment the progress bar by 1
        
        train_acc = train_corr / train_samples
        train_epoch_loss = train_loss / train_samples
        train_precision = precision_metric.compute()
        train_recall = recall_metric.compute()
        print(f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}], Train Cross Entropy Loss: {train_epoch_loss} , Train Accuracy: {train_acc:.4f} , Train Precision: {train_precision:.4f} , Train Recall: {train_recall:.4f}")

        # Validate model
        model.eval()
        val_loss = 0
        val_corr = 0
        best_val_loss = 0
        total_samples = 0
        precision_metric.reset()
        recall_metric.reset()

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc='Validating', unit='batch') as pbar:
                for images, labels in val_loader:
                    images = images.to(cfg.DEVICE)
                    labels = labels.to(cfg.DEVICE)

                    # Forward pass
                    emb = model(images)

                        # Calculate loss
                    raw_logits , loss = criterion(emb, labels )
                    probs = F.softmax(raw_logits, dim=1)
                    predicted = torch.argmax(probs, dim=1)
                    ## Calculate accuracy

                    correct = (predicted == labels).sum().item() 
                    val_corr += correct
                    total_samples += len(labels)

                    val_loss += loss.item() * len(labels)
                    
                    precision_metric.update(predicted, labels)
                    recall_metric.update(predicted, labels)

                    pbar.set_postfix(loss=loss.item())  # Update the progress bar with loss
                    pbar.update(1)
        val_loss = val_loss / total_samples
        val_acc = val_corr / total_samples
        val_precision = precision_metric.compute()
        val_recall = recall_metric.compute()
        print(f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}], Val Cross Entropy Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f} , Val Precision: {val_precision:.4f} , Val Recall: {val_recall:.4f}") 
        
        if val_loss < best_val_loss:
            print("Saving best model")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_folder, "best", 'best_model.pth'))
            torch.save(criterion.state_dict(), os.path.join(exp_folder, 'best', 'best_criterion.pth'))
        if checkpoint_intervals > 0 and epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(exp_folder, "checkpoints", f"checkpoint_epoch_{epoch}.pt"))
            torch.save(criterion.state_dict(), os.path.join(exp_folder, 'checkpoints', f"checkpoint_epoch_{epoch}_criterion.pt"))
        wandb.log({"train_loss ": train_epoch_loss, "train_acc": train_acc , "val_loss": val_loss, "val_acc": val_acc , "train_precision": train_precision , "train_recall": train_recall , "val_precision": val_precision , "val_recall": val_recall})
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    #Save model
    torch.save(model.state_dict(), os.path.join(exp_folder,  "checkpoints", 'last_model.pt'))
    torch.save(criterion.state_dict(), os.path.join(exp_folder,  "checkpoints", 'last_criterion.pt'))
    wandb.save(os.path.join(exp_folder, "checkpoints", 'last_model.pt'))
    wandb.save(os.path.join(exp_folder, "checkpoints", 'last_criterion.pt'))
    wandb.save(os.path.join(exp_folder, "best", 'best_model.pth'))
    wandb.save(os.path.join(exp_folder, 'best', 'best_criterion.pth'))
    finish_time = time.time()
    total_training_time = finish_time - start_time
    print("Total training time: {:.2f} seconds".format(total_training_time))
    wandb.log({"total_training_time": total_training_time})
    


def main():
    ## Initilization
    cfg = Config()
    parser = Config.parse_args()
    Config.update_config_from_args(cfg, parser)
    # Data Declaration
    print(os.getenv("DATA_DIR"))
    print(cfg.DATA_DIR)

    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])


    # WandB
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="face_recognition", entity="baihaqihilmi18-pamukkale-niversitesi")  # Replace with your WandB project and username



## Dataset Init
    print("-----------DATASET LOADING--------")
    dataset = ImageFolder(cfg.DATA_DIR , transform=transform)
    dataset.class_to_idx = {idx: class_name for idx, class_name in enumerate(dataset.classes)}

    # Classes JSON

    with open(os.path.join(cfg.DATA_DIR, 'classes.json'), 'w') as f:
        json.dump(dataset.class_to_idx, f)

    train_indices, test_indices , val_indices= split_dataset(dataset, cfg.TRAIN_RATIO, cfg.VAL_RATIO)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    ## Model Defiintion
    print("-----------DATASET FINISHED--------")

    model = get_model(cfg.MODEL_NAME)
    model.load_state_dict(torch.load(cfg.CHECKPOINT))

    model.to(cfg.DEVICE)
    if cfg.FINE_TUNING:
        for param in model.parameters():
            param.requires_grad = False


    # Loss and Optimizer
    criterion = get_loss(cfg)
    criterion.to(cfg.DEVICE)
    params = [{"params": model.parameters()} , {"params": criterion.parameters()}]

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
    optimizer = get_optimizer(params, cfg)

    # WandB
    wandb.config.update({
    "learning_rate": cfg.LEARNING_RATE,
    "train_batch_size": cfg.TRAIN_BATCH_SIZE,
    "num_epochs": cfg.NUM_EPOCHS,
    "Loss Function": cfg.LOSS,
    "Model" : cfg.MODEL_NAME

})
    
    
    # Train
    train(cfg , model, criterion, optimizer , train_dataset, val_dataset)
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    else:
        os.system("shutdown +2")
        pass