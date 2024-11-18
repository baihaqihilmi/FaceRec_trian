from torch import nn
import math
import torch
from torch.nn import functional as F
from utils.get_configs import Config
class ArcFaceLoss(nn.Module):
    def __init__(self, cfg ):
        super(ArcFaceLoss, self).__init__()

        num_classes = cfg.NUM_CLASSES
        embedding_size = cfg.EMBEDDING_SIZE
        margin = cfg.MARGIN
        self.scale = cfg.SCALE

        # Cross Entropy
        # Weight matrix W (class centers)
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)  # Initialize weights with Xavier
        
        self.cos_m = math.cos(margin)         # cos(m)
        self.sin_m = math.sin(margin)         # sin(m)
        self.th = math.cos(math.pi - margin)  # Threshold: cos(pi - m)
        self.mm = math.sin(math.pi - margin) * margin  # sin(pi - m) * m
        # Cross Entropy 
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, embeddings , labels):
        #Cross entropy Loss

        # Normalize the weight matrix and embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).to('cuda')  # (batch_size, embedding_size)
        weight_norm = F.normalize(self.weight, p=2, dim=1).to('cuda')  # (num_classes, embedding_size)
        
        # Cosine similarity between embeddings and weights
        cosine = F.linear(embeddings, weight_norm)  # (batch_size, num_classes)
        # Get the cosine value of the target class
        # if labels :
        index = torch.arange(0, cosine.size(0), dtype=torch.long).to(labels.device)
        cosine_of_target_class = cosine[index, labels]
        
        # Calculate sin(theta) from cos(theta)
        sin_theta = torch.sqrt(1.0 - torch.pow(cosine_of_target_class, 2))
        
        # Compute cos(theta + m) using the ArcFace formula
        cos_theta_plus_m = cosine_of_target_class * self.cos_m - sin_theta * self.sin_m
        
        # Apply the margin only to the target class logits
        cosine[index, labels] = torch.where(cosine_of_target_class > self.th, cos_theta_plus_m, cosine_of_target_class - self.mm)
        
        # Multiply the logits by the scale (s)
        raw_logits = cosine * self.scale

        logits = self.cross_entropy(raw_logits , labels)

        print(logits , labels) if torch.isnan(logits).any() else None
        return raw_logits , logits
    

def get_loss(cfg):
    
    return ArcFaceLoss(cfg)
          

if __name__ == "__main__":
    cfg = Config()
    model = get_loss(cfg)