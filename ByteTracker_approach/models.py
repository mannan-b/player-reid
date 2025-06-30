# Network architectures for Player Re-Identification using Body Part Appearances
# Implementation of the research paper: 2310.14469v1.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from config import Config

class AppearanceExtractor(nn.Module):
    """
    Appearance stream using ResNet50 as described in the paper.
    Extracts global appearance features from player images.
    """
    def __init__(self):
        super(AppearanceExtractor, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights=Config.RESNET_WEIGHTS)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add adaptive pooling to get fixed size features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final feature projection
        self.feature_proj = nn.Linear(2048, Config.APPEARANCE_DIM)
        
    def forward(self, x):
        # x shape: (batch_size, 3, height, width)
        features = self.backbone(x)  # (batch_size, 2048, h, w)
        pooled = self.avgpool(features)  # (batch_size, 2048, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch_size, 2048)
        projected = self.feature_proj(pooled)  # (batch_size, 512)
        return projected

class PoseExtractor(nn.Module):
    """
    Body part stream using OpenPose subnetwork as described in the paper.
    Extracts spatial body part features using Part Affinity Fields (PAFs).
    """
    def __init__(self):
        super(PoseExtractor, self).__init__()
        
        # Initial feature extraction (similar to OpenPose first stage)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # PAF stages (simplified version of OpenPose multi-stage architecture)
        self.paf_stage1 = self._make_paf_stage(256, 38)  # 19 PAF fields * 2
        self.paf_stage2 = self._make_paf_stage(256 + 38, 38)
        
        # Confidence map stages
        self.conf_stage1 = self._make_conf_stage(256 + 38, 19)  # 17 keypoints + 2
        self.conf_stage2 = self._make_conf_stage(256 + 38 + 19, 19)
        
        # Feature compression for bilinear pooling
        self.feature_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 4)),  # Compress spatial dimensions
            nn.Flatten(),
            nn.Linear(19 * 8 * 4, Config.POSE_DIM)
        )
        
    def _make_paf_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
    
    def _make_conf_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        # Initial feature extraction
        features = self.initial_conv(x)  # (batch_size, 256, h/8, w/8)
        
        # PAF stages
        paf1 = self.paf_stage1(features)
        paf_input = torch.cat([features, paf1], dim=1)
        paf2 = self.paf_stage2(paf_input)
        
        # Confidence map stages
        conf_input = torch.cat([features, paf2], dim=1)
        conf1 = self.conf_stage1(conf_input)
        
        final_input = torch.cat([features, paf2, conf1], dim=1)
        conf2 = self.conf_stage2(final_input)
        
        # Compress features for bilinear pooling
        compressed = self.feature_compress(conf2)  # (batch_size, 128)
        return compressed

class CompactBilinearPooling(nn.Module):
    """
    Compact Bilinear Pooling implementation as described in the paper.
    Efficiently combines appearance and pose features using random projections.
    """
    def __init__(self, input1_dim, input2_dim, output_dim):
        super(CompactBilinearPooling, self).__init__()
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.output_dim = output_dim
        
        # Random projection matrices (fixed after initialization)
        self.register_buffer('h1', torch.randint(0, output_dim, (input1_dim,)))
        self.register_buffer('s1', 2 * torch.randint(0, 2, (input1_dim,)) - 1)
        self.register_buffer('h2', torch.randint(0, output_dim, (input2_dim,)))
        self.register_buffer('s2', 2 * torch.randint(0, 2, (input2_dim,)) - 1)
        
    def forward(self, x, y):
        """
        x: appearance features (batch_size, input1_dim)
        y: pose features (batch_size, input2_dim)
        """
        batch_size = x.size(0)
        
        # Compute count sketches
        x_sketch = torch.zeros(batch_size, self.output_dim, device=x.device)
        y_sketch = torch.zeros(batch_size, self.output_dim, device=y.device)
        
        # Count sketch for x
        for i in range(self.input1_dim):
            x_sketch[:, self.h1[i]] += self.s1[i] * x[:, i]
            
        # Count sketch for y
        for i in range(self.input2_dim):
            y_sketch[:, self.h2[i]] += self.s2[i] * y[:, i]
        
        # Convolution in frequency domain (equivalent to element-wise product)
        x_fft = torch.fft.fft(x_sketch, dim=1)
        y_fft = torch.fft.fft(y_sketch, dim=1)
        
        # Element-wise product and inverse FFT
        product_fft = x_fft * y_fft
        result = torch.fft.ifft(product_fft, dim=1).real
        
        return result

class TwoStreamReIDNetwork(nn.Module):
    """
    Complete two-stream network for player re-identification.
    Combines appearance and body part features using bilinear pooling.
    """
    def __init__(self):
        super(TwoStreamReIDNetwork, self).__init__()
        
        self.appearance_stream = AppearanceExtractor()
        self.pose_stream = PoseExtractor()
        self.bilinear_pooling = CompactBilinearPooling(
            Config.APPEARANCE_DIM, 
            Config.POSE_DIM, 
            Config.FUSED_DIM
        )
        
        # Final normalization layer
        self.norm = nn.LayerNorm(Config.FUSED_DIM)
        
    def forward(self, x):
        """
        x: input image tensor (batch_size, 3, height, width)
        Returns: normalized fused features (batch_size, fused_dim)
        """
        # Extract appearance features
        appearance_features = self.appearance_stream(x)  # (batch_size, 512)
        
        # Extract pose features
        pose_features = self.pose_stream(x)  # (batch_size, 128)
        
        # Fuse features using bilinear pooling
        fused_features = self.bilinear_pooling(appearance_features, pose_features)  # (batch_size, 8000)
        
        # Normalize features
        normalized_features = self.norm(fused_features)
        
        # L2 normalization for cosine similarity
        normalized_features = F.normalize(normalized_features, p=2, dim=1)
        
        return normalized_features

class TripletLoss(nn.Module):
    """
    Triplet loss implementation for person re-identification training.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        anchor: anchor samples (batch_size, feature_dim)
        positive: positive samples (batch_size, feature_dim)
        negative: negative samples (batch_size, feature_dim)
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)
        
        # Triplet loss
        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        return loss.mean()

def load_pretrained_reid_model(model_path=None):
    """
    Load a pre-trained re-ID model or initialize a new one.
    """
    model = TwoStreamReIDNetwork()
    
    if model_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pre-trained model from {model_path}")
        except:
            print(f"Could not load model from {model_path}, using fresh initialization")
    
    model.to(Config.DEVICE)
    model.eval()
    
    return model

if __name__ == "__main__":
    # Test the network architecture
    model = TwoStreamReIDNetwork()
    model.to(Config.DEVICE)
    
    # Test with dummy input
    dummy_input = torch.randn(4, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    dummy_input = dummy_input.to(Config.DEVICE)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print("Model architecture test passed!")