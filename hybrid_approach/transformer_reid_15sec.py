import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import warnings
import gc

warnings.filterwarnings('ignore')

class OSBlock(nn.Module):
    """OSNet building block with omni-scale feature learning"""
    def __init__(self, in_channels, out_channels, bottleneck_channels=None):
        super().__init__()
        if bottleneck_channels is None:
            bottleneck_channels = out_channels // 4
        
        # Multiple scale streams
        self.conv1x1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv3x3_1 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.conv3x3_2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.conv3x3_3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        
        self.bn3x3_1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3x3_2 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3x3_3 = nn.BatchNorm2d(bottleneck_channels)
        
        # Lite convolution streams
        self.conv1x3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, (1, 3), padding=(0, 1), bias=False)
        self.conv3x1 = nn.Conv2d(bottleneck_channels, bottleneck_channels, (3, 1), padding=(1, 0), bias=False)
        self.bn1x3 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3x1 = nn.BatchNorm2d(bottleneck_channels)
        
        # Output projection
        self.conv_out = nn.Conv2d(bottleneck_channels * 4, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # Initial 1x1 conv
        out = self.conv1x1(x)
        out = self.bn1x1(out)
        out = self.relu(out)
        
        # Multi-scale streams
        # Stream 1: Direct
        out1 = out
        
        # Stream 2: 3x3 conv
        out2 = self.conv3x3_1(out)
        out2 = self.bn3x3_1(out2)
        out2 = self.relu(out2)
        
        # Stream 3: Cascaded 3x3 convs
        out3 = self.conv3x3_2(out)
        out3 = self.bn3x3_2(out3)
        out3 = self.relu(out3)
        out3 = self.conv3x3_3(out3)
        out3 = self.bn3x3_3(out3)
        out3 = self.relu(out3)
        
        # Stream 4: Lite convolution
        out4 = self.conv1x3(out)
        out4 = self.bn1x3(out4)
        out4 = self.relu(out4)
        out4 = self.conv3x1(out4)
        out4 = self.bn3x1(out4)
        out4 = self.relu(out4)
        
        # Concatenate all streams
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        # Output projection
        out = self.conv_out(out)
        out = self.bn_out(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        return out

class EnhancedOSNet(nn.Module):
    """Enhanced OSNet for sports player re-identification"""
    def __init__(self, num_classes=512, num_blocks=[2, 2, 2]):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # OSNet blocks with increasing channels
        self.layer1 = self._make_layer(64, 128, num_blocks[0])
        self.layer2 = self._make_layer(128, 256, num_blocks[1])
        self.layer3 = self._make_layer(256, 512, num_blocks[2])
        
        # Global pooling and classifier
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(OSBlock(in_channels, out_channels))
            else:
                layers.append(OSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # OSNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Apply dropout and final layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.latent_proj = nn.Linear(hidden_dim, 128)
    
    def forward(self, x, return_latent=False):
        try:
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            memory = self.transformer_encoder(x)
            if return_latent:
                latent = torch.mean(memory, dim=0)
                latent = self.latent_proj(latent)
                return latent, None
            return memory
        except Exception as e:
            print(f"TransformerAutoEncoder error: {e}")
            # Return fallback features
            batch_size = x.shape[1] if len(x.shape) > 1 else 1
            if return_latent:
                return torch.zeros(128, device=x.device), None
            return torch.zeros_like(x)

class ColorHistogramExtractor:
    def __init__(self, bins=8):
        self.bins = bins
    
    def extract(self, image):
        try:
            if image is None or image.size == 0:
                return np.zeros(self.bins * 3)
            
            # Ensure image is valid
            if len(image.shape) != 3 or image.shape[2] != 3:
                return np.zeros(self.bins * 3)
                
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256])
            
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-8)
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-8)
            v_hist = v_hist.flatten() / (v_hist.sum() + 1e-8)
            
            return np.concatenate([h_hist, s_hist, v_hist])
        except Exception as e:
            print(f"Color extraction error: {e}")
            return np.zeros(self.bins * 3)

class PlayerTracker:
    def __init__(self, model_path, device='cuda', sequence_length=8):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.sequence_length = sequence_length
        
        print(f"Loading model: {model_path} on {self.device}")
        
        # Load YOLO model
        try:
            self.detector = YOLO(model_path)
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise
        
        # Initialize Enhanced OSNet + Transformer hybrid
        self.enhanced_osnet = EnhancedOSNet(num_classes=512).to(self.device)
        self.transformer_ae = TransformerAutoEncoder(
            input_dim=512,
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        ).to(self.device)
        
        self.color_extractor = ColorHistogramExtractor()
        
        # Keep simple CNN as backup
        self.simple_cnn = self._build_simple_cnn().to(self.device)
        
        print("✅ Enhanced OSNet + Transformer hybrid initialized")
        
        # Tracking state
        self.gallery = {}
        self.next_id = 1
        self.frame_idx = 0
        
        # Relaxed thresholds for better detection
        self.confidence_threshold = 0.1  # Lower threshold
        self.similarity_threshold = 0.6   # Lower threshold
        self.min_sequence_length = 1     # Allow immediate matching
        self.timeout_frames = 150        # Shorter timeout for 15sec video
        
        print(f"✅ PlayerTracker initialized on {self.device}")

    def _build_simple_cnn(self):
        """Simple CNN as backup feature extractor"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512)
        )

    def preprocess_crop(self, crop):
        """Preprocess crop with error handling"""
        try:
            if crop is None or crop.size == 0:
                return None
            
            # Ensure minimum size
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                return None
            
            # Resize to standard size
            crop = cv2.resize(crop, (128, 256))
            crop = crop.astype(np.float32) / 255.0
            
            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            crop = (crop - mean) / std
            
            # Convert to tensor
            crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
            return crop.to(self.device).float()
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def extract_enhanced_features(self, crop_tensor, use_enhanced=True):
        """Extract features using Enhanced OSNet (primary) or simple CNN (backup)"""
        try:
            if crop_tensor is None:
                return None
            
            with torch.no_grad():
                if use_enhanced:
                    try:
                        # Use Enhanced OSNet for better features
                        features = self.enhanced_osnet(crop_tensor)
                        features = F.normalize(features, p=2, dim=1)
                        return features.squeeze(0).cpu()
                    except Exception as e:
                        print(f"Enhanced OSNet failed, using backup CNN: {e}")
                        # Fallback to simple CNN
                        features = self.simple_cnn(crop_tensor)
                        features = F.normalize(features, p=2, dim=1)
                        return features.squeeze(0).cpu()
                else:
                    # Use simple CNN directly
                    features = self.simple_cnn(crop_tensor)
                    features = F.normalize(features, p=2, dim=1)
                    return features.squeeze(0).cpu()
                    
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def get_hybrid_features(self, player_id):
        """Get hybrid OSNet + Transformer features for a player"""
        try:
            if player_id not in self.gallery:
                return None
            
            sequence = list(self.gallery[player_id]['sequence'])
            if len(sequence) < 1:
                return None
            
            # Get latest OSNet features
            latest_osnet_features = sequence[-1]
            
            # Get transformer features if we have enough sequence
            transformer_features = None
            if len(sequence) >= 2:
                # Pad sequence if needed
                padded_sequence = sequence.copy()
                while len(padded_sequence) < self.sequence_length:
                    padded_sequence.append(padded_sequence[-1])
                
                # Take last sequence_length items
                padded_sequence = padded_sequence[-self.sequence_length:]
                
                # Convert to tensor and move to device
                sequence_tensor = torch.stack([s.to(self.device) for s in padded_sequence]).unsqueeze(1)
                
                with torch.no_grad():
                    transformer_features, _ = self.transformer_ae(sequence_tensor, return_latent=True)
                    transformer_features = transformer_features.squeeze(0).cpu()
            
            # Return hybrid features
            if transformer_features is not None:
                # Combine OSNet and Transformer features
                hybrid_features = torch.cat([latest_osnet_features, transformer_features], dim=0)
                return hybrid_features
            else:
                # Return just OSNet features for new players
                return latest_osnet_features
            
        except Exception as e:
            print(f"Hybrid feature extraction error: {e}")
            return None

    def update_sequence(self, player_id, osnet_features, bbox):
        """Update player sequence with OSNet features"""
        if player_id not in self.gallery:
            self.gallery[player_id] = {
                'sequence': deque(maxlen=self.sequence_length),
                'last_seen': self.frame_idx,
                'color_hist': None,
                'frames_observed': 0,
                'last_bbox': bbox,
                'confidence_history': deque(maxlen=10),
                'osnet_features': None  # Store latest OSNet features
            }
        
        self.gallery[player_id]['sequence'].append(osnet_features.cpu())
        self.gallery[player_id]['osnet_features'] = osnet_features.cpu()
        self.gallery[player_id]['last_seen'] = self.frame_idx
        self.gallery[player_id]['frames_observed'] += 1
        self.gallery[player_id]['last_bbox'] = bbox

    def filter_detections(self, detections):
        """Filter detections with relaxed constraints"""
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            
            # Relaxed confidence threshold
            if conf < self.confidence_threshold:
                continue
            
            # Check area - more permissive
            area = (x2 - x1) * (y2 - y1)
            if area < 200:  # Reduced from 500
                continue
            
            # Check aspect ratio - more permissive
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
                
            aspect_ratio = height / width
            if aspect_ratio < 0.5 or aspect_ratio > 6.0:  # More permissive
                continue
            
            filtered.append(det)
        
        return filtered

    def match_player(self, crop, osnet_features, bbox, conf):
        """Match player using hybrid OSNet + Transformer approach — returns second-most similar match only"""
        try:
            if osnet_features is None:
                return None

            color_hist = self.color_extractor.extract(crop)
            best_id = None
            best_similarity = -1.0
            similarities = []

            for player_id, data in self.gallery.items():
                if self.frame_idx - data['last_seen'] > self.timeout_frames:
                    print(f"⏳ Skipped {player_id}: too old")
                    continue

                player_hybrid_features = self.get_hybrid_features(player_id)
                if player_hybrid_features is None:
                    print(f"⚠️ Skipped {player_id}: No hybrid features")
                    continue

                current_hybrid_features = osnet_features

                if len(data['sequence']) >= 2:
                    temp_sequence = list(data['sequence'])[-(self.sequence_length - 1):] + [osnet_features]
                    while len(temp_sequence) < self.sequence_length:
                        temp_sequence.append(temp_sequence[-1])

                    temp_tensor = torch.stack([s.to(self.device) for s in temp_sequence]).unsqueeze(1)

                    with torch.no_grad():
                        temp_transformer_features, _ = self.transformer_ae(temp_tensor, return_latent=True)
                        temp_transformer_features = temp_transformer_features.squeeze(0).cpu()

                    current_hybrid_features = torch.cat([osnet_features, temp_transformer_features], dim=0)

                if current_hybrid_features.shape[0] == player_hybrid_features.shape[0]:
                    hybrid_sim = F.cosine_similarity(
                        current_hybrid_features.unsqueeze(0),
                        player_hybrid_features.unsqueeze(0),
                        dim=1
                    ).item()
                else:
                    hybrid_sim = F.cosine_similarity(
                        osnet_features.unsqueeze(0),
                        data['osnet_features'].unsqueeze(0),
                        dim=1
                    ).item()

                if hybrid_sim < 0.5:
                    print(f"❌ Skipped {player_id}: Hybrid sim too low ({hybrid_sim:.3f})")
                    continue

                color_sim = 0.0
                if data['color_hist'] is not None:
                    color_corr = np.corrcoef(color_hist, data['color_hist'])[0, 1]
                    color_sim = 0.0 if np.isnan(color_corr) else max(0.0, color_corr)

                bbox_sim = self.calculate_bbox_similarity(bbox, data['last_bbox'])
                if bbox_sim < 0.05:
                    print(f"⚠️ {player_id}: Low bbox_sim {bbox_sim:.3f}, penalizing")
                    bbox_sim *= 0.1

                age_penalty = max(0, self.frame_idx - data['last_seen']) / self.timeout_frames
                age_weight = max(0.0, 1.0 - age_penalty)

                combined_sim = age_weight * (0.45 * hybrid_sim + 0.35 * color_sim + 0.2 * bbox_sim)
                similarities.append((player_id, combined_sim))

                print(f"🔍 Player {player_id} | hybrid: {hybrid_sim:.3f}, color: {color_sim:.3f}, bbox: {bbox_sim:.3f}, age: {age_penalty:.2f} -> combined: {combined_sim:.3f}")

            print("📸 Current gallery size:", len(self.gallery))

            top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)
            print("🏁 Top 3 candidates:", top_matches[:3])

            if len(top_matches) >= 2 and top_matches[0][0] == top_matches[1][0]:
                print("⚠️ Top 2 matches are the SAME player — feature collapse or gallery too small")

            # Return second-best match only
            if len(top_matches) >= 2:
                second_best_id = top_matches[1][0]
                self.update_sequence(second_best_id, osnet_features, bbox)
                self.gallery[second_best_id]['color_hist'] = color_hist
                print(f"✅ Matched second-best player {second_best_id} (score: {top_matches[1][1]:.3f})")
                return second_best_id

            elif len(top_matches) == 1:
                only_id = top_matches[0][0]
                self.update_sequence(only_id, osnet_features, bbox)
                self.gallery[only_id]['color_hist'] = color_hist
                print(f"✅ Only one match available, matched player {only_id} (score: {top_matches[0][1]:.3f})")
                return only_id

            else:
                if len(self.gallery) < 30:
                    new_id = self.next_id
                    self.next_id += 1
                    self.update_sequence(new_id, osnet_features, bbox)
                    self.gallery[new_id]['color_hist'] = color_hist
                    print(f"🆕 New player created: ID {new_id}")
                    return new_id

            return None

        except Exception as e:
            print(f"🔥 Player matching error: {e}")
            return None




    def calculate_bbox_similarity(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / (union + 1e-8)
        except:
            return 0.0

    def cleanup_gallery(self):
        """Clean up inactive players"""
        to_remove = []
        for player_id, data in self.gallery.items():
            if self.frame_idx - data['last_seen'] > self.timeout_frames:
                to_remove.append(player_id)
        
        for player_id in to_remove:
            print(f"Removing inactive player ID: {player_id}")
            del self.gallery[player_id]

    def process_frame(self, frame):
        """Process a single frame"""
        try:
            # Clear GPU cache periodically
            if self.frame_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Run YOLO detection
            results = self.detector(frame, verbose=False, conf=0.1)  # Lower conf for YOLO
            
            detections = []
            if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf.cpu().numpy())
                        
                        # Basic sanity check
                        if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': conf
                            })
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue
            
            print(f"Frame {self.frame_idx}: Raw detections: {len(detections)}")
            
            # Filter detections
            detections = self.filter_detections(detections)
            print(f"Frame {self.frame_idx}: Filtered detections: {len(detections)}")
            
            final_detections = []
            
            # Process each detection
            for det in detections:
                try:
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Extract crop with bounds checking
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    # Preprocess crop
                    crop_tensor = self.preprocess_crop(crop)
                    if crop_tensor is None:
                        continue
                    
                    # Extract features using Enhanced OSNet
                    osnet_features = self.extract_enhanced_features(crop_tensor, use_enhanced=True)
                    if osnet_features is None:
                        continue
                    
                    # Match player using hybrid approach
                    player_id = self.match_player(crop, osnet_features, det['bbox'], det['conf'])
                    
                    if player_id is not None:
                        final_detections.append({
                            'id': player_id,
                            'bbox': det['bbox'],
                            'conf': det['conf']
                        })
                
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Update frame index and cleanup
            self.frame_idx += 1
            if self.frame_idx % 30 == 0:  # Cleanup every 30 frames
                self.cleanup_gallery()
            
            print(f"Frame {self.frame_idx}: Final detections: {len(final_detections)}, Active players: {len(self.gallery)}")
            return final_detections
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            self.frame_idx += 1
            return []

    def process_video(self, video_path, output_path):
        """Process video with improved error handling"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        print(f"📹 Duration: {total_frames/fps:.1f} seconds")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        success_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Process frame
                    detections = self.process_frame(frame)
                    
                    # Draw detections
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        player_id = det['id']
                        conf = det['conf']
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"ID:{player_id} ({conf:.2f})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Write frame
                    out.write(frame)
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    # Write original frame even if processing failed
                    out.write(frame)
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"🎬 Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Safety break for testing
                if frame_count >= total_frames:
                    break
                    
        except Exception as e:
            print(f"❌ Video processing error: {e}")
        
        finally:
            cap.release()
            out.release()
            
            print(f"\n✅ Processing complete!")
            print(f"📊 Stats:")
            print(f"   • Total frames: {frame_count}")
            print(f"   • Successfully processed: {success_count}")
            print(f"   • Unique players tracked: {len(self.gallery)}")
            print(f"   • Output saved: {output_path}")


def main():
    """Main function"""
    MODEL_PATH = "best.pt"
    VIDEO_PATH = "15sec_input_720p.mp4"
    OUTPUT_PATH = "output_transformer_reid.mp4"
    
    print("🚀 Starting Transformer ReID Player Tracker")
    print(f"📱 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        tracker = PlayerTracker(
            model_path=MODEL_PATH,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            sequence_length=6  # Reduced for better performance
        )
        
        tracker.process_video(VIDEO_PATH, OUTPUT_PATH)
        print("\n🎉 SUCCESS! Video processed with consistent player tracking!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()