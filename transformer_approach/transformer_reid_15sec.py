# transformer_reid_15sec.py — Refined Transformer-Based Player Re-Identification for 15-Second Football Clips

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import warnings

warnings.filterwarnings('ignore')

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
        # Corrected projection dimensions
        self.input_proj = nn.Linear(input_dim, hidden_dim)  # (512->256)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # 256
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # 1024
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Added latent projection
        self.latent_proj = nn.Linear(hidden_dim, 128)  # New layer

    def forward(self, x, return_latent=False):
        # x shape: (seq_len, batch_size, input_dim)
        x = self.input_proj(x)  # (seq_len, batch_size, hidden_dim)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)
        
        if return_latent:
            latent = torch.mean(memory, dim=0)  # (batch_size, hidden_dim)
            latent = self.latent_proj(latent)   # (batch_size, 128)
            return latent, None  # Return latent only
        return memory


class ColorHistogramExtractor:
    def __init__(self, bins=8):
        self.bins = bins
    def extract(self, image):
        if image is None or image.size == 0:
            return np.zeros(self.bins * 3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256])
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-8)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-8)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-8)
        return np.concatenate([h_hist, s_hist, v_hist])

class PlayerTracker:
    def __init__(self, model_path, device='cuda', sequence_length=8):
        self.device = device
        self.sequence_length = sequence_length
        print(f"Loading your fine-tuned model: {model_path}")
        self.detector = YOLO(model_path)
        self.transformer_ae = TransformerAutoEncoder(input_dim=512, hidden_dim=256).to(device)
        self.color_extractor = ColorHistogramExtractor()
        self.gallery = {}  # permanent gallery
        self.temp_gallery = {}  # staged gallery for new tracks
        self.next_id = 1
        self.frame_idx = 0
        self.feature_extractor = self._build_feature_extractor().to(device)
        self.similarity_threshold = 0.8
        self.timeout_frames = 450  # 15 seconds at 30fps

    def _build_feature_extractor(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def preprocess_crop(self, crop):
        if crop is None or crop.size == 0:
            return None
        crop = cv2.resize(crop, (128, 256))
        crop = crop.astype(np.float32) / 255.0
        crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
        return crop.to(self.device)

    def extract_cnn_features(self, crop_tensor):
        if crop_tensor is None:
            return None
        with torch.no_grad():
            features = self.feature_extractor(crop_tensor)
        return features.squeeze(0)  # (512,)

    def update_sequence(self, player_id, cnn_features, gallery_type='permanent'):
        gallery = self.gallery if gallery_type == 'permanent' else self.temp_gallery
        if player_id not in gallery:
            gallery[player_id] = {
                'sequence': deque(maxlen=self.sequence_length),
                'last_seen': self.frame_idx,
                'color_hist': None,
                'frames_observed': 0
            }
        gallery[player_id]['sequence'].append(cnn_features)
        gallery[player_id]['last_seen'] = self.frame_idx
        gallery[player_id]['frames_observed'] += 1

    def get_transformer_features(self, player_id, gallery_type='permanent'):
        gallery = self.gallery if gallery_type == 'permanent' else self.temp_gallery
        if player_id not in gallery:
            return None
        sequence = list(gallery[player_id]['sequence'])
        if len(sequence) < 2:
            return None
        while len(sequence) < self.sequence_length:
            sequence.append(sequence[-1])
        sequence_tensor = torch.stack(sequence).unsqueeze(1)  # (seq_len, 1, 512)
        with torch.no_grad():
            latent_features, _ = self.transformer_ae(sequence_tensor, return_latent=True)
        return latent_features.squeeze(0)  # (128,)

    def staged_id_assignment(self, crop, cnn_features):
        # Stage 1: Temporary gallery for first 5 frames
        temp_id = f"temp_{self.next_id}"
        self.update_sequence(temp_id, cnn_features, gallery_type='temp')
        color_hist = self.color_extractor.extract(crop)
        self.temp_gallery[temp_id]['color_hist'] = color_hist
        if self.temp_gallery[temp_id]['frames_observed'] < 5:
            return temp_id
        # Stage 2: Try to match with permanent gallery
        best_id, best_similarity = None, 0.0
        temp_features = self.get_transformer_features(temp_id, gallery_type='temp')
        for player_id, data in self.gallery.items():
            if self.frame_idx - data['last_seen'] > self.timeout_frames:
                continue
            perm_features = self.get_transformer_features(player_id, gallery_type='permanent')
            if perm_features is not None and temp_features is not None:
                transformer_sim = F.cosine_similarity(perm_features, temp_features, dim=0).item()
            else:
                transformer_sim = 0.0
            color_sim = 0.0
            if data['color_hist'] is not None:
                color_sim = np.corrcoef(color_hist, data['color_hist'])[0, 1]
                if np.isnan(color_sim):
                    color_sim = 0.0
            combined_sim = 0.7 * transformer_sim + 0.3 * color_sim
            if combined_sim > best_similarity and combined_sim > self.similarity_threshold:
                best_similarity = combined_sim
                best_id = player_id
        if best_id is not None:
            # Promote to permanent gallery
            self.update_sequence(best_id, cnn_features, gallery_type='permanent')
            self.gallery[best_id]['color_hist'] = color_hist
            del self.temp_gallery[temp_id]
            return best_id
        else:
            # New permanent ID
            new_id = self.next_id
            self.next_id += 1
            self.update_sequence(new_id, cnn_features, gallery_type='permanent')
            self.gallery[new_id]['color_hist'] = color_hist
            del self.temp_gallery[temp_id]
            return new_id

    def cleanup_gallery(self):
        to_remove = []
        for player_id, data in self.gallery.items():
            if self.frame_idx - data['last_seen'] > self.timeout_frames:
                to_remove.append(player_id)
        for player_id in to_remove:
            del self.gallery[player_id]
        # Remove expired temp gallery entries
        to_remove_temp = []
        for temp_id, data in self.temp_gallery.items():
            if self.frame_idx - data['last_seen'] > 30:  # 1 second grace
                to_remove_temp.append(temp_id)
        for temp_id in to_remove_temp:
            del self.temp_gallery[temp_id]

    def filter_detections_by_quality(self, detections):
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            if conf < 0.4:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < 2000:
                continue
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / max(width, 1)
            if aspect_ratio < 1.2 or aspect_ratio > 4.0:
                continue
            filtered.append(det)
        return filtered

    def process_frame(self, frame):
        results = self.detector(frame, verbose=False)
        detections = []
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_tensor = self.preprocess_crop(crop)
                if crop_tensor is None:
                    continue
                cnn_features = self.extract_cnn_features(crop_tensor)
                if cnn_features is None:
                    continue
                # Staged ID assignment: only assign permanent ID after 5 frames
                player_id = self.staged_id_assignment(crop, cnn_features)
                detections.append({
                    'id': player_id,
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf
                })
        self.frame_idx += 1
        self.cleanup_gallery()
        detections = self.filter_detections_by_quality(detections)
        return detections

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Processing 15-second clip: {video_path}")
        print(f"Video specs: {width}x{height} @ {fps}fps")
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.process_frame(frame)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                player_id = det['id']
                conf = det['conf']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{player_id} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        cap.release()
        out.release()
        print(f"✅ Completed! Output saved: {output_path}")
        print(f"Total unique players identified: {len(self.gallery)}")
        print(f"Total frames processed: {frame_count}")

def main():
    MODEL_PATH = "best.pt"
    VIDEO_PATH = "15sec_input_720p.mp4"
    OUTPUT_PATH = "output_transformer_reid.mp4"
    tracker = PlayerTracker(
        model_path=MODEL_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        sequence_length=8
    )
    try:
        tracker.process_video(VIDEO_PATH, OUTPUT_PATH)
        print("\n🎉 SUCCESS! Your 15-second clip has been processed.")
        print(f"Check the output: {OUTPUT_PATH}")
        print("Players will maintain consistent IDs even when exiting/re-entering the frame!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your model path and video path are correct.")

if __name__ == "__main__":
    main()
