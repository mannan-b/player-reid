# ⚽ Player Re-Identification

Re-identifying soccer players across frames in a match video, even after they temporarily exit the frame.  
This project implements and compares **three core approaches** for solving the Player Re-ID problem.

👉 [Project Link](https://github.com/mannan-b/player-reid)

---

## 📦 Approaches Implemented

### 1. 🔹 ByteTracker (YOLOv11-based Baseline)
- Uses the `bytetrack` feature from Ultralytics with `best.pt` detection model.
- Remembers player tracks via ID-assigned trajectories.
- **Limitations:** Fails when players re-enter from a different path.

**Key Hyperparameters:**
- `match_thresh`: 0.7  
- `track_buffer`: 300  
- `track_high_thresh`: 0.4  
- `track_low_thresh`: 0.1  

📹 [ByteTracker Output Video](https://drive.google.com/file/d/1eLQxn0Fz_qrQxtcqQ5MJxUV4RBjmy-8I/view?usp=drive_link)

---

### 2. 🔹 Pure Transformer-Based Approach
- Transformer Autoencoder with 4 encoder layers + positional encoding.
- Extracts 128-D latent embeddings via mean pooling.
- Handles temporal reasoning & player memory.

📹 [5000 players output](https://drive.google.com/file/d/1rzMA0GieRZ8v7QzSrIFt77HWrYmOIL6-/view?usp=sharing)  
📹 [1000 players output (tuned)](https://drive.google.com/file/d/15yQdVfdkSQ7BcMEI9yjSwBOlGWK-YIpL/view?usp=sharing)

---

### 3. 🔹 Hybrid OSNet + Transformer (Best Performing)
- **OSNet** learns spatial multi-scale features (jersey, build, shoes).
- **Transformer** learns temporal motion consistency.
- Combines CNN detail with long-term memory via fusion.

📹 [Hybrid Output Video](https://drive.google.com/file/d/1rUGOx4ODgsGPm7eK33IzfPCvunSxMF-T/view?usp=sharing)

---

## ⚙️ Features

- **Dynamic Gallery Buffer** with cleanup & age-weighting
- **Color Histogram Matching** for appearance verification
- **Transformer-enhanced hybrid features**
- **Bounding Box Similarity** as contextual aid

---

## 🚧 Known Issues / Blockers

- Player path changes reduce ByteTracker effectiveness
- Transformer training is sensitive to weight initialization
- Hyperparameter tuning is crucial for real-world success

---

## 🧪 Future Work

- Refined hyperparameter search via optical flow/video analysis
- Explore pure CNN Re-ID pipelines
- Implement full **JDE Tracking** architecture

---

## 📚 References

1. *Runner re-identification in single-view open-world videos* — Suzuki et al.  
2. *Player Re-ID using body part appearances* — Bhosale, Kumar, Doermann

---

## 👤 Author

**Mannan Bajaj**  
[GitHub Profile](https://github.com/mannan-b)
