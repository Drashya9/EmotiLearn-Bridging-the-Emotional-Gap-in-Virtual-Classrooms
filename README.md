# EmotiLearn-Bridging-the-Emotional-Gap-in-Virtual-Classrooms
EmotiLearn is a video-focused emotion recognition system that enhances online education by detecting student emotions—confusion, boredom, frustration, and engagement—using facial cues from video. Powered by 3D CNNs and GRUs, it delivers real-time feedback to help educators personalize virtual learning.

## Features
- **GRU-based model** for temporal modeling of emotions from videos.
- **3D CNN + Ordinal Regression** for fine-grained intensity prediction.
- **ResNet** as a baseline model.
- **Transformer-based model** for sequential analysis (limited performance).
- **Wav2Vec2-based audio model** for acoustic emotion recognition.
- **Zoom chat analysis tools** for engagement metrics.

## Repository Structure
EmotiLearn/
- README.md
- requirements.txt
- src/ # Python scripts
- notebooks/ # Jupyter notebooks (experiments)

## Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/EmotiLearn.git
cd EmotiLearn
pip install -r requirements.txt
```
## Results
GRU achieved the highest accuracy overall, particularly for Confusion (73.47%) and Frustration (83.67%).
3D CNN + Ordinal Regression performed best on Engagement (60.5%).
ResNet baseline underperformed, and Transformer struggled with class imbalance.

## Future Work
Real-time deployment via lightweight models.
Integration of textual chat-based cues.
Domain adaptation for diverse environments.
Ethical deployment with opt-in transparency.
