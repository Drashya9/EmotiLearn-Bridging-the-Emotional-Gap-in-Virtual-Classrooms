import os
import re
import torch
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import mean_squared_error

# Configuration
AUDIO_DIR = "wav_data"
RAW_CSV = "emotion_segments.csv"
FILTERED_CSV = "filtered_segments.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_EMOTIONS = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']

# Match and extract segments from audio files
df = pd.read_csv(RAW_CSV)
df['video'] = df['video'].astype(str).str.strip()
df['start_time'] = df['start_time'].round(3)
df['end_time'] = df['end_time'].round(3)

audio_filenames = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
audio_meta = []

for fname in audio_filenames:
    parts = fname.replace(".wav", "").split("_")
    if len(parts) >= 3:
        try:
            video = "_".join(parts[:-2])
            start, end = round(float(parts[-2]), 3), round(float(parts[-1]), 3)
            audio_meta.append((video, start, end, fname))
        except:
            continue

renamed_rows = []
for _, row in df.iterrows():
    video, start, end = row['video'], round(row['start_time'], 3), round(row['end_time'], 3)
    for a_video, a_start, a_end, fname in audio_meta:
        if video == a_video and start >= a_start and end <= a_end:
            new_name = f"{video}_{start:.4f}_{end:.4f}.wav"
            new_path = os.path.join(AUDIO_DIR, new_name)
            old_path = os.path.join(AUDIO_DIR, fname)
            if not os.path.exists(new_path):
                torchaudio.save(new_path, torchaudio.load(old_path)[0][:, int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)], SAMPLE_RATE)
            row['filename'] = new_name
            renamed_rows.append(row)
            break

filtered_df = pd.DataFrame(renamed_rows)
filtered_df.to_csv(FILTERED_CSV, index=False)
print(f"Matched and clipped {len(filtered_df)} segments out of {len(df)} total rows.")

# Load Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
wav2vec.eval()

# Dataset class
class EmotionAudioDataset(Dataset):
    def __init__(self, csv_path, audio_folder):
        self.df = pd.read_csv(csv_path)
        self.audio_folder = audio_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row[TARGET_EMOTIONS].values.astype(np.float32)
        filepath = os.path.join(self.audio_folder, row['filename'])

        try:
            waveform, sr = torchaudio.load(filepath)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            if waveform.numel() == 0:
                return None
            return waveform.squeeze(0), torch.tensor(target)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

# Collate function
def collate_fn(batch):
    batch = [item for item in batch if item is not None and isinstance(item[0], torch.Tensor) and item[0].dim() == 1]
    if len(batch) == 0:
        return torch.zeros((1, 768)), torch.zeros((1, 6))

    waveforms, targets = zip(*batch)
    waveforms = [w.squeeze().numpy() for w in waveforms]

    try:
        inputs = processor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(DEVICE)
        if hasattr(inputs, 'attention_mask'):
            attention_mask = inputs.attention_mask.to(DEVICE)
            outputs = wav2vec(input_values=input_values, attention_mask=attention_mask)
        else:
            outputs = wav2vec(input_values=input_values)

        embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        return embeddings, torch.stack(targets)
    except Exception as e:
        print(f"Processor error in collate_fn: {e}")
        return torch.zeros((1, 768)), torch.zeros((1, 6))

# Model definition
class EmotionRegressor(nn.Module):
    def __init__(self, input_dim=768, output_dim=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training
print("Preparing data and model...")
dataset = EmotionAudioDataset(FILTERED_CSV, AUDIO_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = EmotionRegressor().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
loss_log = []
print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        if x.shape[0] == 1 and torch.all(x == 0):
            continue
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    loss_log.append(total_loss / len(dataloader))

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in dataloader:
        if x.shape[0] == 1 and torch.all(x == 0):
            continue
        preds = model(x.to(DEVICE)).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

pred_df = pd.DataFrame(all_preds, columns=[f"pred_{e}" for e in TARGET_EMOTIONS])
true_df = pd.DataFrame(all_labels, columns=[f"true_{e}" for e in TARGET_EMOTIONS])
pd.concat([pred_df, true_df], axis=1).to_csv(os.path.join(OUTPUT_DIR, "emotion_predictions.csv"), index=False)

rmse_scores = [np.sqrt(mean_squared_error(all_labels[:, i], all_preds[:, i])) for i in range(6)]
with open(os.path.join(OUTPUT_DIR, "rmse_scores.txt"), "w") as f:
    for e, score in zip(TARGET_EMOTIONS, rmse_scores):
        f.write(f"{e}: RMSE = {score:.4f}\n")

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "emotion_regressor.pth"))

# Plot training loss
plt.plot(range(1, EPOCHS+1), loss_log, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"))
plt.close()

# Plot RMSE scores
sns.barplot(x=TARGET_EMOTIONS, y=rmse_scores)
plt.title("RMSE per Emotion")
plt.ylabel("RMSE")
plt.savefig(os.path.join(OUTPUT_DIR, "rmse_barplot.png"))
plt.close()

# Scatter plots for predictions vs true
for i, emotion in enumerate(TARGET_EMOTIONS):
    plt.scatter(all_labels[:, i], all_preds[:, i], alpha=0.5)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{emotion.capitalize()} Prediction")
    plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{emotion}.png"))
    plt.close()

# Correlation heatmap
corr = pd.DataFrame(all_preds, columns=TARGET_EMOTIONS).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation between Predicted Emotions")
plt.savefig(os.path.join(OUTPUT_DIR, "predicted_emotion_correlation.png"))
plt.close()

print("All processing complete. Outputs saved in 'outputs' folder.")
