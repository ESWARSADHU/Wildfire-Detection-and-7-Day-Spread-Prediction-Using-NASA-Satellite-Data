# Install libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ConvLSTM2D
from tensorflow.keras.callbacks import EarlyStopping
import cartopy.crs as ccrs
from tqdm import tqdm

df = pd.read_csv('combined_data.csv')

df['date'] = pd.to_datetime(df['acq_date'], errors='coerce')
df = df.dropna(subset=['date', 'latitude', 'longitude'])

def fire_confident(row):
    conf = row['confidence']
    if isinstance(conf, str):
        return conf.strip().lower() in ['h', 'high', 'n', 'nominal']
    try:
        return float(conf) >= 50
    except:
        return True

df = df[df.apply(fire_confident, axis=1)]

# ----- 2. Prepare Larger Grid & More Days -----
lat_bins, lon_bins = 36, 72       # Use higher if RAM allows
max_days = min(120, len(df['date'].dt.date.unique()))  # Expand days, fit memory budget
unique_dates = sorted(df['date'].dt.date.unique())[:max_days]
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

fire_images = []
for date in unique_dates:
    daily = df[df['date'].dt.date == date]
    H, _, _ = np.histogram2d(
        daily['latitude'], daily['longitude'],
        bins=[lat_bins, lon_bins],
        range=[[lat_min, lat_max], [lon_min, lon_max]]
    )
    fire_mask = (H > 0).astype(int)
    fire_images.append(fire_mask)
fire_images = np.array(fire_images)

freqs = [im.sum() for im in fire_images]
major_thresh = max(1, int(np.percentile(freqs, 60)))  # 60th percentile for balance
labels = (fire_images.sum(axis=(1,2)) > major_thresh).astype(int)
print(f"Major fire threshold: {major_thresh}")
print(f"All label counts (0=no major fire, 1=major fire): {np.bincount(labels)}")

# ----- 3. Train/Test-Split - Check Distributions -----
X = fire_images[..., np.newaxis]
y = labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("Train label counts:", np.bincount(y_train))
print("Test label counts:", np.bincount(y_test))

if (len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2):
    print("Warning: Not enough positive/negative class in train/test splits, try adjusting major_thresh or increase data.")

# ----- 4. K-Fold Cross Validation -----
skf = StratifiedKFold(n_splits=5)
all_acc, all_prec, all_recall, all_f1 = [], [], [], []
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print("Train labels:", np.bincount(y_tr))
    print("Test labels:", np.bincount(y_te))

    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_tr), y=y_tr)
    class_weights_dict = dict(zip(np.unique(y_tr), class_weights))

    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(lat_bins, lon_bins, 1)),
        MaxPooling2D((2,2)),
        Dropout(0.4),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=20, validation_data=(X_te, y_te),
              callbacks=[early_stop], verbose=0, class_weight=class_weights_dict)
    y_pred_prob = model.predict(X_te)
    y_pred = (y_pred_prob > 0.4).astype(int)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

    all_acc.append(acc)
    all_prec.append(prec)
    all_recall.append(rec)
    all_f1.append(f1)

print("\nAveraged over all folds:")
print(f"Mean Accuracy: {np.mean(all_acc):.3f}")
print(f"Mean Precision: {np.mean(all_prec):.3f}")
print(f"Mean Recall: {np.mean(all_recall):.3f}")
print(f"Mean F1: {np.mean(all_f1):.3f}")

# ----- 5. ConvLSTM Spread Prediction (on full data, regularized) -----
seq_len = 3
if len(fire_images) > seq_len:
    X_seq, y_seq = [], []
    for i in range(len(fire_images) - seq_len):
        X_seq.append(fire_images[i:i+seq_len][..., np.newaxis])
        y_seq.append(fire_images[i+seq_len][..., np.newaxis])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    split_idx = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

    convlstm_model = Sequential([
        ConvLSTM2D(8, (3,3), activation='relu', input_shape=(seq_len, lat_bins, lon_bins, 1), return_sequences=False),
        Dropout(0.4),
        Flatten(),
        Dense(lat_bins * lon_bins, activation='sigmoid'),
        tf.keras.layers.Reshape((lat_bins, lon_bins, 1))
    ])
    convlstm_model.compile(optimizer='adam', loss='binary_crossentropy')
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    convlstm_model.fit(X_train_seq, y_train_seq, epochs=8, validation_data=(X_test_seq, y_test_seq), callbacks=[early_stop], verbose=1)

    # Autoregressive Weekly Prediction
    input_seq = fire_images[-seq_len:].copy()
    predicted_week = []
    for i in range(7):
        input_X = input_seq[np.newaxis, ...][..., np.newaxis]
        pred_mask = convlstm_model.predict(input_X)[0, :, :, 0]
        pred_mask_bin = (pred_mask > 0.5).astype(int)
        predicted_week.append(pred_mask_bin)
        input_seq = np.concatenate([input_seq[1:], pred_mask_bin[np.newaxis, ...]], axis=0)
    predicted_week = np.array(predicted_week)

    # ----- 6. Visualize Actual & Predicted Masks -----
    def plot_fire_on_world(mask, title):
        fig = plt.figure(figsize=(10,5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(resolution='110m')
        lats = np.linspace(lat_min, lat_max, lat_bins)
        lons = np.linspace(lon_min, lon_max, lon_bins)
        lat_idxs, lon_idxs = np.where(mask > 0)
        ax.scatter(lons[lon_idxs], lats[lat_idxs], color='red', s=20, alpha=0.8, label='Active Fire')
        plt.title(title)
        plt.legend()
        plt.show()

    print("Actual last day fire mask:")
    plot_fire_on_world(fire_images[-1], f'Actual Fires: {unique_dates[-1]}')
    print("Predicted fire masks for next 7 days:")
    for i in range(7):
        plot_fire_on_world(predicted_week[i], f'Predicted Fires: Day {i+1} after {unique_dates[-1]}')
else:
    print("Not enough fire masks for ConvLSTM sequence modeling.")
