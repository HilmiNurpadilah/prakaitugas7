import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
import os

# Load data
df = pd.read_csv('data/diabetes.csv', delimiter=';')
print("Kolom pada dataset:", df.columns.tolist())

# Hapus kolom Unnamed (kolom kosong akibat delimiter berlebih)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("Kolom setelah drop Unnamed:", df.columns.tolist())

# Drop baris yang mengandung NaN
# (Jika kolom Unnamed sudah dibuang, dropna tidak akan menghapus semua baris)
df = df.dropna()
print(f"Shape setelah dropna: {df.shape}")


# Pastikan nama kolom konsisten (ada spasi pada 'Diabetes PedigreeFunction')
df = df.rename(columns=lambda x: x.strip())

X = df.drop('Hasil', axis=1)
y = df['Hasil']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Save scaler for inference
joblib.dump(scaler, 'data/scaler_diabetes.save')

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Build model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Save model
model.save('data/model_diabetes.h5')

# Plot loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss vs Validation Loss')
plt.savefig('data/loss_plot.png')

# Evaluate
y_pred = (model.predict(X_test) >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

with open('data/evaluation.txt', 'w') as f:
    f.write(f'Accuracy: {acc}\n')
    f.write(f'Confusion Matrix:\n{cm}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')

print('Accuracy:', acc)
print('Confusion Matrix:\n', cm)
print('Precision:', precision)
print('Recall:', recall)
