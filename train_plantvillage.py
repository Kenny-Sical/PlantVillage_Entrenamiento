import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Ruta del dataset
DATASET_DIR = r"C:\Users\sical\OneDrive\Escritorio\U\Ciclo7\Inteligencia_artificial\PlantVillage"

IMG_SIZE = 128
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print(f"Clases detectadas ({len(classes)}): {classes}")

class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

images = []
labels = []

print("Cargando imágenes...")

IMAGES_PER_CLASS = 200  # Limita para entrenamiento más rápido, sube si tu PC aguanta

for cls in classes:
    cls_dir = os.path.join(DATASET_DIR, cls)
    count = 0
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                img_path = os.path.join(cls_dir, fname)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(class_to_idx[cls])
                count += 1
            except Exception as e:
                print(f"Error con {img_path}: {e}")

print(f"Total de imágenes cargadas: {len(images)}")

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Transfer Learning con MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Congela la base para entrenamiento más rápido y menos sobreajuste

# Modelo con Dropout y regularización
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),                # Dropout para evitar sobreajuste
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early stopping para evitar sobreentrenamiento
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Entrenamiento
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy en test: {acc:.2f}")

# Guardar el modelo y las etiquetas
model.save('modelo_transfer.h5')

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('modelo_transfer.tflite', 'wb') as f:
    f.write(tflite_model)
print("Modelo exportado como modelo_transfer.tflite")

# Guardar las etiquetas
with open('labels.txt', 'w') as f:
    for cls in classes:
        f.write(cls + '\n')
print("Archivo labels.txt creado.")


