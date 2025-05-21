import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Ruta del dataset
DATASET_DIR = r"C:\Users\sical\OneDrive\Escritorio\U\Ciclo7\Inteligencia_artificial\PlantVillage"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 300

def get_general_class(folder_name):
    name = folder_name.lower()
    if "healthy" in name:
        return "healthy"
    if "early_blight" in name:
        return "early_blight"
    if "late_blight" in name:
        return "late_blight"
    if "bacterial_spot" in name:
        return "bacterial_spot"
    if "mosaic_virus" in name:
        return "mosaic_virus"
    if "yellowleaf" in name or "yellowleaf__curl_virus" in name:
        return "yellowleaf_curl_virus"
    if "leaf_mold" in name:
        return "leaf_mold"
    if "septoria" in name:
        return "septoria_leaf_spot"
    if "spider_mites" in name:
        return "spider_mites"
    if "target_spot" in name:
        return "target_spot"
    return folder_name

folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
general_classes = sorted(set(get_general_class(f) for f in folders))
print(f"Clases generales detectadas ({len(general_classes)}): {general_classes}")

class_to_idx = {cls: idx for idx, cls in enumerate(general_classes)}

images = []
labels = []

print("Cargando imágenes...")

for folder in folders:
    general_cls = get_general_class(folder)
    cls_dir = os.path.join(DATASET_DIR, folder)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                img_path = os.path.join(cls_dir, fname)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(class_to_idx[general_cls])
            except Exception as e:
                print(f"Error con {img_path}: {e}")

print(f"Total de imágenes cargadas: {len(images)}")

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),                   # Regularización extra
    tf.keras.layers.Dense(len(general_classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# EarlyStopping para evitar sobreajuste
#callback = tf.keras.callbacks.EarlyStopping(
#    monitor='val_loss',
#    patience=10,
#    restore_best_weights=True
#)

# Entrenamiento con augmentación
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    verbose=2
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy en test: {acc:.2f}")

# Guardar modelo y etiquetas
model.save('modelo_general.h5')

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('modelo_general.tflite', 'wb') as f:
    f.write(tflite_model)
print("Modelo exportado como modelo_general.tflite")

# Guardar las etiquetas generales
with open('labels.txt', 'w') as f:
    for cls in general_classes:
        f.write(cls + '\n')
print("Archivo labels.txt creado.")