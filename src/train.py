import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import config
from src.model import build_mobilenetv2_model, save_class_indices
from src.dataset import generate_sample_dataset


def train_pipeline(epochs=15, batch_size=16):
    """
    Complete training pipeline for MobileNetV2 skin disease classification model:
    1. Loads or generates dataset.
    2. Sets up image data generators with augmentation.
    3. Trains and fine-tunes MobileNetV2 model.
    4. Evaluates test set and saves real metrics and confusion matrix.
    5. Saves trained model artifact (.h5).
    """
    train_dir = os.path.join(config.DATASET_DIR, "train")
    val_dir = os.path.join(config.DATASET_DIR, "val")
    test_dir = os.path.join(config.DATASET_DIR, "test")

    # Check if dataset directories exist; if not, generate synthetic sample dataset
    if not (os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0):
        print("Dataset directory empty. Initializing sample dataset generator...")
        generate_sample_dataset(samples_per_class=40)

    # Setup Image Data Augmentation Generators with MobileNetV2 preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    print(f"Loading image data from '{config.DATASET_DIR}'...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Save class indices mapping
    class_indices = train_generator.class_indices
    # Invert mapping: {index_int: class_code_str}
    idx_to_class = {v: k for k, v in class_indices.items()}
    save_class_indices(idx_to_class)
    num_classes = len(class_indices)
    print(f"Target classes ({num_classes}): {list(class_indices.keys())}")

    # Build and compile MobileNetV2 model
    print("Building MobileNetV2 architecture with custom classification head...")
    model = build_mobilenetv2_model(num_classes=num_classes, freeze_base=True, learning_rate=1e-4)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(config.MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print(f"Starting model training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Fine-tuning phase (unfreeze top layers)
    print("Initiating fine-tuning phase (unfreezing top layers of MobileNetV2)...")
    for layer in model.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    history_ft = model.fit(
        train_generator,
        epochs=max(5, epochs // 2),
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Save final model
    model.save(config.MODEL_PATH)
    print(f"Model successfully saved to '{config.MODEL_PATH}'")

    # Evaluate on Test Set
    print("\n--- Running Evaluation on Test Set ---")
    test_generator.reset()
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Compute Actual Evaluation Metrics
    acc = float(accuracy_score(y_true, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('SkinHealth AI - MobileNetV2 Confusion Matrix')
    plt.xlabel('Predicted Condition')
    plt.ylabel('True Condition')
    plt.tight_layout()
    cm_plot_path = os.path.join(config.MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()

    metrics_results = {
        "accuracy": round(acc, 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_score_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_score_weighted": round(float(f1_weighted), 4),
        "num_test_samples": len(y_true),
        "classes": class_labels,
        "confusion_matrix": cm.tolist()
    }

    with open(config.METRICS_PATH, 'w') as f:
        json.dump(metrics_results, f, indent=4)

    print("\n=== Model Evaluation Summary ===")
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Macro Precision:    {precision_macro * 100:.2f}%")
    print(f"Macro Recall:       {recall_macro * 100:.2f}%")
    print(f"Macro F1-Score:     {f1_macro * 100:.2f}%")
    print(f"Results saved to:   '{config.METRICS_PATH}'")
    print(f"Confusion Matrix:   '{cm_plot_path}'")

    return metrics_results

if __name__ == "__main__":
    train_pipeline(epochs=4, batch_size=16)

