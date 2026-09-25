import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_mobilenetv2_model(num_classes=8, freeze_base=True, learning_rate=1e-4):
    """
    Builds a custom MobileNetV2 architecture with transfer learning classification head.
    
    Args:
        num_classes (int): Number of output target classes (default 8).
        freeze_base (bool): Freeze pre-trained ImageNet base layers.
        learning_rate (float): Initial learning rate for Adam optimizer.
        
    Returns:
        tf.keras.Model: Compiled MobileNetV2 model.
    """
    # Load base MobileNetV2 with ImageNet weights, excluding top classification layer
    base_model = MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base layers if fine-tuning stage 1
    if freeze_base:
        base_model.trainable = False
    else:
        # Unfreeze top layers of base model for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = BatchNormalization(name="batch_norm_1")(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense_256")(x)
    x = Dropout(0.4, name="dropout_1")(x)
    x = Dense(128, activation='relu', name="dense_128")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    predictions = Dense(num_classes, activation='softmax', name="softmax_output")(x)

    model = Model(inputs=base_model.input, outputs=predictions, name="SkinHealth_MobileNetV2")

    # Compile model with Categorical Crossentropy & Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    return model

def load_trained_model(model_path=config.MODEL_PATH):
    """
    Loads a saved trained Keras model from disk.
    
    Returns:
        tf.keras.Model: Loaded model object or None if not found.
    """
    if not os.path.exists(model_path):
        return None
    try:
        model = load_model(model_path, compile=True)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def save_class_indices(class_indices, output_path=config.CLASS_INDICES_PATH):
    """Saves dictionary mapping class indices to class codes."""
    with open(output_path, 'w') as f:
        json.dump(class_indices, f, indent=4)

def load_class_indices(input_path=config.CLASS_INDICES_PATH):
    """Loads dictionary mapping class indices to class codes."""
    if not os.path.exists(input_path):
        # Default mapping fallback
        classes = sorted(list(config.SKIN_CONDITIONS.keys()))
        return {str(i): cls for i, cls in enumerate(classes)}
    with open(input_path, 'r') as f:
        return json.load(f)
