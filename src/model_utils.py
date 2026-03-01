"""
model_utils.py - RetinaScan AI Model Architectures
====================================================
Contains:
- EfficientNetB3 model builder
- ResNet50 model builder
- Ensemble model builder
- Shared utilities (metrics, callbacks, etc.)

WHERE TO USE:
- Write this file on your LAPTOP in src/model_utils.py
- This file is NEVER run directly
- It is IMPORTED by training notebooks (04, 05, 06)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, ResNet50
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import (
    cohen_kappa_score, classification_report,
    confusion_matrix, roc_auc_score
)


# ─────────────────────────────────────────────
# 1. EFFICIENTNETB3 MODEL
# ─────────────────────────────────────────────

def build_efficientnet(
    input_shape=(224, 224, 3),
    num_classes=5,
    dropout_rate=0.3,
    trainable_base=False
):
    """
    Build EfficientNetB3 model with ImageNet pretrained weights.

    Strategy:
      Phase 1 → Freeze base, train only top layers (fast)
      Phase 2 → Unfreeze top layers of base and fine-tune (better accuracy)

    Args:
        input_shape   : Tuple, e.g. (224, 224, 3)
        num_classes   : Number of output classes (5 for DR grading)
        dropout_rate  : Dropout before final Dense layer
        trainable_base: If True, unfreeze all base layers (use for fine-tuning)

    Returns:
        Compiled Keras model
    """
    # Base model - pretrained on ImageNet
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,          # Remove ImageNet classification head
        input_shape=input_shape
    )

    # Phase 1: Freeze entire base model
    base_model.trainable = trainable_base

    # If fine-tuning: unfreeze only top 30 layers
    if trainable_base:
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        print(f"  Fine-tuning: top 30 layers unfrozen")
    else:
        print(f"  Phase 1: all base layers frozen")

    # Build the model
    inputs = keras.Input(shape=input_shape)

    # Base feature extraction
    x = base_model(inputs, training=False)

    # Custom classification head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='EfficientNetB3_DR')

    # Count trainable parameters
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    total_params = sum(
        tf.size(w).numpy() for w in model.weights
    )
    print(f"  EfficientNetB3 built:")
    print(f"    Total params:     {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")

    return model


# ─────────────────────────────────────────────
# 2. RESNET50 MODEL
# ─────────────────────────────────────────────

def build_resnet50(
    input_shape=(224, 224, 3),
    num_classes=5,
    dropout_rate=0.4,
    trainable_base=False
):
    """
    Build ResNet50 model with ImageNet pretrained weights.

    Args:
        input_shape   : Tuple, e.g. (224, 224, 3)
        num_classes   : Number of output classes (5 for DR grading)
        dropout_rate  : Dropout before final Dense layer
        trainable_base: If True, unfreeze top layers (fine-tuning phase)

    Returns:
        Compiled Keras model
    """
    # Base model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Phase 1: Freeze base
    base_model.trainable = trainable_base

    # Fine-tuning: unfreeze top 40 layers
    if trainable_base:
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        print(f"  Fine-tuning: top 40 layers unfrozen")
    else:
        print(f"  Phase 1: all base layers frozen")

    # Build the model
    inputs = keras.Input(shape=input_shape)

    x = base_model(inputs, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout_2')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 4, name='dropout_3')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='ResNet50_DR')

    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    total_params = sum(
        tf.size(w).numpy() for w in model.weights
    )
    print(f"  ResNet50 built:")
    print(f"    Total params:     {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")

    return model


# ─────────────────────────────────────────────
# 3. COMPILE HELPER
# ─────────────────────────────────────────────

def compile_model(model, learning_rate=0.001):
    """
    Compile a model with Adam optimizer and standard metrics.

    Args:
        model        : Keras model to compile
        learning_rate: Initial learning rate

    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    print(f"  Model compiled | LR: {learning_rate}")
    return model


# ─────────────────────────────────────────────
# 4. CALLBACKS
# ─────────────────────────────────────────────

def get_callbacks(model_save_path, patience_early_stop=7, patience_lr=4):
    """
    Standard callbacks for training.

    Args:
        model_save_path   : Path to save best model, e.g. 'models/efficientnet_best.keras'
        patience_early_stop: Epochs to wait before early stopping
        patience_lr       : Epochs to wait before reducing LR

    Returns:
        List of Keras callbacks
    """
    callback_list = [
        # Save best model based on val_accuracy
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        # Stop training if val_loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce LR when val_loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,           # LR = LR * 0.3
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    print(f"  Callbacks set:")
    print(f"    Save path      : {model_save_path}")
    print(f"    Early stop     : patience={patience_early_stop}")
    print(f"    LR reduction   : patience={patience_lr}, factor=0.3")

    return callback_list


# ─────────────────────────────────────────────
# 5. EVALUATION METRICS
# ─────────────────────────────────────────────

def evaluate_model(model, val_generator, validation_steps, num_classes=5):
    """
    Evaluate a trained model and print all metrics.

    Args:
        model           : Trained Keras model
        val_generator   : Validation data generator
        validation_steps: Number of steps to run
        num_classes     : Number of classes

    Returns:
        Dictionary with all metrics
    """
    print("\n📊 Evaluating model...")

    y_true = []
    y_pred_probs = []

    for i in range(validation_steps):
        X_batch, y_batch = next(val_generator)
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    accuracy  = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa     = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy              : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Weighted F1-Score     : {f1:.4f}")
    print(f"  Weighted Precision    : {precision:.4f}")
    print(f"  Weighted Recall       : {recall:.4f}")
    print(f"  Quadratic Kappa (QWK) : {kappa:.4f}")
    print("=" * 60)

    print("\n  Per-class Report:")
    class_names = [f"Grade {i}" for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        'accuracy'  : accuracy,
        'f1'        : f1,
        'precision' : precision,
        'recall'    : recall,
        'kappa'     : kappa,
        'y_true'    : y_true,
        'y_pred'    : y_pred,
        'y_pred_probs': y_pred_probs
    }


# ─────────────────────────────────────────────
# 6. TWO-PHASE TRAINING HELPER
# ─────────────────────────────────────────────

def train_two_phase(
    model_name,         # 'efficientnet' or 'resnet'
    build_fn,           # build_efficientnet or build_resnet50
    train_generator,
    val_generator,
    steps_per_epoch,
    validation_steps,
    class_weights,
    input_shape=(224, 224, 3),
    num_classes=5,
    phase1_epochs=15,
    phase2_epochs=10,
    phase1_lr=1e-3,
    phase2_lr=1e-5,
    save_dir='models'
):
    """
    Two-phase training strategy:
      Phase 1: Freeze base, train only classification head (fast, ~15 epochs)
      Phase 2: Unfreeze top layers, fine-tune with very low LR (~10 epochs)

    Args:
        model_name      : Name string for saving files
        build_fn        : Function that builds the model
        train_generator : Training data generator
        val_generator   : Validation data generator
        steps_per_epoch : Training steps per epoch
        validation_steps: Validation steps per epoch
        class_weights   : Dict of class weights for imbalance
        input_shape     : Image input shape
        num_classes     : Number of classes
        phase1_epochs   : Epochs for Phase 1
        phase2_epochs   : Epochs for Phase 2
        phase1_lr       : Learning rate for Phase 1
        phase2_lr       : Learning rate for Phase 2 (much lower!)
        save_dir        : Directory to save models

    Returns:
        (model, phase1_history, phase2_history)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{model_name}_best.keras"

    print("\n" + "=" * 60)
    print(f"  PHASE 1 TRAINING — {model_name.upper()}")
    print(f"  Frozen base | LR={phase1_lr} | Epochs={phase1_epochs}")
    print("=" * 60)

    # Build with frozen base
    model = build_fn(
        input_shape=input_shape,
        num_classes=num_classes,
        trainable_base=False
    )
    model = compile_model(model, learning_rate=phase1_lr)

    callbacks_p1 = get_callbacks(
        save_path,
        patience_early_stop=6,
        patience_lr=3
    )

    history1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=phase1_epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1
    )

    print(f"\n  Phase 1 complete!")
    print(f"  Best val_accuracy: {max(history1.history['val_accuracy']):.4f}")

    # ── Phase 2: Fine-tuning ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  PHASE 2 FINE-TUNING — {model_name.upper()}")
    print(f"  Partial unfreeze | LR={phase2_lr} | Epochs={phase2_epochs}")
    print("=" * 60)

    # Rebuild with partial unfreeze
    model = build_fn(
        input_shape=input_shape,
        num_classes=num_classes,
        trainable_base=True
    )

    # Load best weights from phase 1
    model.load_weights(save_path)

    # Compile with very low LR
    model = compile_model(model, learning_rate=phase2_lr)

    callbacks_p2 = get_callbacks(
        save_path,
        patience_early_stop=5,
        patience_lr=3
    )

    history2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=phase2_epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1
    )

    print(f"\n  Phase 2 complete!")
    print(f"  Best val_accuracy: {max(history2.history['val_accuracy']):.4f}")
    print(f"  Model saved to: {save_path}")

    return model, history1, history2


# ─────────────────────────────────────────────
# 7. PLOT TRAINING HISTORY
# ─────────────────────────────────────────────

def plot_training_history(history1, history2=None, model_name='Model', save_path=None):
    """
    Plot training accuracy, loss and AUC curves.
    If history2 is provided, plots Phase 1 + Phase 2 together.

    Args:
        history1  : Phase 1 history object
        history2  : Phase 2 history object (optional)
        model_name: String for plot title
        save_path : If provided, saves the figure to this path
    """
    import matplotlib.pyplot as plt

    # Combine histories if both provided
    if history2:
        acc     = history1.history['accuracy']     + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss    = history1.history['loss']         + history2.history['loss']
        val_loss= history1.history['val_loss']     + history2.history['val_loss']
        auc     = history1.history['auc']          + history2.history['auc']
        val_auc = history1.history['val_auc']      + history2.history['val_auc']
        phase1_end = len(history1.history['accuracy'])
    else:
        acc     = history1.history['accuracy']
        val_acc = history1.history['val_accuracy']
        loss    = history1.history['loss']
        val_loss= history1.history['val_loss']
        auc     = history1.history['auc']
        val_auc = history1.history['val_auc']
        phase1_end = None

    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name} — Training History', fontsize=16, fontweight='bold')

    # Accuracy
    axes[0].plot(epochs, acc,     label='Train', linewidth=2)
    axes[0].plot(epochs, val_acc, label='Val',   linewidth=2)
    if phase1_end:
        axes[0].axvline(x=phase1_end, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, loss,     label='Train', linewidth=2)
    axes[1].plot(epochs, val_loss, label='Val',   linewidth=2)
    if phase1_end:
        axes[1].axvline(x=phase1_end, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(epochs, auc,     label='Train', linewidth=2)
    axes[2].plot(epochs, val_auc, label='Val',   linewidth=2)
    if phase1_end:
        axes[2].axvline(x=phase1_end, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    axes[2].set_title('AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")

    plt.show()


# ─────────────────────────────────────────────
# 8. QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  model_utils.py — Sanity Check")
    print("=" * 60)

    print("\n[1] Building EfficientNetB3...")
    eff_model = build_efficientnet(input_shape=(224, 224, 3), num_classes=5)
    eff_model = compile_model(eff_model, learning_rate=1e-3)

    print("\n[2] Building ResNet50...")
    res_model = build_resnet50(input_shape=(224, 224, 3), num_classes=5)
    res_model = compile_model(res_model, learning_rate=1e-3)

    print("\n[3] Testing forward pass...")
    dummy_input = np.random.rand(2, 224, 224, 3).astype(np.float32)
    eff_out = eff_model.predict(dummy_input, verbose=0)
    res_out = res_model.predict(dummy_input, verbose=0)

    print(f"  EfficientNet output shape : {eff_out.shape}")
    print(f"  ResNet50 output shape     : {res_out.shape}")
    print(f"  EfficientNet softmax sum  : {eff_out.sum(axis=1)}")
    print(f"  ResNet50 softmax sum      : {res_out.sum(axis=1)}")

    print("\n✅ All checks passed! model_utils.py is ready.")
    print("=" * 60)
