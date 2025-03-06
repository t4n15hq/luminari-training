import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, ResNet50, MobileNetV2, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, multiply, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 10
LR_REDUCTION_PATIENCE = 5

# Define dataset paths
base_path = '/Users/tanishqpadwal/Desktop/Skin Disease Detection/dataset'
train_dir = os.path.join(base_path, 'train')
output_dir = '/Users/tanishqpadwal/Desktop/Skin Disease Detection/model_output_ensemble_improved'
os.makedirs(output_dir, exist_ok=True)

# Create run-specific output directory
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_output_dir = os.path.join(output_dir, f'run_{run_timestamp}')
os.makedirs(run_output_dir, exist_ok=True)

# Tensorflow memory growth setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Using GPU: {len(physical_devices)} device(s) found")
else:
    print("No GPU found, using CPU")

# Load dataset and split into train/validation
def load_dataset(train_dir):
    image_paths = []
    labels = []
    
    print(f"Scanning directory: {train_dir}")
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            class_images = 0
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(img_path)
                    labels.append(class_name)
                    class_images += 1
            class_counts[class_name] = class_images
            print(f"Found {class_images} images in class: {class_name}")
    
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    
    # Save the raw dataset statistics
    plt.figure(figsize=(15, 8))
    pd.Series(class_counts).sort_values().plot(kind='barh')
    plt.title('Class Distribution in Original Dataset')
    plt.xlabel('Number of Images')
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, 'original_class_distribution.png'))
    plt.close()
    
    # Split into train (80%) and validation (20%)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    print(f"Total images: {len(df)}, Training: {len(train_df)}, Validation: {len(val_df)}")
    
    # Compute class weights for imbalance handling
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(train_df['label']), class_weights)}
    
    return train_df, val_df, class_weight_dict

# Enhanced preprocessing for targeted augmentation
def create_generators(train_df, val_df, model_name):
    # Select appropriate preprocessing function
    if model_name == 'EfficientNetB3':
        preprocess_func = tf.keras.applications.efficientnet.preprocess_input
    elif model_name == 'ResNet50':
        preprocess_func = tf.keras.applications.resnet50.preprocess_input
    elif model_name == 'MobileNetV2':
        preprocess_func = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == 'DenseNet121':
        preprocess_func = tf.keras.applications.densenet.preprocess_input
    else:
        preprocess_func = tf.keras.applications.efficientnet.preprocess_input

    # Standard train data generator with appropriate augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Create train generator    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        x_col='image_path', 
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), 
        batch_size=BATCH_SIZE,
        class_mode='categorical', 
        shuffle=True
    )

    # Validation generator (no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df, 
        x_col='image_path', 
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), 
        batch_size=BATCH_SIZE,
        class_mode='categorical', 
        shuffle=False
    )

    return train_generator, val_generator

# Self-attention module for improved feature focus
def attention_module(x, ratio=8):
    # Channel attention
    channel_axis = -1
    filters = x.shape[channel_axis]
    
    # Shared MLP for channel attention
    shared_layer_one = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(filters, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    # Average pooling
    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, filters))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    # Max pooling
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True))(x)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    # Combine channel attentions
    cbam_feature = tf.keras.layers.add([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)
    
    # Apply channel attention
    x = multiply([x, cbam_feature])
    
    return x

# Build models with different architectures, enhanced with attention
def build_model(base_model_name, num_classes):
    if base_model_name == 'EfficientNetB3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif base_model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError(f"Unsupported model: {base_model_name}")

    # Add custom classification head with attention
    x = base_model.output
    
    # Add attention module
    x = attention_module(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )

    return model, base_model

# Enhanced progressive training function
def train_model(model, base_model, train_generator, val_generator, class_weight_dict, model_name):
    model_path = os.path.join(run_output_dir, f'best_{model_name}.h5')
    log_dir = os.path.join(run_output_dir, f'logs_{model_name}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True, 
            min_delta=0.001,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=LR_REDUCTION_PATIENCE, 
            min_lr=1e-6, 
            verbose=1
        ),
        ModelCheckpoint(
            model_path, 
            save_best_only=True, 
            monitor='val_top3_accuracy', 
            mode='max',
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]

    # Phase 1: Train only the top layers
    print(f"Phase 1: Training top layers of {model_name}...")
    history1 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=15,
        validation_data=val_generator,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save Phase 1 metrics
    val_accuracy1 = max(history1.history.get('val_accuracy', [0])) if history1.history.get('val_accuracy') else 0
    val_top3_accuracy1 = max(history1.history.get('val_top3_accuracy', [0])) if history1.history.get('val_top3_accuracy') else 0
    
    # Progressive unfreezing for fine-tuning
    total_layers = len(base_model.layers)
    unfreeze_stages = 5  # Number of progressive unfreezing stages
    
    # Initialize histories list
    all_histories = [history1]
    
    # Default metrics in case all fine-tuning fails
    val_accuracy2 = 0
    val_top3_accuracy2 = 0
    
    try:
        for stage in range(1, unfreeze_stages + 1):
            # Clear previous session to avoid memory leaks
            tf.keras.backend.clear_session()
            
            # Calculate layers to unfreeze in this stage
            unfreeze_percent = stage / unfreeze_stages
            layers_to_unfreeze = int(total_layers * unfreeze_percent)
            
            print(f"Phase 2.{stage}: Fine-tuning {model_name} - unfreezing {layers_to_unfreeze}/{total_layers} layers ({unfreeze_percent*100:.1f}%)")
            
            # Re-freeze all layers first
            for layer in base_model.layers:
                layer.trainable = False
                
            # Then unfreeze progressively more layers
            for layer in base_model.layers[-layers_to_unfreeze:]:
                layer.trainable = True
            
            # Calculate reduced learning rate based on stage
            stage_lr = LEARNING_RATE / (5 * (stage + 1))
            
            # Recompile with lower learning rate
            model.compile(
                optimizer=Adam(learning_rate=stage_lr),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
            )
            
            # Train for this stage (fewer epochs for later stages)
            stage_epochs = 10 if stage < 3 else 5
            
            stage_history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=stage_epochs,
                validation_data=val_generator,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            all_histories.append(stage_history)
            
            # Update best metrics
            if stage_history.history.get('val_accuracy'):
                val_accuracy2 = max(val_accuracy2, max(stage_history.history['val_accuracy']))
            
            if stage_history.history.get('val_top3_accuracy'):
                val_top3_accuracy2 = max(val_top3_accuracy2, max(stage_history.history['val_top3_accuracy']))
            
    except Exception as e:
        print(f"Error during progressive fine-tuning: {e}")
        print("Using best results so far")
    
    # Try to load the best model from checkpoints
    best_model = model  # Default to current model
    try:
        if os.path.exists(model_path):
            print(f"Loading best model from {model_path}")
            best_model = tf.keras.models.load_model(model_path, compile=False)
            # Recompile
            best_model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
            )
        else:
            print(f"Best model file not found: {model_path}, using current model")
    except Exception as e:
        print(f"Error loading best model: {e}")
        print("Using current model state")
    
    # Combine all histories for plotting
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = []
        for h in all_histories:
            if key in h.history:
                combined_history[key].extend(h.history[key])
    
    # Plot combined training history
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(combined_history['accuracy'], label='Training Accuracy')
    plt.plot(combined_history['val_accuracy'], label='Validation Accuracy')
    if 'val_top3_accuracy' in combined_history:
        plt.plot(combined_history['val_top3_accuracy'], label='Validation Top-3 Accuracy')
    plt.title(f'{model_name} Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(combined_history['loss'], label='Training Loss')
    plt.plot(combined_history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, f'{model_name}_training_history.png'))
    plt.close()
    
    # Get best validation metrics across phases
    val_accuracy = max(val_accuracy1, val_accuracy2)
    val_top3_accuracy = max(val_top3_accuracy1, val_top3_accuracy2)
    
    print(f"{model_name} best validation accuracy: {val_accuracy:.4f}")
    print(f"{model_name} best top-3 accuracy: {val_top3_accuracy:.4f}")
    
    return best_model, val_accuracy, val_top3_accuracy

def main():
    # Save code copy for reproducibility
    shutil.copy2(__file__, os.path.join(run_output_dir, 'train3_backup.py'))
    
    # Load dataset with enhanced class analysis
    train_df, val_df, class_weight_dict = load_dataset(train_dir)
    
    # Get initial validation generator to determine class indices
    _, initial_val_generator = create_generators(train_df, val_df, 'EfficientNetB3')
    class_indices = initial_val_generator.class_indices
    num_classes = len(class_indices)
    
    # Model architectures to use
    model_names = ['EfficientNetB3', 'ResNet50', 'MobileNetV2', 'DenseNet121']
    
    models = []
    val_accuracies = []
    val_top3_accuracies = []
    
    # Train each model with enhanced monitoring
    for model_name in model_names:
        print(f"\n{'='*20} Training {model_name} {'='*20}")
        
        # Create model-specific generators with appropriate preprocessing
        train_generator, val_generator = create_generators(train_df, val_df, model_name)
        
        # Build model with attention mechanism
        model, base_model = build_model(model_name, num_classes)
        
        # Train model with progressive unfreezing
        trained_model, val_acc, val_top3_acc = train_model(
            model, base_model, train_generator, val_generator, 
            class_weight_dict, model_name
        )
        
        models.append(trained_model)
        val_accuracies.append(val_acc)
        val_top3_accuracies.append(val_top3_acc)
        
        # Print model performance
        print(f"{model_name} Validation Accuracy: {val_acc:.4f}")
        print(f"{model_name} Validation Top-3 Accuracy: {val_top3_acc:.4f}")
        
        # Clean up to free memory
        tf.keras.backend.clear_session()
    
    # Calculate standard ensemble weights based on validation performance
    raw_weights = np.array(val_top3_accuracies) + 1e-5
    weights = raw_weights / np.sum(raw_weights)
    
    print("\nStandard model weights for ensemble:")
    for i, (name, weight) in enumerate(zip(model_names, weights)):
        print(f"{name}: {weight:.3f}")
    
    # Save model weights to file
    with open(os.path.join(run_output_dir, 'model_weights.txt'), 'w') as f:
        f.write("Model weights for ensemble:\n")
        for i, (name, acc, top3_acc, weight) in enumerate(zip(model_names, val_accuracies, val_top3_accuracies, weights)):
            f.write(f"{name}: acc={acc:.4f}, top3_acc={top3_acc:.4f}, weight={weight:.4f}\n")

if __name__ == "__main__":
    main()