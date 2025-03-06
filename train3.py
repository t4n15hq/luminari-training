import os
import ssl
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, ResNet50, MobileNetV2, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, multiply, Reshape, Permute, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.experimental import CosineDecay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import cv2
from tqdm import tqdm
import shutil
import json

# Try to import SMOTE, but continue if not available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available - continuing without it")

# Check for ViT availability
try:
    from tensorflow.keras.applications.vit import ViT
    VIT_AVAILABLE = True
    print("Vision Transformer (ViT) is available")
except ImportError:
    VIT_AVAILABLE = False
    print("Vision Transformer not available in this TensorFlow version")
    
# Fix SSL issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # Increased max epochs
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 10  # Increased patience
LR_REDUCTION_PATIENCE = 5  # Increased patience

# Define dataset paths
base_path = '/Users/tanishqpadwal/Desktop/Skin Disease Detection/dataset'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')
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
    
    # Identify minority and majority classes
    train_class_dist = train_df['label'].value_counts()
    min_class_count = train_class_dist.min()
    max_class_count = train_class_dist.max()
    
    # Define minority classes as those with fewer than 1.5x min samples
    minority_classes = train_class_dist[train_class_dist < min_class_count * 1.5].index.tolist()
    
    # Define very poor classes that need extra attention
    poorest_classes = [
        'Cellulitis Impetigo and other Bacterial Infections',
        'Poison Ivy Photos and other Contact Dermatitis',
        'Herpes HPV and other STDs Photos',
        'Lupus and other Connective Tissue diseases',
        'Light Diseases and Disorders of Pigmentation'
    ]
    
    # Define majority classes as those with more than 0.7x max samples
    majority_classes = train_class_dist[train_class_dist > max_class_count * 0.7].index.tolist()
    
    print(f"Minority classes: {minority_classes}")
    print(f"Poorest performing classes: {poorest_classes}")
    print(f"Majority classes: {majority_classes}")
    
    # Save class weights to file
    with open(os.path.join(run_output_dir, 'class_weights.json'), 'w') as f:
        # Convert class_weight_dict keys to strings for JSON
        serializable_weights = {str(k): float(v) for k, v in class_weight_dict.items()}
        json.dump(serializable_weights, f, indent=4)
    
    return train_df, val_df, class_weight_dict, minority_classes, poorest_classes, majority_classes, class_counts

# Enhanced preprocessing for targeted augmentation
def create_generators(train_df, val_df, model_name, minority_classes=None, poorest_classes=None):
    # Select appropriate preprocessing function
    if model_name == 'EfficientNetB3':
        preprocess_func = tf.keras.applications.efficientnet.preprocess_input
    elif model_name == 'ResNet50':
        preprocess_func = tf.keras.applications.resnet50.preprocess_input
    elif model_name == 'MobileNetV2':
        preprocess_func = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == 'DenseNet121':
        preprocess_func = tf.keras.applications.densenet.preprocess_input
    elif model_name == 'ViT':
        preprocess_func = tf.keras.applications.vit.preprocess_input
    else:
        raise ValueError(f"Unknown model name: {model_name}")

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
    
    # For poorest performing classes, create specialized augmentation generators
    if poorest_classes:
        # Apply class-specific augmentation weights for problematic classes
        class_specific_weights = {}
        for class_name in train_df['label'].unique():
            if class_name in poorest_classes:
                # Apply stronger augmentation to poorest classes
                class_specific_weights[class_name] = 3.0  # Higher weight for poorest classes
            elif class_name in minority_classes:
                # Apply medium augmentation to minority classes
                class_specific_weights[class_name] = 2.0  
            else:
                class_specific_weights[class_name] = 1.0
                
        print(f"Applied stronger augmentation weights to {len(poorest_classes)} poorest classes")
        print(f"Applied medium augmentation weights to {len(minority_classes)} minority classes")
    
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
    elif base_model_name == 'ViT' and VIT_AVAILABLE:
        # Only try to use ViT if it's available
        base_model = ViT(
            image_size=IMG_SIZE,
            patch_size=16,
            num_heads=12,
            transformer_layers=12,
            mlp_dim=3072,
            include_top=False,
            weights='imagenet'
        )
    else:
        if base_model_name == 'ViT':
            print("Vision Transformer not available, defaulting to EfficientNetB3")
            base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
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
            
            # Setup cosine decay learning rate
            initial_epoch = sum(len(h.history.get('accuracy', [])) for h in all_histories)
            decay_steps = len(train_generator) * 5  # 5 epochs per decay cycle
            lr_schedule = CosineDecay(stage_lr, decay_steps)
            
            # Recompile with lower learning rate
            model.compile(
                optimizer=Adam(learning_rate=lr_schedule),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
            )
            
            # Train for this stage (fewer epochs for later stages)
            stage_epochs = 10 if stage < 3 else 5
            
            stage_history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=initial_epoch + stage_epochs,
                initial_epoch=initial_epoch,
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
            best_model = load_model(model_path, compile=False)
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
    
    # Generate individual model evaluation
    val_generator.reset()
    y_pred = best_model.predict(val_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get class names for reports
    class_indices = val_generator.class_indices
    idx_to_label = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_label[i] for i in range(len(class_indices))]
    
    # Classification report for this model
    report = classification_report(val_generator.classes, y_pred_classes, target_names=class_names)
    with open(os.path.join(run_output_dir, f'{model_name}_classification_report.txt'), 'w') as f:
        f.write(f"{model_name} Validation Metrics:\n")
        f.write(f"Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Top-3 Accuracy: {val_top3_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Save confusion matrix for this model
    cm = confusion_matrix(val_generator.classes, y_pred_classes)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', 
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_model, val_accuracy, val_top3_accuracy

# Class-weighted ensemble predictions
def class_weighted_ensemble_predictions(models, val_generator, model_weights, class_model_weights=None):
    """Generate ensemble predictions with optional per-class model weighting"""
    all_predictions = []
    
    for i, model in enumerate(models):
        print(f"Generating predictions from model {i+1}/{len(models)}...")
        val_generator.reset()
        all_predictions.append(model.predict(val_generator, steps=len(val_generator), verbose=1))
    
    # Get class indices
    class_indices = val_generator.class_indices
    num_classes = len(class_indices)
    
    # Initialize array for final predictions
    final_preds = np.zeros_like(all_predictions[0])
    
    # If we have class-specific model weights
    if class_model_weights is not None:
        print("Using class-specific model weights")
        
        # For each sample
        for i in range(final_preds.shape[0]):
            # Get predicted class probabilities from each model
            for class_idx in range(final_preds.shape[1]):
                # Apply class-specific model weights
                for model_idx in range(len(models)):
                    weight = class_model_weights[class_idx][model_idx]
                    final_preds[i, class_idx] += weight * all_predictions[model_idx][i, class_idx]
    else:
        # Standard weighted ensemble
        print("Using standard model weights")
        
        for i, model in enumerate(models):
            # Apply weight to model predictions
            final_preds += all_predictions[i] * model_weights[i]
    
    # Normalize predictions
    row_sums = final_preds.sum(axis=1, keepdims=True)
    final_preds = final_preds / (row_sums + 1e-8)  # Add small epsilon to avoid division by zero
    
    return final_preds

# Generate per-class model weights based on validation performance
def calculate_class_model_weights(models, val_generator):
    """Calculate per-class weights for each model based on validation performance"""
    num_classes = len(val_generator.class_indices)
    model_count = len(models)
    
    # Initialize matrix to hold class-specific model accuracies
    class_model_accuracies = np.zeros((num_classes, model_count))
    
    # For each model
    for model_idx, model in enumerate(models):
        print(f"Calculating class-specific weights for model {model_idx+1}/{model_count}...")
        
        # Generate predictions
        val_generator.reset()
        predictions = model.predict(val_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_generator.classes
        
        # For each class
        for class_idx in range(num_classes):
            # Find samples of this class
            class_samples = np.where(y_true == class_idx)[0]
            
            if len(class_samples) > 0:
                # Calculate accuracy for this class
                class_correct = np.sum(y_pred[class_samples] == class_idx)
                class_accuracy = class_correct / len(class_samples)
                class_model_accuracies[class_idx, model_idx] = class_accuracy
    
    # Convert accuracies to weights
    class_model_weights = {}
    
    for class_idx in range(num_classes):
        # Add small epsilon to ensure all weights are positive
        weights = class_model_accuracies[class_idx, :] + 1e-5
        
        # Normalize to get weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all models perform poorly, use equal weights
            weights = np.ones(model_count) / model_count
            
        class_model_weights[class_idx] = weights
    
    # Log class-specific model weights
    class_indices = val_generator.class_indices
    idx_to_label = {v: k for k, v in class_indices.items()}
    
    with open(os.path.join(run_output_dir, 'class_model_weights.txt'), 'w') as f:
        f.write("Class-specific model weights:\n\n")
        for class_idx, weights in class_model_weights.items():
            f.write(f"Class: {idx_to_label[class_idx]}\n")
            for model_idx, weight in enumerate(weights):
                f.write(f"  Model {model_idx+1}: {weight:.4f}\n")
            f.write("\n")
    
    return class_model_weights

# Enhanced analyze_results function
def analyze_results(y_true, y_pred, class_indices, output_dir, title="Ensemble"):
    """Analyze and visualize results including top-3 metrics and confusion patterns"""
    # Get class names
    idx_to_label = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_label[i] for i in range(len(class_indices))]
    
    # Calculate top-1 predictions
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate top-3 predictions
    top3_indices = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]
    top3_hits = [y_true[i] in top3_indices[i] for i in range(len(y_true))]
    top3_accuracy = np.mean(top3_hits)
    
    print(f"{title} Top-1 Accuracy: {np.mean(y_pred_classes == y_true):.4f}")
    print(f"{title} Top-3 Accuracy: {top3_accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(f"\n{title} Classification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(output_dir, f'{title.lower()}_classification_report.txt'), 'w') as f:
        f.write(f"{title} Top-1 Accuracy: {np.mean(y_pred_classes == y_true):.4f}\n")
        f.write(f"{title} Top-3 Accuracy: {top3_accuracy:.4f}\n\n")
        f.write(report)
    
    # Generate and save confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=False, cmap='Blues', 
                  xticklabels=class_names,
                  yticklabels=class_names)
        plt.title(f'{title} Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{title.lower()}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze confusion patterns
        confusion_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((i, j, cm[i, j]))
        
        # Sort by confusion count
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Save top confused pairs
        with open(os.path.join(output_dir, f'{title.lower()}_confusion_patterns.txt'), 'w') as f:
            f.write("Top confused class pairs:\n\n")
            for true_idx, pred_idx, count in confusion_pairs[:20]:  # Top 20 confusion pairs
                f.write(f"True: {class_names[true_idx]}, Predicted: {class_names[pred_idx]}, Count: {count}\n")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    # Per-class top-3 accuracy with robust handling
    class_top3_acc = {}
    for i, class_name in enumerate(class_names):
        # Get indices of samples belonging to this class - handle empty arrays properly
        class_samples = np.where(np.array(y_true) == i)[0]
        if len(class_samples) > 0:
            # Calculate top-3 accuracy for this class
            class_hits = [i in top3_indices[j] for j in class_samples]
            class_top3_acc[class_name] = np.mean(class_hits)
        else:
            class_top3_acc[class_name] = 0
    
    # Plot per-class top-3 accuracy if we have data
    try:
        if class_top3_acc and any(class_top3_acc.values()):
            plt.figure(figsize=(15, 10))
            acc_series = pd.Series(class_top3_acc)
            if not acc_series.empty and not acc_series.isna().all():
                acc_series.sort_values(ascending=False).plot(kind='bar')
                plt.title(f'{title} Top-3 Accuracy by Class')
                plt.ylabel('Top-3 Accuracy')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{title.lower()}_top3_accuracy_by_class.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save top-3 accuracy by class to file
                with open(os.path.join(output_dir, f'{title.lower()}_top3_by_class.txt'), 'w') as f:
                    f.write(f"{title} Top-3 Accuracy by Class:\n\n")
                    for class_name, accuracy in acc_series.sort_values(ascending=False).items():
                        f.write(f"{class_name}: {accuracy:.4f}\n")
            else:
                print("Warning: No valid top-3 accuracy data to plot")
        else:
            print("Warning: Empty class top-3 accuracy dictionary")
    except Exception as e:
        print(f"Error plotting top-3 accuracy by class: {e}")
    
    return top3_accuracy, class_top3_acc