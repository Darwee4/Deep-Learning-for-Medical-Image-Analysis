import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

class MedicalImageClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """Initialize the medical image classifier"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the transfer learning model using ResNet50"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return models.Model(inputs, outputs)
    
    def preprocess_data(self, image_dir, batch_size=32):
        """Create data generators with preprocessing and augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            image_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            image_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def evaluate_model(self, generator):
        """Evaluate model using AUC-ROC and confusion matrix"""
        y_true = generator.classes
        y_pred = self.model.predict(generator)
        
        # Calculate AUC-ROC
        auc = roc_auc_score(y_true, y_pred[:, 1])
        print(f"AUC-ROC Score: {auc:.4f}")
        
        # Calculate confusion matrix
        y_pred_classes = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)
        print("Confusion Matrix:")
        print(cm)
        
        return auc, cm
    
    def classify_image(self, image_path):
        """Classify a new medical image"""
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=self.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = self.model.predict(img_array)
        return predictions[0]
    
    def save_model(self, path):
        """Save the trained model"""
        self.model.save(path)
        
    def load_model(self, path):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(path)
