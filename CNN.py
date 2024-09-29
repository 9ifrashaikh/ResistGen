import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


dataset_path = r"C:\Users\LENOVO\Desktop\ResistGen\ResistGen\Dataset"  
save_model_path_resnet = "best_resnet_model.keras"
save_model_path_densenet = "best_densenet_model.keras"


img_height, img_width = 224, 224
batch_size = 8  
num_classes = 3  


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#  function for creating ResNet50 model
def create_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

#  function for creating DenseNet121 model
def create_densenet_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Compile and train the model
def compile_and_train(model, model_name, save_model_path):
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss', save_best_only=True, verbose=1)

    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else 1,
        validation_data=validation_generator,
        validation_steps=validation_steps if validation_steps > 0 else 1,
        epochs=20,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print(f"{model_name} training completed.")
    return history

#  plot training history
def plot_comparison(resnet_history, densenet_history):
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(resnet_history.history['accuracy'], label='ResNet Train Accuracy', color='blue')
    plt.plot(resnet_history.history['val_accuracy'], label='ResNet Val Accuracy', color='blue', linestyle='--')
    plt.plot(densenet_history.history['accuracy'], label='DenseNet Train Accuracy', color='green')
    plt.plot(densenet_history.history['val_accuracy'], label='DenseNet Val Accuracy', color='green', linestyle='--')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(resnet_history.history['loss'], label='ResNet Train Loss', color='blue')
    plt.plot(resnet_history.history['val_loss'], label='ResNet Val Loss', color='blue', linestyle='--')
    plt.plot(densenet_history.history['loss'], label='DenseNet Train Loss', color='green')
    plt.plot(densenet_history.history['val_loss'], label='DenseNet Val Loss', color='green', linestyle='--')
    plt.title('Model Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

#  compare ResNet and DenseNet models
if __name__ == "__main__":
    # Train ResNet model
    resnet_model = create_resnet_model()
    print("Training ResNet model...")
    resnet_history = compile_and_train(resnet_model, "ResNet", save_model_path_resnet)

    # Train DenseNet model
    densenet_model = create_densenet_model()
    print("Training DenseNet model...")
    densenet_history = compile_and_train(densenet_model, "DenseNet", save_model_path_densenet)

    # Plot comparison of ResNet and DenseNet
    plot_comparison(resnet_history, densenet_history)
