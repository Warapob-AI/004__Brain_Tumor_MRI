from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import BatchNormalization

def build_model():
    # Using ResNet50 Pretrained Model for Transfer Learning
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    # Unfreeze the top layers of ResNet50 for fine-tuning
    for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers
        layer.trainable = True

    # Add custom classification layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax')(x)

    # Define the final model
    model = Model(inputs=base_model.input, outputs=x)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
