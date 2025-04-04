from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        '004__Brain_Tumor_MRI/Dataset/Training',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        '004__Brain_Tumor_MRI/Dataset/Testing',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, test_generator