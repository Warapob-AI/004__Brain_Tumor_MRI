from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from build_model import build_model
from load_data import load_data

def train_model():
    train_generator, test_generator = load_data()

    model = build_model()

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        '004__Brain_Tumor_MRI/Model/brain_tumor_best_model.h5', 
        monitor='val_loss',
        mode='min', 
        save_best_only=True,
        verbose=1 
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=60,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        callbacks=[early_stopping, checkpoint] 
    )

    with open('004__Brain_Tumor_MRI/Logs/training_logs.txt', 'w') as f:
        f.write(str(history.history))
