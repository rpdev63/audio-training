import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_data(csv_file_path, random_state=1):
    df = pd.read_csv(csv_file_path)
    df['features'] = [np.asarray(np.load(feature_path))
                      for feature_path in tqdm(df[f'mfcc_features_path'])]
    num_classes = 10
    df['labels_categorical'] = df['classID'].apply(
        lambda x: np.eye(num_classes)[x])

    # Add one dimension for the channel
    X = np.array(df['features'].tolist())
    X = X.reshape(X.shape + (1,))
    y = np.array(df['labels_categorical'].tolist())

    # Create validation and test
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.30,
                                                        random_state=random_state,
                                                        stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                    y_test,
                                                    test_size=0.5,
                                                    random_state=random_state,
                                                    stratify=y_test)
    return X_train, X_test, X_val, y_train, y_test, y_val


def schedule(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.95
    return lr


def launch_training(model, X_train, y_train, X_val, y_val, lr=0.001, bs=256, epochs=100, patience=10, decay=1):
    model.summary()
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=patience, restore_best_weights=True)
    checkpointer = ModelCheckpoint(
        filepath='saved_models/best_fcn.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
    callbacks = [checkpointer, early_stopping]
    if decay < 1:
        lr_scheduler = LearningRateScheduler(schedule)
        callbacks.append(lr_scheduler)
    # Train the model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=bs,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks
                        )
    return model, history


def get_eval(model, history, X_test, y_test):
    # Plot training and validation loss
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    # Save the plot as a PNG file
    plt.savefig('training_validation_accuracy.png')
    plt.show()
    best_val_accuracy = round(max(history.history['val_accuracy']), 3)
    print(f"Best Validation Accuracy: {best_val_accuracy}")
    y_pred_probs = model.predict(X_test)
    y_pred = np.round(y_pred_probs)
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print(f"Test Accuracy:{accuracy}")
