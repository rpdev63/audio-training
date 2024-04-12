import optuna
from utils import launch_training, get_data
from models import improved_model


def objective(trial):

    LEARNING_RATE = trial.suggest_loguniform('learning_rate', 5e-4, 0.05)
    BATCH_SIZE = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128, 256, 512])
    EPOCHS = 150
    PATIENCE = EPOCHS / 10
    DROPOUT = trial.suggest_float('dropout_ratio', 0.05, 0.5)
    DECAY = 0.95
    # RS = 1

    model = improved_model(
        input_shape=X_train.shape[1:], dropout_ratio=DROPOUT)
    trained_model, history = launch_training(
        model, X_train, y_train, X_val, y_val, lr=LEARNING_RATE, bs=BATCH_SIZE, epochs=EPOCHS, patience=PATIENCE, decay=DECAY)
    best_val_accuracy = round(max(history.history['val_accuracy']), 3)
    return best_val_accuracy


if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="audio-recognize2",
        direction="maximize"
    )
X_train, X_test, X_val, y_train, y_test, y_val = get_data(
    "extracted.csv", random_state=1)

study.optimize(objective, n_trials=50, n_jobs=-1)


# !optuna-dashboard sqlite:///db.sqlite3
