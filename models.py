from keras.applications import VGG16
from keras.layers import Dense, Flatten, Dropout, Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D


def transfert_model(keep_weights=True, num_classes=10):

    # Load pre-trained VGG16 model without top layers
    base_model = VGG16(weights='imagenet' if keep_weights else None, include_top=False,
                       input_shape=(224, 224, 3) if keep_weights else (39, 174, 1))

    # Freeze convolutional base
    for layer in base_model.layers:
        layer.trainable = False

    # Add new fully connected layers
    x = Flatten()(base_model.output)
    # num_classes is the number of classes in your dataset
    output = Dense(num_classes, activation='softmax')(x)

    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)
    return model


def base_model(num_classes=10, input_shape=None, dropout_ratio=None):

    model = Sequential()
    if input_shape is None:
        model.add(Input(shape=(None, None, 1)))
    else:
        model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=16, kernel_size=(2, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Conv2D(filters=32, kernel_size=(2, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=(2, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=(2, 4), activation='relu'))
    model.add(GlobalAveragePooling2D())
    if dropout_ratio is not None:
        model.add(Dropout(dropout_ratio))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def improved_model(num_classes=10, input_shape=None, dropout_ratio=0.25):
    model = Sequential()
    if input_shape is None:
        model.add(Input(shape=(None, None, 1)))
    else:
        model.add(Input(shape=input_shape))

    model.add(Conv2D(filters=32, kernel_size=(2, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=(2, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=(2, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=256, kernel_size=(2, 4), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout_ratio))
    model.add(Dense(num_classes, activation='softmax'))
    return model
