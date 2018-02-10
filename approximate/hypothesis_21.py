import os
import glob
import datetime
import hashlib
import json
import numpy
from numpy import random

from keras import models
from keras import layers

from matplotlib import pyplot

name, ext = os.path.splitext(__file__)
MODELS_DIR = 'models_' + name

examples_amount = 10000
x_start = 0
x_end = 300
x_size = x_end - x_start
separation_factor = 0.8
noise_magnitude = 0.01
batch_size = 32
epochs = 10


def gaussian(x, mu, sigma=1, scale=1):
    """
    x - array of arguments
    mu - expected value in probability theory
    sigma - standard deviation in statistics
    """
    return numpy.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * numpy.sqrt(2 * numpy.pi)) / scale


def lagrange(x, x0, gamma=1, scale=1):
    return 1.0 / (numpy.pi * gamma * (1 + ((x - x0) / gamma)**2)) / scale


functions = [
    gaussian,
    lagrange
]


def generate_data():
    X = []
    Y = []
    Z = []

    x = numpy.linspace(x_start, x_end, x_size)

    for i in range(examples_amount):
        y = numpy.zeros(x_size)
        z = numpy.zeros(x_size)

        for j in range(random.randint(0, 10)):
            function_id = numpy.random.randint(0, len(functions))
            function = functions[function_id]

            r = random.random()
            offset = r * (x_end - x_start) + x_start
            param = 3 + 7 * random.random()

            y = y + function(x, offset, param)

            index = int(r * x_size)
            z[index] = 1

        y = y.reshape((-1, 1))
        y_max = numpy.max(y)
        if y_max > 0:
            y = y / y_max

        X.append(x)
        Y.append(y)
        Z.append(z)

    X = numpy.array(X)
    Y = numpy.array(Y)
    Z = numpy.array(Z)

    return X, Y, Z


def get_dataset(data=None):
    _, y, z = data or generate_data()

    separator = int(examples_amount * separation_factor)

    y_train = y[:separator]
    z_train = z[:separator]

    y_test = y[separator:]
    z_test = z[separator:]

    # add noise
    #y_test = random.normal(y_test, noise_magnitude)

    return (y_train, z_train), (y_test, z_test)


def create_model():
    input = layers.Input(shape=(x_size, 1))
    x = layers.Conv1D(32, 5, padding='same')(input)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, padding='same')(x)
    x = layers.Activation('linear')(x)

    flag = False
    if flag:
        x = layers.Reshape((300, 32, 1))(x)
        x = layers.Conv2D(1, (5, 32), padding='same')(x)
        x = layers.MaxPooling2D((1, 32))(x)
        x = layers.Reshape((300,))(x)
        output = layers.Activation('sigmoid')(x)
    else:
        x = layers.Reshape((x_size * 32, 1))(x)
        x = layers.MaxPooling1D(32, 32)(x)
        output = x = layers.Reshape((x_size,))(x)
        output = layers.Activation('relu')(x)

    model = models.Model(inputs=input, outputs=output)

    return model


def compile_model(model):
    loss='mean_squared_error'
    #loss='categorical_crossentropy'

    #optimizer = optimizers.rmsprop(lr=0.001, decay=1e-6)
    optimizer = 'adam'

    model.compile(loss=loss,
        optimizer=optimizer,
        metrics=['accuracy'])


def train_model(model, y_train, z_train, y_test, z_test):
    history = model.fit(y_train, z_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(y_test, z_test),
        shuffle=True)

    return history


def plot_history(history):
    y = history.history['val_acc']

    pyplot.plot(y)
    pyplot.show()


def plot_test(model, y_test, z_test):
    x = numpy.linspace(x_start, x_end, x_size)
    predict = model.predict(y_test)

    for y, z, p in zip(y_test, z_test, predict):
        pyplot.plot(x, y)
        pyplot.plot(x, z)
        pyplot.plot(x, p)

        try:
            pyplot.show()
        except KeyboardInterrupt:
            break


def save_model(model):
    """
    Model name: <file name>_<date>_<model arch hash>
    """

    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    name, ext = os.path.splitext(__file__)

    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%dT%H:%M:%S')

    config = model.get_config()
    raw = json.dumps(config)
    hash = hashlib.md5(raw.encode('utf-8')).hexdigest()

    model_name = name + '_' + date + '_' + hash + '.h5'
    model_path = os.path.join(MODELS_DIR, model_name)

    #model.save(model_path)
    models.save_model(model, model_path)  # *.h5
    print('Saved trained model at %s ' % model_path)


def load_model(model_path):
    return models.load_model(model_path)


def load_model_choice():
    name, ext = os.path.splitext(__file__)
    search_pattern = name + '_*.h5'
    search_pattern = os.path.join(MODELS_DIR, search_pattern)

    files = glob.glob(search_pattern)

    if len(files) == 0:
        raise ValueError('No saved models')

    print('Saved models:')
    for i, f in enumerate(files):
        print('[%i]' % i, f)

    index = int(input('Choice model file: '))
    model_path = files[index]
    model = load_model(model_path)

    return model


def main():
    (x_train, y_train), (x_test, y_test) = get_dataset()

    answer = input('Train model from scratch? (YES/no): ')
    if answer == 'no':
        model = load_model_choice()
    else:
        model = create_model()

    model.summary()
    compile_model(model)
    history = train_model(model, x_train, y_train, x_test, y_test)

    save_model(model)

    plot_history(history)
    plot_test(model, x_test, y_test)


if __name__ == '__main__':
    main()

