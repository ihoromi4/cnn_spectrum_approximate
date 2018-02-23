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

examples_amount = 100000
x_start = 0
x_end = 50
x_size = x_end - x_start
param_min = 3
param_max = 7
separation_factor = 0.8
noise_magnitude = 0.01
batch_size = 128
epochs = 30


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
    Z_type = []
    Z_params = []

    x = numpy.linspace(x_start, x_end, x_size)

    for i in range(examples_amount):
        y = numpy.zeros(x_size)

        # add target function
        function_id = numpy.random.randint(0, len(functions))
        function = functions[function_id]

        # центр функции в середине вектора
        offset = 1/2 * (x_end - x_start) + x_start
        target_random_param = random.random()
        target_param = param_min + (param_max - param_min) * target_random_param

        y = y + function(x, offset, target_param)

        # add another (noise) functions
        for j in range(random.randint(0, 3)):
            function_id = numpy.random.randint(0, len(functions))
            function = functions[function_id]

            random_offset = random.random()
            offset = random_offset * (x_end - x_start) + x_start

            random_param = random.random()
            param = param_min + (param_max - param_min) * random_param

            y = y + function(x, offset, param)

        y_max = numpy.max(y)
        scale = y_max
        if y_max > 0:
            y = y / y_max

        y = y.reshape((-1, 1))

        z_type = numpy.zeros(len(functions))
        z_type[function_id] = 1

        z_params = numpy.array([target_random_param, scale])

        X.append(x)
        Y.append(y)
        Z_type.append(z_type)
        Z_params.append(z_params)

    X = numpy.array(X)
    Y = numpy.array(Y)
    Z_type = numpy.array(Z_type)
    Z_params = numpy.array(Z_params)

    return X, Y, (Z_type, Z_params)


def get_data(data=None):
    _, y, z = data or generate_data()

    separator = int(examples_amount * separation_factor)

    y_train = y[:separator]
    z_train = [arr[:separator] for arr in z]

    y_test = y[separator:]
    z_test = [arr[separator:] for arr in z]

    # add noise
    #y_test = random.normal(y_test, noise_magnitude)

    return (y_train, z_train), (y_test, z_test)


def create_model():
    input = layers.Input(shape=(x_size, 1))
    x = layers.Conv1D(32, 5, padding='same')(input)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 5, padding='same')(x)
    x = layers.Activation('relu')(x)
    conv_out = layers.Flatten()(x)

    # type
    x = layers.Dense(len(functions))(conv_out)
    output_1 = layers.Activation('softmax')(x)

    # x0, a, scale
    x = layers.Dense(32)(conv_out)
    x = layers.Activation('tanh')(x)
    x = layers.Dense(2)(x)
    output_2 = layers.Activation('relu')(x)

    outputs = [output_1, output_2]
    model = models.Model(inputs=input, outputs=outputs)

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
    model.fit(y_train, z_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(y_test, z_test),
        shuffle=True)


def test_model(model, y_test, z_test):
    result = model.evaluate(y_test, z_test)

    return result


def test_predict(model, y_test, z_test):
    predict = model.predict(y_test)

    for i, (z, p) in enumerate(zip(z_test, predict)):
        print(i, ':', z, p)


def show_predict(model, y_test, z_test):
    predict = model.predict(y_test)

    #import pdb; pdb.set_trace()

    for i, (type, params) in enumerate(zip(*predict)):
        function_id = numpy.argmax(type)
        function = functions[function_id]

        x0 = 1/2 * (x_end - x_start) + x_start
        a, scale = params
        a = param_min + (param_max - param_min) * a
        scale += 0.001

        print('Prediction:', function.__name__, x0, a, scale)

        x = numpy.linspace(x_start, x_end, x_size)
        y = function(x, x0, a, scale)
        #y = y / numpy.max(y)

        pyplot.plot(x, y_test[i])
        pyplot.plot(x, y)
        pyplot.show()


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
    (y_train, z_train), (y_test, z_test) = get_data()

    model = create_model()
    model.summary()

    compile_model(model)
    train_model(model, y_train, z_train, y_test, z_test)

    result = test_model(model, y_test, z_test)
    print('Model acc:', result)

    #test_predict(model, y_test, z_test)

    save_model(model)

    show_predict(model, y_test, z_test)


def test_gaussian():
    x = numpy.linspace(x_start, x_end, x_size)
    g = gaussian(x, (x_start + x_end) / 2)

    pyplot.plot(x, g)
    pyplot.show()


def test_lagrange():
    x = numpy.linspace(x_start, x_end, x_size)
    g = lagrange(x, (x_start + x_end) / 2)

    pyplot.plot(x, g)
    pyplot.show()


def show_generated_data():
    X, Y, Z = generate_data()

    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        print('Out:', z)

        pyplot.plot(x, y)
        pyplot.show()
        
        print('Press ENTER to show next example:')
        input()


def show_train_data():
    data = X, Y, Z = generate_data()
    (y_train, z_train), (y_test, z_test) = get_data(data)

    for i, (x, y, z) in enumerate(zip(X, y_train, z_train)):
        print('Out:', z)

        pyplot.plot(x, y)
        pyplot.show()
        
        print('Press ENTER to show next example:')
        input()


def show_test_data():
    data = X, Y, Z = generate_data()
    (y_train, z_train), (y_test, z_test) = get_data(data)

    for i, (x, y, z) in enumerate(zip(X, y_test, z_test)):
        print('Out:', z)

        pyplot.plot(x, y)
        pyplot.show()
        
        print('Press ENTER to show next example:')
        input()


if __name__ == '__main__':
    #test_gaussian()
    #test_lagrange()
    #show_generated_data()
    #show_test_data()
    main()

