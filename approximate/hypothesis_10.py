"""
Определение типа функции, смещения и параметров, при условии, что данные состоят только из одной функции.
"""

import numpy
from numpy import random

from keras import models
from keras import layers

from matplotlib import pyplot


examples_amount = 100000
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
    Z_type = []
    Z_params = []

    x = numpy.linspace(x_start, x_end, x_size)

    for i in range(examples_amount):
        function_id = numpy.random.randint(0, len(functions))
        function = functions[function_id]

        offset = random.random() * (x_end - x_start) + x_start
        param = 3 + 7 * random.random()

        y = function(x, offset, param)
        y = y.reshape((-1, 1))
        y_max = numpy.max(y)
        y = y / y_max

        z_type = numpy.zeros(len(functions))
        z_type[function_id] = 1

        z_params = numpy.array([(offset - x_start) / (x_end - x_start), param, 1.0])

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
    x = layers.Activation('relu')(x)
    x = layers.Dense(3)(x)
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

        x0, a, scale = params
        x0 = x0 * (x_end - x_start) + x_start

        print('Prediction:', function.__name__, x0, a, scale)

        x = numpy.linspace(x_start, x_end, x_size)
        y = function(x, x0, a, scale)
        y = y / numpy.max(y)

        pyplot.plot(x, y_test[i])
        pyplot.plot(x, y)
        pyplot.show()    


def main():
    (y_train, z_train), (y_test, z_test) = get_data()

    model = create_model()
    model.summary()

    compile_model(model)
    train_model(model, y_train, z_train, y_test, z_test)

    #result = test_model(model, y_test, z_test)
    #print('Model acc:', result)

    #test_predict(model, y_test, z_test)

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

