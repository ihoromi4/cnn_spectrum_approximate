import numpy
from numpy import random

from keras import models
from keras import layers

from matplotlib import pyplot


examples_amount = 10000
x_start = 0
x_end = 300
x_size = x_end - x_start
separation_factor = 0.8
noise_magnitude = 0.01
batch_size = 32
epochs = 10


def gaussian(x, mu, sigma=1):
    """
    x - array of arguments
    mu - expected value in probability theory
    sigma - standard deviation in statistics
    """
    return numpy.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * numpy.sqrt(2 * numpy.pi))


def lagrange(x, x0, gamma=1):
    return 1.0 / (numpy.pi * gamma * (1 + ((x - x0) / gamma)**2))


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
        function_id = numpy.random.randint(0, len(functions))
        function = functions[function_id]

        offset = random.random() * (x_end - x_start) + x_start
        param = 3 + 7 * random.random()

        y = function(x, offset, param)
        y = y.reshape((-1, 1))
        y_max = numpy.max(y)
        y = y / y_max

        z = numpy.zeros(len(functions))
        z[function_id] = 1

        X.append(x)
        Y.append(y)
        Z.append(z)

    X = numpy.array(X)
    Y = numpy.array(Y)
    Z = numpy.array(Z)

    return X, Y, Z


def get_data(data=None):
    _, y, z = data or generate_data()

    separator = int(examples_amount * separation_factor)

    y_train = y[:separator]
    z_train = z[:separator]

    y_test = y[separator:]
    z_test = z[separator:]

    # add noise
    y_test = random.normal(y_test, noise_magnitude)

    return (y_train, z_train), (y_test, z_test)


def create_model():
    input = layers.Input(shape=(x_size, 1))
    x = layers.Conv1D(32, 5, padding='same')(input)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(len(functions))(x)
    output = layers.Activation('softmax')(x)

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


def main():
    (y_train, z_train), (y_test, z_test) = get_data()

    model = create_model()
    model.summary()

    compile_model(model)
    train_model(model, y_train, z_train, y_test, z_test)

    result = test_model(model, y_test, z_test)
    print('Model acc:', result)

    test_predict(model, y_test, z_test)


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

