import numpy
from numpy import random
from scipy import optimize
from matplotlib import pyplot

from keras import models

examples_amount = 100
x_start = 0
x_end = 300
x_size = x_end - x_start
param_min = 0.3
param_max = 10
separation_factor = 0.8
noise_magnitude = 0.01


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


def generate_spectrum(data):
    x = numpy.linspace(x_start, x_end, x_size)
    y = numpy.zeros(x_size)

    for type, params in data:
        function = functions[type]
        y = y + function(x, *params)

    y_max = numpy.max(y)
    if y_max > 0:
        y = y / y_max

    return y


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


def load_model(model_path):
    return models.load_model(model_path)


def plot_maximums(y, maximums):
    for yi, mi in zip(y, maximums):
        #import pdb; pdb.set_trace()
        m = numpy.zeros(x_size)
        m[mi] = 1

        pyplot.plot(yi)
        pyplot.plot(m)
        pyplot.show()


def find_maximums(model, y):
    predict = model.predict(y)
    max_x, max_y = (predict > 0.2).nonzero()
    examples = predict.shape[0]
    maximums = [max_y[max_x == i] for i in range(examples)]

    return maximums


def find_params(model, y, maximums):
    result = []
    for m in maximums:
        start = m - 25
        end = m + 25

        max_i = len(y) - 1
        input = y[max(0, start): min(max_i, end)]

        pad = (max(0, -start), max(0, end-max_i))
        pad = (pad, (0, 0))
        input = numpy.pad(input, pad, 'constant')
        input = input.reshape((1,) + input.shape)

        #import pdb; pdb.set_trace()
        type, params = model.predict(input)
        a, scale = params[0]
        type = numpy.argmax(type)
        x0 = m/x_size * (x_end - x_start) + x_start
        a = param_min + (param_max - param_min) * a
        scale += 0.00001
        result.append((type, (x0, a, scale)))

    return result


def get_fitfunc(config):
    def fitfunc(x, *args):
        if len(args) != 3 * len(config):
            raise TypeError('fitfunc() takes {argv} positional arguments but {argl} was given'.format(argv=3*len(config), argl=len(args)))

        r = None
        for type, _ in config:
            x0, param, scale, *args = args
            f = functions[type](x, x0, param, scale)
            r = r + f if r is not None else f
            
        r = r / numpy.max(r)
        return r

    return fitfunc


def specify_params_ols(y, config):
    #import pdb; pdb.set_trace()

    if len(config) == 0:
        return config

    y = y.flatten()

    fitfunc = get_fitfunc(config)

    x = numpy.linspace(x_start, x_end, x_size)
    x0 = numpy.concatenate([params for type, params in config])

    try:
        new_params = optimize.curve_fit(fitfunc, x, y, x0)
    except RuntimeError as err:
        print(RuntimeError, err)
        return config

    new_params = numpy.split(new_params[0], len(config))

    types = [type for type, params in config]

    new_config = list(zip(types, new_params))

    return new_config


def main():
    (y_train, z_train), (y_test, z_test) = get_dataset()

    DIR = 'models_hypothesis_23'
    h21 = load_model(DIR + '/hypothesis_21.h5')
    h22 = load_model(DIR + '/hypothesis_22.h5')

    maximums = find_maximums(h21, y_train)
    #plot_maximums(y_train, maximums)

    for i in range(examples_amount):
        params = find_params(h22, y_train[i], maximums[i])
        print('Params:', params)

        opt_params = specify_params_ols(y_train[i], params)

        s = generate_spectrum(params)
        opt_s = generate_spectrum(opt_params)
        print(s)

        pyplot.plot(y_train[i])
        #pyplot.plot(s)
        pyplot.plot(opt_s)
        pyplot.show()


if __name__ == '__main__':
    main()


