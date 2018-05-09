import os
import glob
import datetime
import json
import hashlib

import numpy
from numpy import random
from keras import models
from keras import layers
from keras import initializers
from keras import backend as K

from matplotlib import pyplot

# train
batch_size = 32
steps_per_epoch = 250
epochs = 1000

# data
x_start = 0
x_end = 400
x_size = x_end - x_start

param_min = 5
param_max = 30
param_diff = param_max - param_min

CUSTOM_OBJECTS = {}

def custom_object(func):
    CUSTOM_OBJECTS[func.__name__] = func
    return func


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


@custom_object
def attention_loss(y_true, y_pred):
    #x = numpy.arange(0, 32, 1)
    #f = lagrange(x, 5, 2)
    #f = f / numpy.max(f)
    #filters = K.constant(f.reshape(f.shape[0], 1, 1))

    filters = K.constant(numpy.ones((32, 1, 1)))

    attention = K.clip(K.conv1d(y_true, filters, 1, 'same'), 0, 1)
    diff = (y_true - y_pred) * attention

    return K.mean(K.square(diff))


@custom_object
def sharp_loss(y_true, y_pred):
    diff = y_true - y_pred
    filters = K.constant(numpy.array([-0.25, -0.25, 2, -0.25, -0.25]).reshape((5, 1, 1)))
    diff = K.conv1d(diff, filters, 1, 'same')

    return K.mean(K.square(diff))


@custom_object
def exp_loss(y_true, y_pred):
    return K.mean(K.exp(K.square(y_true - y_pred))) - 1


@custom_object
def adaptive_loss(y_true, y_pred):
    diff = y_true - y_pred
    f = K.sum(K.abs(diff) * y_true) / K.sum(y_true)
    f = K.sqrt(f)
    k = diff * (1 - f) + diff * y_true * f
    return K.mean(K.square(k))


@custom_object
def true_accuracy(y_true, y_pred):
    return K.sum(y_pred * y_true) / K.sum(y_true)


@custom_object
def area_accuracy(y_true, y_pred):
    kernel_size = 15
    filters = K.constant(numpy.ones((kernel_size, 1, 1)))
    attention = K.clip(K.conv1d(y_true, filters, 1, 'same'), 0, 1)
    diff = y_true - y_pred

    return 1 - K.sum(K.abs(diff) * attention) / K.sum(attention)


def one_random_spectrum():
    x = numpy.linspace(x_start, x_end, x_size)
    y = numpy.zeros(x_size)
    z = numpy.zeros(x_size)
    peak_number = random.randint(0, 5)

    for peak in range(peak_number):
        function_id = random.randint(0, len(functions))
        function = functions[function_id]

        r_offset = random.random()
        offset = r_offset * x_size + x_start
        param = param_min + param_diff * random.random()

        y = y + function(x, offset, param)

        index = int(r_offset * x_size)
        z[index] = 1

    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))

    y_max = numpy.max(y)
    if y_max > 0:
        y = y / y_max

    return y, z


def one_random_mixed_spectrum():
    x = numpy.linspace(x_start, x_end, x_size)
    y = numpy.zeros(x_size)
    z = numpy.zeros(x_size)
    peak_number = random.randint(1, 4)
    center = x_start + x_size / 2
    magnitude = 50

    for peak in range(peak_number):
        function_id = random.randint(0, len(functions))
        function = functions[function_id]

        r_offset = random.random() - 0.5
        offset = center + magnitude * r_offset
        param = param_min + param_diff * random.random()

        y = y + function(x, offset, param)

        index = int(x_size / 2 + magnitude * r_offset)
        z[index] = 1

    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))

    y_max = numpy.max(y)
    if y_max > 0:
        y = y / y_max

    return y, z

one_random_spectrum = one_random_mixed_spectrum


def specturm_generator(batch_size):
    while True:
        data = [one_random_spectrum() for i in range(batch_size)]
        Y, Z = map(numpy.array, zip(*data))

        yield Y, Z



def Conv1D(*args, **kwargs):
    #kernel_initializer = initializers.RandomUniform(minval=0.005, maxval=0.005)
    kernel_initializer = 'glorot_uniform'

    return layers.Conv1D(
        *args,
        padding='same',
        #dilation_rate=2,
        activation='relu', 
        use_bias=False, 
        kernel_initializer=kernel_initializer,
        **kwargs)


def create_model():
    input = x = layers.Input(shape=(None, 1))
    x = Conv1D(32, 3)(x)
    x = Conv1D(32, 3)(x)
    x = Conv1D(64, 3)(x)
    x = Conv1D(64, 3)(x)
    x = Conv1D(64, 3)(x)
    x = Conv1D(64, 3)(x)
    x = Conv1D(32, 5)(x)
    x = Conv1D(32, 5)(x)
    x = Conv1D(32, 5)(x)
    x = Conv1D(32, 5)(x)
    x = Conv1D(16, 7)(x)
    x = Conv1D(8, 7)(x)
    x = Conv1D(8, 7)(x)
    x = Conv1D(8, 7)(x)

    x = Conv1D(1, 1)(x)

    model = models.Model(inputs=input, outputs=x)

    return model


def compile_model(model):
    #loss = 'mean_squared_error'
    #loss = 'mean_absolute_error'
    #loss = 'binary_crossentropy'
    #loss = attention_loss
    #loss = sharp_loss
    #loss = exp_loss
    loss = adaptive_loss

    optimizer = 'adam'

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[true_accuracy, area_accuracy])


def train_model(model, generator):
    return model.fit_generator(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=generator,
        validation_steps=4)


def save_train_history(history, file_name):
    name, ext = os.path.splitext(__file__)
    MODELS_DIR = 'history_' + name

    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    file_path = os.path.join(MODELS_DIR, file_name + '.json')
    with open(file_path, 'w') as f:
        json.dump(history.history, f)


def save_model(model):
    """Model name: <file-name>_<date>_<model-arch-hash>"""

    name, ext = os.path.splitext(__file__)
    MODELS_DIR = 'models_' + name

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


def load_model_choice():
    name, ext = os.path.splitext(__file__)
    MODELS_DIR = 'models_' + name
    search_pattern = '*.h5'
    search_pattern = os.path.join(MODELS_DIR, search_pattern)

    files = glob.glob(search_pattern)

    if len(files) == 0:
        print('No saved models')
        input('Press ENTER to continue...')
        return

    print('Saved models:')
    print('[-1] Create new model')
    for i, f in enumerate(files):
        print('[%i]' % i, f)

    index = int(input('Choice model file: '))
    
    if index == -1:
        return None

    model_path = files[index]
    model = models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    return model


def plot_test(model):
    while True:
        y, z = one_random_spectrum()
        y = y.reshape((1,) + y.shape)
        z = z.reshape((1,) + z.shape)

        x = numpy.linspace(x_start, x_end, x_size)

        p = model.predict(y)
        
        #p_max = numpy.max(p)
        #if p_max > 0:
        #    p = p / p_max

        pyplot.plot(x, y.flatten())
        pyplot.plot(x, z.flatten())
        pyplot.plot(x, p.flatten())

        try:
            pyplot.show()
        except KeyboardInterrupt:
            break


def main():
    model = load_model_choice()
    if not model:
        model = create_model()

    model.summary()

    compile_model(model)

    generator = specturm_generator(batch_size)

    try:
        history = train_model(model, generator)

        loss = 'adaptive_loss'
        name = '{loss}_{epochs}_epochs_{batch_size}_batch_size'.format(
            loss=loss,
            epochs=epochs,
            batch_size=batch_size)
        save_train_history(history, name)
    except KeyboardInterrupt:
        plot_test(model)

        choice = input('\nDo you want to save model? [yes/NO]: ')
        if choice in ('YES', 'Yes', 'yes', 'Y', 'y'):
            save_model(model)
            return
        else:
            raise

    plot_test(model)

    save_model(model)


if __name__ == '__main__':
    main()

