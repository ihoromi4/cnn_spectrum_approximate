import numpy
from keras import models
from keras import layers
from matplotlib import pyplot

examples_amount = 10000
x_start = 250
x_end = 300
x_size = x_end - x_start
separation_factor = 0.8
batch_size = 32
epochs = 10


def gaussian(x, offset):
    return numpy.exp(-(x - offset)**2/10)


def lagrange(x, offset):
    return 1.0 / (1 + ((x - offset)/3)**2)


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

        offset = numpy.random.random() * (x_end - x_start) + x_start

        y = function(x, offset)
        y = y.reshape((-1, 1))

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

    return (y_train, z_train), (y_test, z_test)


def create_model():
    input = layers.Input(shape=(x_size, 1))
    x = layers.Conv1D(32, 5, padding='same')(input)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
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


def test_generated_data():
    X, Y, Z = generate_data()

    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        print('Out:', z)

        pyplot.plot(x, y)
        pyplot.show()
        
        print('Press ENTER to show next example:')
        input()


if __name__ == '__main__':
    #test_gaussian()
    #test_lagrange()
    #test_generated_data()
    main()

