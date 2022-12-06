import csv
import data_def
import numpy as np
import tensorflow as tf


def ohe(num):
    re = [0] * 10
    re[num] = 1
    return re


def create_x_y_train_from_csv():
    mnist_train_in_csv = open("./saved_data/mnist_train.csv")
    type(mnist_train_in_csv)

    csvreader = csv.reader(mnist_train_in_csv)
    header = next(csvreader)

    mnist_train = [[int(i) for i in row] for row in csvreader]
    y = [ohe(int(row.pop(0))) for row in mnist_train]
    x = mnist_train

    data_def.set_file('./saved_data/x_train.pkl', x)
    data_def.set_file('./saved_data/y_train.pkl', y)


def create_x_y_test_from_csv():
    mnist_test_in_csv = open("./saved_data/mnist_test.csv")
    type(mnist_test_in_csv)

    csvreader = csv.reader(mnist_test_in_csv)
    header = next(csvreader)

    mnist_train = [[int(i) for i in row] for row in csvreader]
    y = [ohe(int(row.pop(0))) for row in mnist_train]
    x = mnist_train

    data_def.set_file('./saved_data/x_test.pkl', x)
    data_def.set_file('./saved_data/y_test.pkl', y)


def create_model():
    x_train = np.array(data_def.get_file('./saved_data/x_train.pkl'))
    y_train = np.array(data_def.get_file('./saved_data/y_train.pkl'))
    x_test = data_def.get_file('./saved_data/x_test.pkl')
    y_test = data_def.get_file('./saved_data/y_test.pkl')

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(32, activation='sigmoid'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ]
    )

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

    data_def.set_model('./saved_data/my_model', model)


create_model()
