import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import dataset as db
import sys
import os
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
argv = sys.argv[1:]

UNIT = 64
DIM = 512
VOC = 50264
PAD = 1
PAD_TKN = 512
BATCH = 200
BATCH_SIZE = 50
EPOCH = 50
MAX_SEQ = 256
DROP = 0.5  # best is 0.5
MODEL_PATH = './models/bilstm.h5'


def buildModel():
    model = models.Sequential()
    model.add(layers.Input((MAX_SEQ, PAD_TKN)))
    model.add(layers.Masking(PAD))
    # model.add(layers.Embedding(input_dim=VOC, output_dim=DIM, mask_zero=True))
    # model.add(layers.AveragePooling2D(pool_size=(1, PAD_TKN)))
    # model.add(layers.Reshape((MAX_SEQ, DIM)))
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(UNIT, return_sequences=True)))
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(UNIT, return_sequences=False)))
    model.add(layers.Dropout(DROP))  # Dropout 层
    model.add(layers.Dense(10, activation='sigmoid'))  # 输出层
    model.summary()
    return model


def loadModel():
    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(e)
        return buildModel()


def summary():
    model = loadModel()
    model.summary()


def pad(xs):
    arr = []
    for x in xs:
        while len(x) < MAX_SEQ:
            x.append([PAD] * PAD_TKN)
        arr.append(x[:MAX_SEQ])
    return arr


def train(batch=BATCH, batch_size=BATCH_SIZE, epoch=EPOCH, start=1):
    model = loadModel()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.Precision(),
                           keras.metrics.Recall()])
    id = start
    print("Batch:", batch)
    print("Batch Size:", batch_size)
    print("Total:", batch*batch_size)
    while (batch > 0):
        xs = []
        ys = []
        print("Current Batch:", batch)
        print("Current Id:", id)
        while (len(xs) < batch_size):
            data = db.getXY(id)
            id = id+1
            if (data == None):
                continue
            xs.append(data['x'])
            ys.append(data['y'])
        tx = tf.convert_to_tensor(pad(xs))
        ty = tf.convert_to_tensor(ys)
        print(tx)
        print(ty)
        model.fit(tx, ty, batch_size=batch_size,
                  epochs=epoch, shuffle=True)
        model.save(MODEL_PATH)
        batch = batch-1


def evaluate(start=20000, batch=10000):
    model = loadModel()
    id = start
    print("Batch:", batch)
    print("Start:", start)
    xs = []
    ys = []
    while (batch > 0):
        print("Current Batch:", batch)
        print("Current Id:", id)
        data = db.getXY(id)
        id = id+1
        if (data == None):
            continue

        xs.append(data['x'])
        ys.append(data['y'])
        batch = batch-1

    tx_eval = tf.convert_to_tensor(pad(xs))
    ty_eval = tf.convert_to_tensor(ys)

    # Make predictions on the evaluation data
    # Device context manager
    y_pred = model.predict(tx_eval)

    # Convert the predictions to binary labels
    y_pred_binary = tf.round(y_pred)
    print(ty_eval)
    print(y_pred_binary)

    # Compute the evaluation metrics
    accuracy = keras.metrics.BinaryAccuracy()(ty_eval, y_pred_binary)
    precision = keras.metrics.Precision()(ty_eval, y_pred_binary)
    recall = keras.metrics.Recall()(ty_eval, y_pred_binary)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("==========================================================")
    print("Total")
    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("==========================================================")


if (argv[0] == 'summary'):
    summary()
if (argv[0] == 'train'):
    train()
if (argv[0] == 'evaluate'):
    evaluate()
