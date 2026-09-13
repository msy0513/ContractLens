import os
import tensorflow as tf
from tensorflow import keras
import dataset as db
import sys
argv = sys.argv[1:]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


DIM = 768
VOC = 50264
PAD = 0.0
BATCH = 100
BATCH_SIZE = 200
EPOCH = 100
MAX_SEQ = 256
DROP = 0.5  # best is 0.5

MODEL_PATH = './models/smartbert_cnn.h5'


def buildModel():
    inputs = keras.layers.Input((MAX_SEQ, DIM))
    conv1 = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    pool1 = keras.layers.MaxPool1D(pool_size=(2))(conv1)
    conv2 = keras.layers.Conv1D(
        filters=128, kernel_size=4, padding='same', activation='relu')(pool1)
    pool2 = keras.layers.MaxPool1D(pool_size=(2))(conv2)
    conv3 = keras.layers.Conv1D(
        filters=256, kernel_size=5, padding='same', activation='relu')(pool2)
    pool3 = keras.layers.MaxPool1D(pool_size=(2))(conv3)
    flat = keras.layers.Flatten()(pool3)
    dense = keras.layers.Dense(64, activation='relu')(flat)
    drop = keras.layers.Dropout(DROP)(dense)
    outputs = keras.layers.Dense(10, activation='sigmoid')(drop)
    model = keras.Model(inputs=inputs, outputs=outputs)
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
        while (len(x) < MAX_SEQ):
            x.append([PAD]*DIM)
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
            data = db.getXY2(id)
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
        batch = batch-1
        model.save(MODEL_PATH)


def evaluate(start=21000, batch=10000):
    model = loadModel()
    id = start
    print("Batch:", batch)
    print("Start:", start)
    xs = []
    ys = []
    while (batch > 0):
        print("Current Batch:", batch)
        print("Current Id:", id)
        data = db.getXY2(id)
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
