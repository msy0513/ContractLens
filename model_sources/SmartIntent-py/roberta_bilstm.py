import tensorflow as tf
from tensorflow import keras
import dataset as db
import dataset_ljw as db2
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
argv = sys.argv[1:]

UNIT = 128
BATCH = 80
BATCH_SIZE = 200
EPOCH = 100
DROP = 0.5
MAX_SEQ = 256
DIM = 768
PAD = 0.0

MODEL_PATH = './models/roberta_bilstm.keras'


def buildModel():
    model = keras.Sequential()
    model.add(keras.layers.Masking(
        mask_value=PAD, input_shape=(MAX_SEQ, DIM)))
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(units=UNIT, return_sequences=False)))
    # model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.Dropout(DROP))
    model.add(keras.layers.Dense(10, activation='sigmoid'))
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
            x.append([PAD] * DIM)
        arr.append(x[:MAX_SEQ])
    return arr


def train(batch=BATCH, batch_size=BATCH_SIZE, epoch=EPOCH, start=1):
    model = loadModel()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryFocalCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ]
    )

    id = start
    print("Batch:", batch)
    print("Batch Size:", batch_size)
    print("Total:", batch * batch_size)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir="./logs", histogram_freq=1)
    # 动态学习率调度
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_delta=0.001)
    # 早停机制
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.0005, patience=10, restore_best_weights=True)

    callbacks = [tensorboard_callback]

    while batch > 0:
        print("Current Batch:", batch)
        print("Start Id:", id)
        x, y, id = db.getBatch(id, batch_size)
        print("End Id:", id)

        tx = tf.convert_to_tensor(pad(x))
        ty = tf.convert_to_tensor(y)
        print(tx)
        print(ty)

        model.fit(tx, ty, batch_size=batch_size, epochs=epoch,
                  shuffle=True, callbacks=callbacks)

        batch -= 1
        id += 1
        model.save(MODEL_PATH)


def train_balance(batch=20, epoch=100, start=1, end=17197):
    model = loadModel()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.BinaryFocalCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ]
    )

    id = start
    print("Batch:", batch)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir="./logs", histogram_freq=1)

    callbacks = [tensorboard_callback]

    while batch > 0:
        print("Current Batch:", batch)
        print("Start Id:", id)
        x, y, id = db2.getBatch_v2(start, end)
        print("End Id:", id)

        tx = tf.convert_to_tensor(pad(x))
        ty = tf.convert_to_tensor(y)
        print(tx)
        print(ty)

        model.fit(tx, ty, epochs=epoch, shuffle=True, callbacks=callbacks)

        batch -= 1
        model.save(MODEL_PATH)


def evaluate(start=20000, batch=10000):
    model = loadModel()
    print("Batch:", batch)
    print("Start:", start)
    x, y, id = db.getBatch(start, batch)
    print("End:", id)

    tx_eval = tf.convert_to_tensor(pad(x))
    ty_eval = tf.convert_to_tensor(y)

    y_pred = model.predict(tx_eval)
    y_pred_binary = tf.round(y_pred)

    # 初始化每个类别的指标列表
    category_accuracy = []
    category_precision = []
    category_recall = []
    category_f1 = []

    # 对每个类别计算指标
    for category, index in db.TYPE.items():
        accuracy = keras.metrics.BinaryAccuracy()(
            ty_eval[:, index], y_pred_binary[:, index])
        precision = keras.metrics.Precision()(
            ty_eval[:, index], y_pred_binary[:, index])
        recall = keras.metrics.Recall()(
            ty_eval[:, index], y_pred_binary[:, index])
        f1 = 2 * (precision * recall) / (precision +
                                         recall + keras.backend.epsilon())

        category_accuracy.append(accuracy.numpy())
        category_precision.append(precision.numpy())
        category_recall.append(recall.numpy())
        category_f1.append(f1.numpy())

        print(f"Category '{category}':")
        print("Accuracy:", category_accuracy[-1])
        print("Precision:", category_precision[-1])
        print("Recall:", category_recall[-1])
        print("F1 Score:", category_f1[-1])
        print("----------------------------------------------------------")

    print("==========================================================")
    print("Average Metrics")
    print("Accuracy:", sum(category_accuracy) / len(db.TYPE))
    print("Precision:", sum(category_precision) / len(db.TYPE))
    print("Recall:", sum(category_recall) / len(db.TYPE))
    print("F1 Score:", sum(category_f1) / len(db.TYPE))
    print("==========================================================")

    accuracy = keras.metrics.BinaryAccuracy()(ty_eval, y_pred_binary)
    precision = keras.metrics.Precision()(ty_eval, y_pred_binary)
    recall = keras.metrics.Recall()(ty_eval, y_pred_binary)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("==========================================================")
    print("Total Metrics")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("==========================================================")


if argv and argv[0] == 'summary':
    summary()
if argv and argv[0] == 'train':
    train()
if argv and argv[0] == 'train2':
    train_balance()
if argv and argv[0] == 'evaluate':
    evaluate()
