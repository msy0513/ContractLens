import os
import sys
import dataset as db
import tensorflow as tf
import dataset_ljw as db2
from tensorflow import keras


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


MODEL_PATH = './models/smartbert_bilstm_ljw_v22_v2.keras'
# calculate by the data count of entire dataset


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

import tensorflow as tf

"""
# 自定义学习率调度器类
class ClassBalancedLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, class_counts, total_samples, min_lr=1e-6):
        super(ClassBalancedLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr   # 初始学习率
        self.class_counts = class_counts  # 类别的样本数
        self.total_samples = total_samples  # 数据集中的总样本数
        self.min_lr = min_lr   # 设置最小学习率
    
    def on_epoch_begin(self, epoch, logs=None):
        # 计算每个类别的学习率：根据类别样本数相对总样本数的比例调整学习率
        class_balanced_lr = {}
        for class_name, count in self.class_counts.items():
            # 学习率根据类别样本数进行缩放，数据量越少的类别学习率越高
            if count > 0:
                class_balanced_lr[class_name] = self.initial_lr * (self.total_samples / (count * len(self.class_counts)))
            else:
                class_balanced_lr[class_name] = self.min_lr  # 避免除以零的情况
        
        # 设置学习率的下限
        for class_name, lr in class_balanced_lr.items():
            if lr < self.min_lr:
                class_balanced_lr[class_name] = self.min_lr
        
        print(f"Class Balanced Learning Rates for epoch {epoch}: {class_balanced_lr}")
        # 如果需要，可以在这里将不同的学习率应用到模型的相关层
"""


# 增加一个最大学习率限制
max_lr = 0.01  # 设置一个合理的上限

class ClassBalancedLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, class_counts, total_samples, min_lr=1e-6, max_lr=0.01):
        super(ClassBalancedLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.class_counts = class_counts
        self.total_samples = total_samples
        self.min_lr = min_lr
        self.max_lr = max_lr  # 添加最大学习率
    
    def on_epoch_begin(self, epoch, logs=None):
        class_balanced_lr = {}
        for class_name, count in self.class_counts.items():
            if count > 0:
                lr = self.initial_lr * (self.total_samples / (count * len(self.class_counts)))
                # 限制学习率的上下限
                lr = max(min(lr, self.max_lr), self.min_lr)
                class_balanced_lr[class_name] = lr
            else:
                class_balanced_lr[class_name] = self.min_lr
        
        print(f"Class Balanced Learning Rates for epoch {epoch}: {class_balanced_lr}")


"""
# 引入平滑因子，避免学习率变化过大
smooth_factor = 0.9

class ClassBalancedLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, class_counts, total_samples, min_lr=1e-6, max_lr=0.01, smooth_factor=0.9):
        super(ClassBalancedLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.class_counts = class_counts
        self.total_samples = total_samples
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.smooth_factor = smooth_factor  # 引入平滑因子
    
    def on_epoch_begin(self, epoch, logs=None):
        class_balanced_lr = {}
        for class_name, count in self.class_counts.items():
            if count > 0:
                lr = self.initial_lr * (self.total_samples / (count * len(self.class_counts)))
                # 限制学习率并引入平滑因子
                lr = max(min(lr * self.smooth_factor, self.max_lr), self.min_lr)
                class_balanced_lr[class_name] = lr
            else:
                class_balanced_lr[class_name] = self.min_lr
        
        print(f"Class Balanced Learning Rates for epoch {epoch}: {class_balanced_lr}")
"""


def train(batch=BATCH, batch_size=BATCH_SIZE, epoch=EPOCH, start=1):
    model = loadModel()
    model.compile(optimizer=keras.optimizers.Adam(),
                  # loss=keras.losses.BinaryFocalCrossentropy(),
                  loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.Precision(),
                           keras.metrics.Recall()])

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir="./logs", histogram_freq=1)
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_delta=0.001)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.0005, patience=10, restore_best_weights=True)

    callbacks = [tensorboard_callback]
    # callbacks = [tensorboard_callback, lr_schedule, early_stopping]

    id = start

    while batch > 0:
        print("Current Batch:", batch)
        print("Start Id:", id)
        x, y, id = db.getBatch(id, batch_size)
        print("End Id:", id)

        # 动态计算当前 batch 的类别分布
        class_counts = {key: 0 for key in db.TYPE.keys()}  # 初始化类别计数
        for label in y:
            for idx, value in enumerate(label):
                if value == 1:
                    class_name = list(db.TYPE.keys())[idx]
                    class_counts[class_name] += 1
        
        # 打印当前 batch 的类别分布
        print(f"Class counts for current batch: {class_counts}")
        
        total_samples = sum(class_counts.values())  # 计算当前批次的总样本数
        
        # 实例化自定义的类均衡学习率调度器
        lr_scheduler = ClassBalancedLearningRateScheduler(initial_lr=0.001, class_counts=class_counts, total_samples=total_samples)

        tx = tf.convert_to_tensor(pad(x))  # 数据预处理
        ty = tf.convert_to_tensor(y)

        model.fit(tx, ty, batch_size=batch_size, epochs=epoch,
                  shuffle=True, callbacks=callbacks + [lr_scheduler])  # 添加学习率调度器
        
        batch -= 1
        id += 1
        model.save(MODEL_PATH)

    """
    while batch > 0:
        print("Current Batch:", batch)
        print("Start Id:", start)
        x, y, id = db2.getBatch_v2(1, 17197)
        print("End Id:", start)

        # 动态计算当前 batch 的类别分布
        class_counts = {key: 0 for key in db.TYPE.keys()}  # 初始化类别计数
        for label in y:
            for idx, value in enumerate(label):
                if value == 1:
                    class_name = list(db.TYPE.keys())[idx]
                    class_counts[class_name] += 1
        
        # 打印当前 batch 的类别分布
        print(f"Class counts for current batch: {class_counts}")
        
        total_samples = sum(class_counts.values())  # 计算当前批次的总样本数
        
        # 实例化自定义的类均衡学习率调度器
        lr_scheduler = ClassBalancedLearningRateScheduler(initial_lr=0.001, class_counts=class_counts, total_samples=total_samples)

        tx = tf.convert_to_tensor(pad(x))  # 数据预处理
        ty = tf.convert_to_tensor(y)

        model.fit(tx, ty, batch_size=batch_size, epochs=epoch,
                  shuffle=True, callbacks=callbacks + [lr_scheduler])  # 添加学习率调度器

        batch -= 1
        model.save(MODEL_PATH)
"""





"""
def train(batch=BATCH, batch_size=BATCH_SIZE, epoch=EPOCH, start=1):
    model = loadModel()
    model.compile(optimizer=keras.optimizers.Adam(),
                  # loss=keras.losses.BinaryFocalCrossentropy(),
                  loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.Precision(),
                           keras.metrics.Recall()])

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
    
    
    while batch > 0:
        print("Current Batch:", batch)
        print("Start Id:", id)
        x, y, id = db2.getBatch_v2(1, 17197)
        print("End Id:", id)

        tx = tf.convert_to_tensor(pad(x))
        ty = tf.convert_to_tensor(y)
        print(tx)
        print(ty)

        model.fit(tx, ty, batch_size=batch_size, epochs=epoch,
                  shuffle=True, callbacks=callbacks)

        batch -= 1
        # id += 1
        model.save(MODEL_PATH)
    """
        


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

    # Compute the evaluation metrics
    accuracy = keras.metrics.BinaryAccuracy()(ty_eval, y_pred_binary)
    precision = keras.metrics.Precision()(ty_eval, y_pred_binary)
    recall = keras.metrics.Recall()(ty_eval, y_pred_binary)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("==========================================================")
    print("Total Metrics")
    print(f"Model saved and evaluated from: {MODEL_PATH}")
    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("==========================================================")


if argv and argv[0] == 'summary':
    summary()
if argv and argv[0] == 'train':
    train()
    evaluate()
if argv and argv[0] == 'evaluate':
    evaluate()


"""
==========================================================
python ./smartbert_bilstm_ljw_2.py train
python ./smartbert_bilstm_ljw_2.py evaluate
==========================================================
"""


