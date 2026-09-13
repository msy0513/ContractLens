import tensorflow as tf
from tensorflow import keras
import dataset as db
import dataset_ljw as db2
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
argv = sys.argv[1:]


UNIT = 128
BATCH = 80
BATCH_SIZE = 200
EPOCH = 100
DROP = 0.5
MAX_SEQ = 256
DIM = 768
PAD = 0.0


# 两次基础训练
# MODEL_PATH = './models/smartbert_bilstm_128base.keras'
# 一次补全训练（0.0001）
MODEL_PATH = './models/smartbert_bilstm_128balance_lr0.0001.keras'


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


if argv and argv[0] == 'summary':
    summary()
if argv and argv[0] == 'train':
    train()
    evaluate()
if argv and argv[0] == 'evaluate':
    evaluate()


"""
==========================================================
python ./smartbert_bilstm_ljw.py train
python ./smartbert_bilstm_ljw.py evaluate
==========================================================
十种意图的数据分布(ID: 1 - 17197)
fee 5267
disableTrading 1710
blacklist 1615
reflect 3793
maxTX 4931
mint 1355
honeypot 125
reward 466
rebase 81
maxSell 28
==========================================================
目前最好的模型结果(smartbert_bilstm copy.keras)
Accuracy: tf.Tensor(0.97505, shape=(), dtype=float32)
Precision: tf.Tensor(0.9269809, shape=(), dtype=float32)
Recall: tf.Tensor(0.89620465, shape=(), dtype=float32)                    
F1 Score: tf.Tensor(0.911333, shape=(), dtype=float32)
==========================================================
目前训练使用的超参数(可以训练两次)
UNIT = 128
BATCH = 80
BATCH_SIZE = 200
EPOCH = 100
DROP = 0.5
MAX_SEQ = 256
DIM = 768
PAD = 0.0
==========================================================
20_2. 使用老模型;
Accuracy: tf.Tensor(0.97469, shape=(), dtype=float32)
Precision: tf.Tensor(0.9105997, shape=(), dtype=float32)
Recall: tf.Tensor(0.91270006, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.9116487, shape=(), dtype=float32)

21_2. 使用老模型; 调整了loss函数的默认参数



22_2. 使用老模型; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器; 设置学习率上限(没加学习率调度和早停)



==========================================================
20. 使用老模型;
Accuracy: tf.Tensor(0.97174007, shape=(), dtype=float32)
Precision: tf.Tensor(0.9010129, shape=(), dtype=float32)
Recall: tf.Tensor(0.90151674, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.9012648, shape=(), dtype=float32)

21. 使用老模型; 调整了loss函数的默认参数
Accuracy: tf.Tensor(0.97160995, shape=(), dtype=float32)
Precision: tf.Tensor(0.9049435, shape=(), dtype=float32)
Recall: tf.Tensor(0.8956455, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.90027046, shape=(), dtype=float32)

22. 使用老模型; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器; 设置学习率上限(没加学习率调度和早停)
Accuracy: tf.Tensor(0.97342, shape=(), dtype=float32)
Precision: tf.Tensor(0.90957034, shape=(), dtype=float32)
Recall: tf.Tensor(0.90410286, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.90682834, shape=(), dtype=float32)

23. 使用老模型; 采用自定义类均衡学习率调度器; 设置学习率上限(没加学习率调度和早停)
Accuracy: tf.Tensor(0.9675801, shape=(), dtype=float32)
Precision: tf.Tensor(0.89662343, shape=(), dtype=float32)
Recall: tf.Tensor(0.87418747, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.8852634, shape=(), dtype=float32)

24. 使用老模型; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器; 设置学习率上限(加上了学习率调度和早停)
Accuracy: tf.Tensor(0.9522699, shape=(), dtype=float32)
Precision: tf.Tensor(0.8730631, shape=(), dtype=float32)
Recall: tf.Tensor(0.77975816, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.823777, shape=(), dtype=float32)

25. 使用老模型; 采用自定义类均衡学习率调度器; 设置学习率上限(加上了学习率调度和早停)
Accuracy: tf.Tensor(0.95564, shape=(), dtype=float32)
Precision: tf.Tensor(0.8752376, shape=(), dtype=float32)
Recall: tf.Tensor(0.80464107, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.8384559, shape=(), dtype=float32)
==========================================================
目前训练使用的超参数
UNIT = 128
BATCH = 40
BATCH_SIZE = 100
EPOCH = 50
DROP = 0.5
MAX_SEQ = 256
DIM = 768
PAD = 0.0
==========================================================
模型训练结果的记录(记得每次训练后更新MODEL_PATH)

1. 使用老模型;
Accuracy: tf.Tensor(0.94895, shape=(), dtype=float32)
Precision: tf.Tensor(0.86760944, shape=(), dtype=float32)
Recall: tf.Tensor(0.7589991, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.80967826, shape=(), dtype=float32)

2. 使用老模型; 采用随机平衡补全训练
Model saved and evaluated from: ./models/smartbert_bilstm_ljw_v2.keras
Accuracy: tf.Tensor(0.93807, shape=(), dtype=float32)
Precision: tf.Tensor(0.7298063, shape=(), dtype=float32)
Recall: tf.Tensor(0.9005382, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.80623263, shape=(), dtype=float32)

3. 使用老模型; 调整了loss函数的默认参数
Accuracy: tf.Tensor(0.95021, shape=(), dtype=float32)
Precision: tf.Tensor(0.85285217, shape=(), dtype=float32)
Recall: tf.Tensor(0.787936, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.8191099, shape=(), dtype=float32)

4. 使用老模型; 采用随机平衡补全训练; 调整了loss函数的默认参数
Accuracy: tf.Tensor(0.93503994, shape=(), dtype=float32)
Precision: tf.Tensor(0.71969396, shape=(), dtype=float32)
Recall: tf.Tensor(0.8942476, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.7975315, shape=(), dtype=float32)

5. 使用老模型; 调整了loss函数的默认参数; 用上了动态学习率和早停机制; 采用自定义类均衡学习率调度器
Accuracy: tf.Tensor(0.93460995, shape=(), dtype=float32)
Precision: tf.Tensor(0.80786306, shape=(), dtype=float32)
Recall: tf.Tensor(0.71237856, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.75712216, shape=(), dtype=float32)

6. 使用老模型; 采用随机平衡补全训练; 调整了loss函数的默认参数; 用上了动态学习率和早停机制; 采用自定义类均衡学习率调度器
Accuracy: tf.Tensor(0.91761994, shape=(), dtype=float32)
Precision: tf.Tensor(0.67682534, shape=(), dtype=float32)
Recall: tf.Tensor(0.81184036, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.7382102, shape=(), dtype=float32)

7. 使用老模型; 用上了动态学习率和早停机制
Accuracy: tf.Tensor(0.93178, shape=(), dtype=float32)
Precision: tf.Tensor(0.836344, shape=(), dtype=float32)
Recall: tf.Tensor(0.6504508, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.7317763, shape=(), dtype=float32)

8. 使用老模型; 采用随机平衡补全训练; 用上了动态学习率和早停机制
Accuracy: tf.Tensor(0.91615, shape=(), dtype=float32)
Precision: tf.Tensor(0.6559406, shape=(), dtype=float32)
Recall: tf.Tensor(0.8705529, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.74816036, shape=(), dtype=float32)

9. 使用老模型; 采用自定义类均衡学习率调度器
Accuracy: tf.Tensor(0.9551, shape=(), dtype=float32)
Precision: tf.Tensor(0.87882996, shape=(), dtype=float32)
Recall: tf.Tensor(0.7959041, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.8353139, shape=(), dtype=float32)

10. 使用老模型; 采用随机平衡补全训练; 采用自定义类均衡学习率调度器
Accuracy: tf.Tensor(0.93, shape=(), dtype=float32)
Precision: tf.Tensor(0.697818, shape=(), dtype=float32)
Recall: tf.Tensor(0.90081775, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.7864291, shape=(), dtype=float32)

11. 使用老模型; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器
Accuracy: tf.Tensor(0.9528, shape=(), dtype=float32)
Precision: tf.Tensor(0.8737622, shape=(), dtype=float32)
Recall: tf.Tensor(0.78325295, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.82603574, shape=(), dtype=float32)

12. 使用老模型; 采用随机平衡补全训练; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器
Accuracy: tf.Tensor(0.93086994, shape=(), dtype=float32)
Precision: tf.Tensor(0.7060299, shape=(), dtype=float32)
Recall: tf.Tensor(0.88551056, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.7856501, shape=(), dtype=float32)

13. 使用老模型; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器; 设置学习率上限
Model saved and evaluated from: ./models/smartbert_bilstm_ljw_v13.keras
Accuracy: tf.Tensor(0.9509, shape=(), dtype=float32)
Precision: tf.Tensor(0.8690598, shape=(), dtype=float32)
Recall: tf.Tensor(0.77332777, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.8184037, shape=(), dtype=float32)

14. 使用老模型; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器; 设置学习率上限; 引入平滑因子
Accuracy: tf.Tensor(0.92120993, shape=(), dtype=float32)
Precision: tf.Tensor(0.66074824, shape=(), dtype=float32)
Recall: tf.Tensor(0.92339414, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.77029824, shape=(), dtype=float32)
==========================================================
UNIT = 128
BATCH = 160
BATCH_SIZE = 100
EPOCH = 50
DROP = 0.3
MAX_SEQ = 256
DIM = 768
PAD = 0.0
==========================================================
15. 使用老模型;
Accuracy: tf.Tensor(0.97095, shape=(), dtype=float32)
Precision: tf.Tensor(0.9179006, shape=(), dtype=float32)
Recall: tf.Tensor(0.8752359, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.8960607, shape=(), dtype=float32)

16. 使用老模型; 调整了loss函数的默认参数
Accuracy: tf.Tensor(0.96905, shape=(), dtype=float32)
Precision: tf.Tensor(0.9020367, shape=(), dtype=float32)
Recall: tf.Tensor(0.8791501, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.89044636, shape=(), dtype=float32)

17. 使用老模型; 采用自定义类均衡学习率调度器
Accuracy: tf.Tensor(0.96815, shape=(), dtype=float32)
Precision: tf.Tensor(0.90169024, shape=(), dtype=float32)
Recall: tf.Tensor(0.87250996, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.88686013, shape=(), dtype=float32)

18.  使用老模型; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器
Accuracy: tf.Tensor(0.96497, shape=(), dtype=float32)
Precision: tf.Tensor(0.87425524, shape=(), dtype=float32)
Recall: tf.Tensor(0.8820158, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.87811834, shape=(), dtype=float32)

19. 使用老模型; 调整了loss函数的默认参数; 采用自定义类均衡学习率调度器; 设置学习率上限
Accuracy: tf.Tensor(0.97175, shape=(), dtype=float32)
Precision: tf.Tensor(0.91752726, shape=(), dtype=float32)
Recall: tf.Tensor(0.88180614, shape=(), dtype=float32)
F1 Score: tf.Tensor(0.89931214, shape=(), dtype=float32)

目前最高:       0.91
目前我的最高:    0.89
==========================================================
"""




"""
def train(batch=BATCH, batch_size=BATCH_SIZE, epoch=EPOCH, start=1):
    model = loadModel()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryFocalCrossentropy(),
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