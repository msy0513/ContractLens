import pymysql
import json
import os
import tensorflow as tf
from tensorflow.keras import backend as K


HOST = os.environ.get('SMARTINTENT_DB_HOST', '127.0.0.1')
DATABASE = os.environ.get('SMARTINTENT_DB_NAME', 'web3')
USER = os.environ.get('SMARTINTENT_DB_USER', 'web3')
PASSWORD = os.environ.get('SMARTINTENT_DB_PASSWORD', '')
PORT = int(os.environ.get('SMARTINTENT_DB_PORT', '3306'))
CHARSET = os.environ.get('SMARTINTENT_DB_CHARSET', 'utf8')
TYPE = {
    'fee': 0,
    'disableTrading': 1,
    'blacklist': 2,
    'reflect': 3,
    'maxTX': 4,
    'mint': 5,
    'honeypot': 6,
    'reward': 7,
    'rebase': 8,
    'maxSell': 9
}

WEIGHT = [1, 5, 5, 1, 1, 10, 20, 10, 20, 20]


def weighted_binary_crossentropy(weights):
    def w_binary_crossentropy(y_true, y_pred):
        weights_v = tf.constant(weights, dtype=tf.float32)
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = bce * weights_v
        return K.mean(weighted_bce, axis=-1)
    return w_binary_crossentropy


# 使用权重数组创建损失函数
loss_fn = weighted_binary_crossentropy(WEIGHT)

db = pymysql.connect(host=HOST, database=DATABASE, user=USER,
                     password=PASSWORD, port=PORT, charset=CHARSET)

cursor = db.cursor()


# Get data one by one
def getOne(id):
    sql = """
    SELECT t.Id, t.ContractId, c.Id, c.ContractAddress, c.Embedding2, t.Scams
    FROM tokens AS t
    INNER JOIN contracts AS c on t.ContractId=c.Id
    WHERE t.Id=%d;
    """ % id

    try:
        cursor.execute(sql)
        res = cursor.fetchall()

        # average embedding
        data = json.loads(res[0][4])
        x = []
        for i in data:
            for j in data[i]:
                x.append(data[i][j])
        # y
        scams = json.loads(res[0][5])
        y = [0] * 10
        for item in scams:
            y[TYPE[item['type']]] = 1
        return {'x': x, 'y': y}
    except Exception as e:
        return None


# Get batch of data
def getBatch(start_id, n):
    sql = """
    SELECT t.Id, t.ContractId, c.Id, c.ContractAddress, c.Embedding2, t.Scams
    FROM tokens AS t
    INNER JOIN contracts AS c ON t.ContractId = c.Id
    WHERE t.Id>=%d
    LIMIT %d;
    """ % (start_id, n)

    try:
        cursor.execute(sql)
        res = cursor.fetchall()

        # Initialize x and y arrays
        x_all, y_all = [], []
        max_id = start_id  # Initialize max_id with start_id

        for row in res:
            try:
                # Update max_id to the current row's t.Id if it's greater
                if row[0] > max_id:
                    max_id = row[0]

                # Check if data is null
                if row[4] is None:
                    print(f"ID: {row[0]} - Embedding2 data is null")
                    continue

                # Parse embedding data
                data = json.loads(row[4])

                x = []
                for data_i in data.values():
                    x.extend(data_i.values())

                # Parse scams data
                scams = json.loads(row[5])
                y = [0] * 10
                for item in scams:
                    y[TYPE[item['type']]] = 1

                x_all.append(x)
                y_all.append(y)

            except (json.JSONDecodeError, KeyError) as e:
                # Handle JSON parsing or key related errors
                print(f"ID: {row[0]} - Error parsing JSON or missing key: {e}")
            except Exception as e:
                # Handle any other exceptions
                print(f"ID: {row[0]} - An unexpected error occurred: {e}")

        return x_all, y_all, max_id

    except Exception as e:
        # Top-level exception handling for cursor operations
        print(f"Database operation failed: {e}")
        return [], [], start_id


'''
res = getXY2(1, 1)
print(res)
'''
