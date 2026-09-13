import pymysql
import random
import json
import os
from collections import Counter


# 数据库连接信息
HOST = os.environ.get('SMARTINTENT_DB_HOST', '127.0.0.1')
DATABASE = os.environ.get('SMARTINTENT_DB_NAME', 'web3')
USER = os.environ.get('SMARTINTENT_DB_USER', 'web3')
PASSWORD = os.environ.get('SMARTINTENT_DB_PASSWORD', '')
PORT = int(os.environ.get('SMARTINTENT_DB_PORT', '3306'))
CHARSET = os.environ.get('SMARTINTENT_DB_CHARSET', 'utf8')


# 标签类型字典
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


# 不考虑重复出现的ID
def getBatch_v2(start_id, end_id):
    # 建立数据库连接
    db = pymysql.connect(host=HOST, database=DATABASE, user=USER,
                         password=PASSWORD, port=PORT, charset=CHARSET)
    cursor = db.cursor()

    all_ids = []  # 用于存储每个标签随机选出的id
    x_all, y_all = [], []  # 用于存储特征和标签数据

    # 对于每一个标签，获取所有的id，并随机选取10个
    for label in TYPE.keys():
        # 获取当前标签对应的id数组，只筛选在 start_id 和 end_id 范围内的 id
        sql = f"""
        SELECT t.Id 
        FROM tokens AS t 
        WHERE JSON_CONTAINS(t.Scams, '{{"type": "{label}"}}')
        AND t.Id >= {start_id} AND t.Id <= {end_id};
        """
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            ids_for_label = [row[0] for row in result]  # 提取id

            # 检查是否少于10个id
            if len(ids_for_label) < 10:
                print(
                    f"Warning: Less than 10 ids found for label {label} in range {start_id}-{end_id}!")

            # 随机选取10个id
            random_ids = random.sample(
                ids_for_label, min(len(ids_for_label), 10))

            # 将这些id加入总id列表
            all_ids.extend(random_ids)

        except Exception as e:
            print(f"Error fetching ids for label {label}: {e}")
            continue  # 发生错误时跳过当前标签

    # 使用这些id从数据库中获取对应的特征和标签
    if all_ids:
        try:
            # 从数据库中查询特征和标签数据
            sql = f"""
            SELECT t.Id, c.Embedding2, t.Scams
            FROM tokens AS t
            INNER JOIN contracts AS c ON t.ContractId = c.Id
            WHERE t.Id IN ({', '.join(map(str, all_ids))});
            """
            cursor.execute(sql)
            result = cursor.fetchall()

            # 处理查询结果，提取x和y
            for row in result:
                try:
                    # 解析特征数据 (Embedding2)
                    data = json.loads(row[1])
                    x = []
                    for data_i in data.values():
                        x.extend(data_i.values())

                    # 解析标签数据 (Scams)
                    scams = json.loads(row[2])
                    y = [0] * 10  # 初始化标签数组
                    for item in scams:
                        y[TYPE[item['type']]] = 1

                    # 将特征和标签加入列表
                    x_all.append(x)
                    y_all.append(y)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing data for ID {row[0]}: {e}")
                except Exception as e:
                    print(f"Unexpected error for ID {row[0]}: {e}")

        except Exception as e:
            print(f"Database operation failed: {e}")

    # 关闭数据库连接
    cursor.close()
    db.close()

    # 返回特征、标签和随机选出的id集合
    return x_all, y_all, all_ids


# 避免了重复出现的ID
def getBatch_v3(start_id, end_id):
    # 建立数据库连接
    db = pymysql.connect(host=HOST, database=DATABASE, user=USER,
                         password=PASSWORD, port=PORT, charset=CHARSET)
    cursor = db.cursor()

    all_ids = []  # 用于存储每个标签随机选出的id
    selected_ids = set()  # 用于存储已经选择的id，避免重复
    x_all, y_all = [], []  # 用于存储特征和标签数据

    # 对于每一个标签，获取所有的id，并随机选取10个
    for label in TYPE.keys():
        # 获取当前标签对应的id数组，只筛选在 start_id 和 end_id 范围内的 id
        sql = f"""
        SELECT t.Id 
        FROM tokens AS t 
        WHERE JSON_CONTAINS(t.Scams, '{{"type": "{label}"}}')
        AND t.Id >= {start_id} AND t.Id <= {end_id};
        """
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            ids_for_label = [row[0] for row in result]  # 提取id

            # 检查是否少于10个id（这个判断其实可以放去重后面）
            if len(ids_for_label) < 10:
                print(
                    f"Warning: Less than 10 ids found for label {label} in range {start_id}-{end_id}!")

            # 过滤掉已选过的id
            available_ids = [
                id for id in ids_for_label if id not in selected_ids]

            # 检查是否少于10个id
            if len(available_ids) < 10:
                print(
                    f"Warning: Less than 10 available ids found for label {label} in range {start_id}-{end_id} after removing duplicates!")

            # 随机选取新的id
            random_ids = random.sample(
                available_ids, min(len(available_ids), 10))

            # 将新选出的id加入all_ids并更新selected_ids集合
            all_ids.extend(random_ids)
            selected_ids.update(random_ids)

        except Exception as e:
            print(f"Error fetching ids for label {label}: {e}")
            continue  # 发生错误时跳过当前标签

    # 使用这些id从数据库中获取对应的特征和标签
    if all_ids:
        try:
            # 从数据库中查询特征和标签数据
            sql = f"""
            SELECT t.Id, c.Embedding2, t.Scams
            FROM tokens AS t
            INNER JOIN contracts AS c ON t.ContractId = c.Id
            WHERE t.Id IN ({', '.join(map(str, all_ids))});
            """
            cursor.execute(sql)
            result = cursor.fetchall()

            # 处理查询结果，提取x和y
            for row in result:
                try:
                    # 解析特征数据 (Embedding2)
                    data = json.loads(row[1])
                    x = []
                    for data_i in data.values():
                        x.extend(data_i.values())

                    # 解析标签数据 (Scams)
                    scams = json.loads(row[2])
                    y = [0] * 10  # 初始化标签数组
                    for item in scams:
                        y[TYPE[item['type']]] = 1

                    # 将特征和标签加入列表
                    x_all.append(x)
                    y_all.append(y)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing data for ID {row[0]}: {e}")
                except Exception as e:
                    print(f"Unexpected error for ID {row[0]}: {e}")

        except Exception as e:
            print(f"Database operation failed: {e}")

    # 关闭数据库连接
    cursor.close()
    db.close()

    # 返回特征、标签和随机选出的id集合
    return x_all, y_all, all_ids


'''
# 指定start_id和end_id
start_id = 1
end_id = 17197
print("==========================================================")
print(f"Processing data in ID range from {start_id} to {end_id}")

# 获取数据
x_all, y_all, all_ids = getBatch_v2(start_id, end_id)

# 统计每种意图的数量
type_count = {label: 0 for label in TYPE.keys()}
for y in y_all:
    for label, index in TYPE.items():
        if y[index] == 1:
            type_count[label] += 1

# 打印每种意图的样本数量
print("==========================================================")
for label, count in type_count.items():
    print(f"Category '{label}' has {count} samples")
print("==========================================================")
print("Type distribution:", type_count)
print("==========================================================")

# 统计重复出现的ID
id_counter = Counter(all_ids)
repeated_ids = {id: count for id, count in id_counter.items() if count > 1}
if repeated_ids:
    print("Repeated IDs and their counts:")
    for id, count in repeated_ids.items():
        print(f"ID {id} appears {count} times.")
else:
    print("No repeated IDs found.")
print("==========================================================")
'''
