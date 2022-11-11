import random
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
import warnings
import lightgbm as lgb

raw_train = pd.read_csv('./data/train.csv', nrows=420000, skiprows=0)
raw_test = pd.read_csv('./data/test.csv', nrows=5200, skiprows=0)
submit_df = pd.read_csv('./data/submit_example.csv')

print("原始数据：")
# display(raw_train, raw_test, submit_df)
display(raw_train)
display(raw_test)
display(submit_df)

# 预处理
print("预处理数据中……")
for df in [raw_train, raw_test]:
    # 处理空值
    for f in ['category_code', 'brand']:
        df[f].fillna('<unkown>', inplace=True)

    # 处理时间
    df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S UTC')
    df['timestamp'] = df['event_time'].apply(lambda x: time.mktime(x.timetuple()))
    df['timestamp'] = df['timestamp'].astype(int)

# 排序
raw_train = raw_train.sort_values(['user_id', 'timestamp'])
raw_test = raw_test.sort_values(['user_id', 'timestamp'])

# 处理非数值特征
print("处理非数值特征中……")
df = pd.concat([raw_train, raw_test], ignore_index=True)
for f in ['event_type', 'category_code', 'brand']:
    # 构建编码器
    le = LabelEncoder()  # LabelEncoder进行编码
    le.fit(df[f])

    # 设置新值
    raw_train[f] = le.transform(raw_train[f])
    raw_test[f] = le.transform(raw_test[f])
# print("df:")
# display(df[df['event_type'] == 'purchase']['event_type'])
# print("raw_train:")
# display(raw_train[['event_type', 'category_code', 'brand']])
# print("raw_test:")
# display(raw_test[['event_type', 'category_code', 'brand']])
# print("pre")
# display(raw_train)
# display(df)

# 清洗数据
# 删除无用特征
print("删除无用特征中……")
useless = ['event_time', 'user_session', 'timestamp']
for df in [raw_train, raw_test]:
    df.drop(columns=useless, inplace=True)

# 删除浏览量很大，但是购买很少的用户数据
print("删除低价值用户数据中……")
buy_data = raw_train[['user_id', 'event_type']]
for uid in buy_data['user_id']:
    uid_data = buy_data[buy_data['user_id'] == uid]
    buy = uid_data[uid_data['event_type'] == 1].shape[0]
    buy_rate = buy / uid_data.shape[0]  # 用户的购买率
    # print(buy_rate)
    # 删除操作记录大于200条但购买率小于0.005的用户的记录
    if buy_rate < 0.005 and uid_data.shape[0] > 200:
        raw_train.drop(raw_train[raw_train['user_id'] == uid].index, inplace=True)

# 构造训练集和测试集
print("数据处理完成，开始构造训练集和测试集……")
# 滑动窗口构造数据集
# 为了让机器学习模型能够处理时序数据，必须通过滑动窗口构造数据，滑动窗口法，便可以从时间序列，创建出输入数据x、输出数据y，后一个时间点的动作为前一个时间点的预测值
# 训练集数据生成：滑动窗口
# 用前一个时间节点的数据预测后一个时间节点是商品
print("构造训练集中……")
train_df = pd.DataFrame()
user_ids = raw_train['user_id'].unique()  # 训练集中用户编号 53918
# print("raw_train.shape[0]")  # 数据第一列的行数
# print(raw_train.shape)    # 数据尺寸
for uid in tqdm(user_ids):
    # 以该用户的记录作为预测依据
    user_data = raw_train[raw_train['user_id'] == uid].copy(deep=True)  # 深复制数据操作在副本进行
    if user_data.shape[0] < 2:  # 数据行尺寸--记录行数
        # 小于两条的，直接忽略
        continue

    # 用用户序列的最后一个product_id作为label
    # user_data['y'] = np.nan
    # a = user_data['product_id'].tail(1).values[-1]
    # user_data.fillna(value=a, inplace=True)

    # 用前一个时间节点的数据(x输入)预测后一个时间节点的商品(y输出)
    # 滑动窗口，步长为1--用后一个时间节点的product_id作为该时间节点的label输出结果
    user_data['y'] = user_data['product_id'].shift(-1)
    user_data = user_data.head(user_data.shape[0] - 1)  # 舍弃最后一行，最后一行输出y为空值
    train_df = train_df.append(user_data)

train_df['y'] = train_df['y'].astype(int)
train_df = train_df.reset_index(drop=True)  # 重置索引--drop=true不将原来的不连续索引并入结果数据
train_df.drop(columns=['user_id'], inplace=True)  # 删除训练集中的user_id列，并覆盖原数据
# print("train_df_reset_index")
# print(train_df)

# 测试集数据生成，随机取每个用户的一次操作用来做预测
print("构造测试集中……")
test_df = raw_test.groupby(['user_id'], as_index=False).apply(lambda x: x.sample(1))
# print("display")
# print(train_df.dtypes)
# print(test_df.dtypes)

# 模型训练并预测
print("训练预测中……")
user_ids = test_df['user_id'].unique()  # 测试集中的用户编号 558
preds = []  # 存放预测结果
for uid in tqdm(user_ids):
    # 原始测试数据里对应在测试集中用户所操作的所有的product_id
    pids = raw_test[raw_test['user_id'] == uid]['product_id'].unique()

    # 找到训练集中有相同product_id的数据作为当前用户的训练集
    p_train = train_df[train_df['product_id'].isin(pids)]

    # 只取最后一条进行预测
    user_test = test_df[test_df['user_id'] == uid].drop(columns=['user_id'])  # 测试集该用户操作记录
    # X_train + y_train = p_train
    X_train = p_train.iloc[:, :-1]  # 训练输入——X
    y_train = p_train['y']  # 训练预测输出——y

    if len(X_train) > 0:  # 对应训练集有数据
        # 训练
        clf = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=30)
        clf.fit(X_train, y_train)

        # 预测
        pred = clf.predict(user_test)[0]
    else:
        # 训练集中无对应数据
        # 随机取一条数据作为预测值
        a = random.randint(0, user_test.shape[0] - 1)
        pred = user_test['product_id'].iloc[a]

    preds.append(pred)

# 模型得分
warnings.filterwarnings("ignore")
print('LightGBM Model accuracy score :{0:0.4f}'.format(accuracy_score(test_df['product_id'], preds)))
print("precision score: ", precision_score(test_df['product_id'], preds, average='macro'))
print("Recall score: ", recall_score(test_df['product_id'], preds, average='macro'))
print("f1 score: ", f1_score(test_df['product_id'], preds, average='macro'))

submit_df['product_id'] = preds

# 文件保存预测结果
print("预测结果保存中……")
submit_df.to_csv('baseline.csv', index=False)
print("预测结果成功保存……")
