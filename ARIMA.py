import time

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ------------------------------------------------
# # 用于ARIMA模型确定参数（省去该步骤）
# def draw_acf_pacf(data, lags):
#     f = plt.figure(facecolor='white')
#     ax1 = f.add_subplot(211)
#     plot_acf(data, ax=ax1, lags=lags)
#     ax2 = f.add_subplot(212)
#     plot_pacf(data, ax=ax2, lags=lags)
#     plt.subplots_adjust(hspace=0.5)
#     plt.show()
#
#
# data = pd.read_csv('./xfeat_speed.csv')
# data = data["Speed(m/s)"]
# # draw_acf_pacf(data, lags=50)
# q = 30
# p = 22
# d = 1
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字符集显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 设置负号正确显示
#
#
# # 显示数据趋势图
# def draw_data(data):
#     plt.figure()
#     plt.plot(data)


# -----------------------------------------------

def Kar(measurement, prediction, alpha=0.5):
    """
    卡尔曼滤波器
    measurement: 真实测量值
    prediction: 预测值（采用ARIMA预测值）
    return: 卡尔曼滤波器预测结果
    """
    result = (1 - alpha) * measurement + alpha * prediction
    return result


# 读取历史数据
data = pd.read_csv('./xfeat_speed.csv')
data = data["Speed(m/s)"]
# ARIMA预测
start = time.time()
sub_data = data[:400]
# draw_data(sub_data)
model = ARIMA(sub_data, order=(1, 1, 1)).fit()  # 构建ARIMA模型，order参数表示p、d、q
predict_data = model.predict(0, 430)  # 预测数据
forecast = model.forecast(30)  # 预测未来数据
end = time.time()

# 进行卡尔曼滤波
result_data = []
for i in range(1, 400):
    result_data.append(Kar(sub_data[i - 1], predict_data[i], alpha=0.8))
print(end - start)

# 绘制原数据和预测数据对比图
plt.plot(sub_data, label='原数据')
plt.plot(predict_data, label='预测数据')
plt.plot(result_data, label="卡尔曼滤波数据")
plt.legend()
plt.show()
