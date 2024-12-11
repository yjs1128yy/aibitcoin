import numpy as np
import pandas as pd
import pyupbit
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_normal, RandomNormal, HeUniform
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 수집
data_raw= pyupbit.get_ohlcv("KRW-BTC", interval="day", count=400)
data_raw['adj_close_change'] = data_raw['close'].pct_change().fillna(0)
data_raw['volume_change'] = data_raw['volume'].pct_change().fillna(0)

def add_technical_indicators(df):
    df['MA20'] = df['close'].rolling(window=20).mean()  # 20일 이동평균
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()  # 20일 EMA
    df['stddev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['MA20'] + (df['stddev'] * 2)  # Bollinger Bands 상한선
    df['lower_band'] = df['MA20'] - (df['stddev'] * 2)  # Bollinger Bands 하한선
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))  # RSI 계산

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()  # MACD
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # MACD Signal Line
    
    df['%K'] = ((df['close'] - df['low'].rolling(14).min()) / 
                (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100  # Stochastic %K
    df['%D'] = df['%K'].rolling(3).mean()  # Stochastic %D
    
    df['adj_close_change'] = df['close'].pct_change().fillna(0)
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    return df.dropna()

data_raw = add_technical_indicators(data_raw)
# 2. 학습에 사용할 데이터 준비
data = data_raw[['adj_close_change', 'volume_change', 'MA20', 'EMA20', 'upper_band', 'lower_band', 'RSI', 'MACD', 'Signal', '%K', '%D']]
scaler = StandardScaler()

# 3. 데이터 정규화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 4. 시퀀스 데이터 준비
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])
        y.append(data[i + time_steps, 0])  # 다음 날의 수정 종가 변화량
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)

# 학습/검증 데이터 분할
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# 5. LSTM 모델 구축
model = Sequential()
model.add(LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_initializer=glorot_normal(), recurrent_initializer=RandomNormal(),
               bias_initializer=HeUniform(), kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01),
               return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_initializer=glorot_normal(), recurrent_initializer=RandomNormal(),
               bias_initializer=HeUniform(), kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

# 6. 모델 컴파일 및 학습
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')

early_stopping = EarlyStopping(
    monitor='val_loss',       # 검증 손실을 기준으로 조기 종료
    patience=5,                # 성능이 개선되지 않으면 5 epoch 후에 종료
    restore_best_weights=True  # 조기 종료 시, 가장 좋은 가중치를 복원
)

history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_val, y_val), callbacks = [early_stopping])

# 7. 예측 및 스케일 복구
predicted_changes = model.predict(X_test)
predicted_changes = scaler.inverse_transform(np.concatenate((predicted_changes, np.zeros((len(predicted_changes), data.shape[1] - 1))), axis=1))[:, 0]

# 실제 및 예측 종가 계산
actual_prices = data_raw['close'].iloc[-len(predicted_changes)-1:].values
predicted_prices = [actual_prices[0]]  # 초기값 설정

for i, change in enumerate(predicted_changes):
    predicted_price = actual_prices[i] * (1 + change)
    predicted_prices.append(predicted_price)


predicted_prices = np.array(predicted_prices[1:])

print(predicted_changes[:5])

# 8. 결과 시각화
plt.figure(figsize=(14, 7))
plt.plot(range(len(actual_prices[-len(predicted_prices):])), actual_prices[-len(predicted_prices):], label='Actual Adjusted Close Price')
plt.plot(range(len(predicted_prices)), predicted_prices, label='Predicted Adjusted Close Price', linestyle='--')
plt.title('Bitcoin Adjusted Close Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
