import pyupbit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import joblib  # 스케일러 저장을 위한 라이브러리

# 1. 데이터 수집 (비트코인 데이터 불러오기)
df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=200)

# 2. 기술적 지표 추가 (RSI, 이동평균선)
df['MA7'] = talib.SMA(df['close'], timeperiod=7)   # 7일 이동평균
df['MA14'] = talib.SMA(df['close'], timeperiod=14) # 14일 이동평균
df['RSI7'] = talib.RSI(df['close'], timeperiod=7)  # 7일 RSI
df['RSI14'] = talib.RSI(df['close'], timeperiod=14)  # 14일 RSI
df['EMA7'] = talib.EMA(df['close'], timeperiod=7)  # 7일 EMA
df['EMA14'] = talib.EMA(df['close'], timeperiod=14)  # 14일 EMA
df['BBANDS_upper'], df['BBANDS_middle'], df['BBANDS_lower'] = talib.BBANDS(df['close'], timeperiod=20)  # 20일 Bollinger Bands
df['stochastic'] = talib.STOCH(df["high"], df["low"], df["close"], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0]
df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)

# 3. 결측치 제거
df = df.dropna()

# 4. 피처와 타겟 설정 (shift(-1)로 미래의 가격 예측)
X = df[['open', 'high', 'low', 'close', 'MA7', 'MA14', 'RSI7', 'RSI14', 'EMA7', 'EMA14', 'BBANDS_upper', 'BBANDS_middle', 'BBANDS_lower', 'macd', 'macd_signal', 'macd_hist']]  # 피처
y = df['close'].shift(-1)  # 타겟 (다음 날 종가)

# 마지막 행 제거 (shift로 인해 타겟 값이 없는 마지막 데이터 제거)
X = X[:-1]
y = y.dropna()

# 5. 데이터 정규화 (MinMaxScaler 사용)
scaler_X = MinMaxScaler(feature_range=(0, 1))  # 피처용 스케일러
scaler_y = MinMaxScaler(feature_range=(0, 1))  # 타겟용 스케일러

# 피처와 타겟 값 스케일링
X_scaled = scaler_X.fit_transform(X)  # 입력 피처 스케일링
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))  # 타겟 스케일링

# 스케일러를 파일로 저장
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# 6. 학습 데이터와 테스트 데이터로 분리 (8:2 비율)
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# 7. LSTM 모델을 위한 데이터 전처리 (3D 형태로 변환)
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 8. LSTM 모델 구축
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(1, X_train.shape[1])))  # 첫 번째 LSTM 레이어
model.add(LSTM(units=64, return_sequences=True))  # 두 번째 LSTM 레이어
model.add(LSTM(units=64, return_sequences=False))  # 세 번째 LSTM 레이어
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))        # 출력 레이어 (다음 날 종가 예측)

# 9. 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 10. 모델 학습
early_stopping = EarlyStopping(
    monitor='val_loss',       # 검증 손실을 기준으로 조기 종료
    patience=5,                # 성능이 개선되지 않으면 5 epoch 후에 종료
    restore_best_weights=True  # 조기 종료 시, 가장 좋은 가중치를 복원
)
model.fit(X_train_lstm, y_train, epochs=150, batch_size=100, verbose=1, callbacks = [early_stopping])
print("학습 중단")

# 11. 모델 저장
model.save('bitcoin_lstm_model.h5')
print("모델과 스케일러가 저장되었습니다.")

# 12. 모델 예측 (테스트 데이터에 대해 예측 수행)
y_pred_scaled = model.predict(X_test_lstm)

# 13. 예측된 값과 실제 값을 역변환 (스케일 복구)
y_pred = joblib.load('scaler_y.pkl').inverse_transform(y_pred_scaled)
y_test_actual = joblib.load('scaler_y.pkl').inverse_transform(y_test)

# 14. 예측 결과와 실제 결과 비교
print("실제 종가:", y_test_actual[:5])
print("예측 종가 (LSTM):", y_pred[:5].flatten())

# 15. 결과 시각화 
plt.figure(figsize=(14,7))

# 실제 값 출력
plt.plot(y_test_actual, label="Actual Price", color='blue')

plt.plot(y_pred.flatten(), label="Predicted Price", color='red')

plt.title("Bitcoin Price Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Price (KRW)")
plt.legend()
plt.show()
