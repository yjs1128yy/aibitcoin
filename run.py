import pyupbit
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib  # 스케일러 불러오기용

# 1. 저장된 모델 불러오기
model = load_model('bitcoin_lstm_model.h5')
print("모델이 로드되었습니다.")

# 스케일러 불러오기
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# 2. 새로운 데이터 수집 (비트코인 데이터 불러오기)
df_new = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=20)  # 최근 20일 데이터

# 3. 기술적 지표 추가 (RSI, 이동평균선)
df_new['MA7'] = talib.SMA(df_new['close'], timeperiod=7)   # 7일 이동평균
df_new['MA14'] = talib.SMA(df_new['close'], timeperiod=14) # 14일 이동평균
df_new['RSI7'] = talib.RSI(df_new['close'], timeperiod=7)  # 7일 RSI
df_new['RSI14'] = talib.RSI(df_new['close'], timeperiod=14)  # 14일 RSI
df_new['EMA7'] = talib.EMA(df_new['close'], timeperiod=7)  # 7일 EMA
df_new['EMA14'] = talib.EMA(df_new['close'], timeperiod=14)  # 14일 EMA
df_new['BBANDS_upper'], df_new['BBANDS_middle'], df_new['BBANDS_lower'] = talib.BBANDS(df_new['close'], timeperiod=20)  # 20일 Bollinger Bands
df_new['stochastic'] = talib.STOCH(df_new["high"], df_new["low"], df_new["close"], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0]
df_new['macd'], df_new['macd_signal'], df_new['macd_hist'] = talib.MACD(df_new["close"], fastperiod=12, slowperiod=26, signalperiod=9)

# 4. 결측치 제거
df_new = df_new.dropna()

# 5. 피처 설정
X_new = df_new[['open', 'high', 'low', 'close', 'MA7', 'MA14', 'RSI7', 'RSI14', 'EMA7', 'EMA14']]

# 6. 새로운 데이터를 학습 때 사용한 스케일러로 정규화
X_new_scaled = scaler_X.transform(X_new)

# 7. 새로운 입력 데이터를 LSTM 모델에 맞게 변환 (3D 형태로 변환)
X_new_lstm = np.reshape(X_new_scaled, (X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

# 8. 모델을 사용해 최신 데이터를 바탕으로 미래 가격 예측 (가장 최근 데이터 기준으로 예측)
y_pred_scaled_new = model.predict(X_new_lstm[-1].reshape(1, 1, X_new_lstm.shape[2]))

# 9. 예측된 값을 역변환 (스케일 복구)
y_pred_new = scaler_y.inverse_transform(y_pred_scaled_new)

# 10. 예측 결과 출력
print("미래 예측 종가 (다음날):", y_pred_new.flatten()[0])
