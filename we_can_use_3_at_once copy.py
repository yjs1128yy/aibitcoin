import pyupbit
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import xgboost as xgb
import lightgbm as lgb

# Define assets to analyze
assets = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP']  # 추가 가능

# Define function to get crypto data and add features
def get_crypto_data(ticker, interval='day', count=400):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    df['close_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['rsi_14'] = calculate_rsi(df['close'], window=14)
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1: 구매, 0: 판매
    df.dropna(inplace=True)
    return df

# RSI 계산
def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 데이터 전처리
def preprocess_set(df, features):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['target'].values
    return train_test_split(X, y, test_size=0.3, random_state=42)

# LSTM 모델 구성
def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        layers.LSTM(64, kernel_initializer='glorot_normal', recurrent_initializer='random_normal',
                    bias_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l2(0.01), activation='tanh', 
                    recurrent_activation='hard_sigmoid', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Reshape data for LSTM
def reshape_data(X, time_step=10):
    X_reshaped = []
    for i in range(time_step, len(X)):
        X_reshaped.append(X[i - time_step:i])
    return np.array(X_reshaped)

# XGBoost, LightGBM 훈련 및 테스트
def train_xgb_lgb(X_train, X_test, y_train, y_test, model_type='xgb'):
    if model_type == 'xgb':
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    elif model_type == 'lgb':
        model = lgb.LGBMClassifier(objective='binary', min_data_in_leaf=5, min_split_gain=0.01, class_weight='balanced')
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds) * 100



# 각 코인에 대한 훈련 및 평가를 위한 주요 기능 정의
def evaluate_models(assets, features, time_step=10, epochs=100, batch_size=32):
    results = []
    
    for asset in assets:
        df = get_crypto_data(asset)
        
        # 데이터 분할
        split1, split2 = int(len(df) * 0.3), int(len(df) * 0.6)
        df1, df2, df3 = df[:split1], df[split1:split2], df[split2:]
        
        accuracies = {'Asset': asset}
        
        for i, df_split in enumerate([df1, df2, df3], start=1):
            X_train, X_test, y_train, y_test = preprocess_set(df_split, features)
            X_train_lstm, X_test_lstm = reshape_data(X_train, time_step), reshape_data(X_test, time_step)
            y_train_lstm, y_test_lstm = y_train[time_step:], y_test[time_step:]
            
            # LSTM 모델 학습
            lstm_model = create_lstm_model(X_train_lstm.shape[1:])
            lstm_model.fit(X_train_lstm, y_train_lstm, validation_split=0.1, epochs=epochs, 
                           batch_size=batch_size, verbose=0)
            lstm_accuracy = lstm_model.evaluate(X_test_lstm, y_test_lstm, verbose=0)[1] * 100
            accuracies[f'LSTM_Set_{i}'] = lstm_accuracy
            
            # XGBoost, LightGBM 훈련 및 테스트
            accuracies[f'XGBoost_Set_{i}'] = train_xgb_lgb(X_train, X_test, y_train, y_test, 'xgb')
            accuracies[f'LightGBM_Set_{i}'] = train_xgb_lgb(X_train, X_test, y_train, y_test, 'lgb')
        
        results.append(accuracies)
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    results_df['LSTM_Average'] = results_df[[f'LSTM_Set_{i}' for i in range(1, 4)]].mean(axis=1)
    results_df['XGBoost_Average'] = results_df[[f'XGBoost_Set_{i}' for i in range(1, 4)]].mean(axis=1)
    results_df['LightGBM_Average'] = results_df[[f'LightGBM_Set_{i}' for i in range(1, 4)]].mean(axis=1)
    
    # 최종 결과 표 표시
    print(results_df[['Asset', 'LSTM_Average', 'XGBoost_Average', 'LightGBM_Average']])
    return results_df

# 기능 정의 및 평가 실행
features = ['close_change', 'volume_change', 'ma_20', 'rsi_14']
results_df = evaluate_models(assets, features)
