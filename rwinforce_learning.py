import pyupbit
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
from collections import deque
import random

# 파라미터 설정
EPISODES = 1000
STATE_SIZE = 8  # 종가, 거래량, SMA, EMA, Bollinger Bands, RSI, MACD, Stochastic
ACTION_SIZE = 3  # buy, hold, sell
BATCH_SIZE = 32

# 기술 지표 계산 함수 예제
def calculate_indicators(df):
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return df  # 데이터가 비어있으면 그대로 반환
    
    # 기술 지표 계산
    df['SMA'] = df['close'].rolling(window=10).mean()
    df['EMA'] = df['close'].ewm(span=10, adjust=False).mean()
    
    # RSI 계산 (close price changes로 RSI 계산)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    # MACD 계산
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    # 볼린저 밴드 계산
    rolling_mean = df['close'].rolling(window=10).mean()
    rolling_std = df['close'].rolling(window=10).std()
    df['Bollinger_upper'] = rolling_mean + (rolling_std * 2)
    df['Bollinger_lower'] = rolling_mean - (rolling_std * 2)

    # NaN 값을 0으로 채우기
    df = df.fillna(0)
    
    return df


# DQN 모델 생성 함수
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_dim=STATE_SIZE))
    model.add(Dropout(rate=hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(Dense(ACTION_SIZE, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])), loss='mse')
    return model

# DQN 에이전트 정의
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = None

    def build_model(self):
        tuner = RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=3,
            directory='my_dir',
            project_name='crypto_rl'
        )
        # 임의의 훈련 데이터로 튜닝
        tuner.search(x=np.random.random((BATCH_SIZE, STATE_SIZE)), y=np.random.random((BATCH_SIZE, ACTION_SIZE)), epochs=5, validation_split=0.2)
        self.model = tuner.get_best_models(num_models=1)[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 에이전트 초기화 및 모델 구축
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
agent.build_model()

# 데이터 수집 및 시뮬레이션
tickers = pyupbit.get_tickers(fiat="KRW")

for episode in range(EPISODES):
    for ticker in tickers:
        # 400일의 데이터를 가져옵니다.
        df = pyupbit.get_ohlcv(ticker, interval="day", count=400)
        
        # 데이터가 None이거나 비어 있는지 확인
        if df is None or df.empty:
            print(f"No data available for {ticker}")
            continue
        
        # 기술 지표를 계산하여 데이터프레임에 추가
        df = calculate_indicators(df)
        
        # 데이터가 충분한지 확인 (최소 10일 필요)
        if len(df) < 10:
            print(f"Not enough data for {ticker}")
            continue
        
        # 초기 상태 설정 - 최근 10일간의 데이터로 상태 초기화
        state = df.iloc[-10:].values.reshape(1, -1)
        total_reward = 0

        for t in range(390):  # 400일 중 마지막 10일을 제외하고 학습
            action = agent.act(state)
            
            # next_state 슬라이싱 시 충분한 데이터 확인
            if len(df) <= (10 + t):
                print(f"Insufficient data for next_state at step {t} for {ticker}")
                break

            # 다음 상태
            next_state = df.iloc[t:t+10].values.reshape(1, -1)
            
            # next_state가 비어 있지 않은지 확인
            if next_state.size == 0:
                print(f"Empty next_state encountered for {ticker} at step {t}")
                break
            
            # 보상 계산 - 현재 상태와 다음 상태 간의 종가 변화율을 기준으로 계산
            current_price = state[0][-1]  # 현재 상태의 마지막 종가
            next_price = next_state[0][-1]  # 다음 상태의 마지막 종가
            
            if current_price != 0:  # 0으로 나누는 오류 방지
                reward = (next_price - current_price) / current_price * 100
            else:
                reward = 0

            # 메모리에 저장
            done = (t == 389)  # 마지막 상태일 때 종료
            agent.remember(state, action, reward, next_state, done)
            
            # 상태 업데이트
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode+1}/{EPISODES}, {ticker}: Total Reward: {total_reward:.2f}")
                break

        # 메모리가 충분하면 배치 학습
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)




# 최종 결과 시각화
def plot_results(predictions):
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_predictions[:10]
    coins, percentages = zip(*top_10)
    plt.bar(coins, percentages)
    plt.xlabel('Coins')
    plt.ylabel('Predicted Increase (%)')
    plt.title('Top 10 Coins Expected to Rise')
    plt.show()

    print("Top 10 coins expected to increase:")
    for coin, percentage in top_10:
        print(f"{coin}: {percentage:.2f}%")

# 임의의 예측 결과 시각화
predictions = {ticker: np.random.random() * 10 for ticker in tickers}  # 예측된 상승률 데이터
plot_results(predictions)
