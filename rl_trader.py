import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
from collections import deque
import random
import json

class ActorCritic(nn.Module):
    """
    Actor-Critic 네트워크
    - Actor: 행동 정책 학습 (매수/매도/홀딩)
    - Critic: 상태 가치 평가
    """
    def __init__(self, state_dim=5, action_dim=3):
        super(ActorCritic, self).__init__()
        
        # 공통 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor 네트워크 (정책)
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 네트워크 (가치)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class Memory:
    """경험 리플레이 메모리"""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, action_prob):
        self.memory.append((state, action, reward, next_state, done, action_prob))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, action_probs = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            torch.FloatTensor(action_probs)
        )
    
    def clear(self):
        self.memory.clear()
    
    def __len__(self):
        return len(self.memory)

class PPOTrader:
    """
    PPO(Proximal Policy Optimization) 알고리즘 기반 트레이딩 에이전트
    - 멀티모달 특징(캔들스틱, 감성, 가격)을 입력으로 받아 거래 결정
    - 포트폴리오 가치 최대화를 목표로 학습
    """
    def __init__(self, state_dim=5, action_dim=3, gamma=0.99, clip_ratio=0.2, 
                 lr=0.0003, batch_size=64, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim  # 상태 차원 (캔들스틱, 감성, 가격 등)
        self.action_dim = action_dim  # 행동 차원 (매수, 매도, 홀딩)
        self.gamma = gamma  # 할인 계수
        self.clip_ratio = clip_ratio  # PPO 클리핑 비율
        self.lr = lr  # 학습률
        self.batch_size = batch_size  # 배치 크기
        self.epochs = epochs  # 에포크 수
        self.device = device
        
        # 액터-크리틱 네트워크
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 경험 리플레이 메모리
        self.memory = Memory()
        
        # 로깅 설정
        self.logger = logging.getLogger('PPOTrader')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        # 학습 기록
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'total_loss': [],
            'rewards': []
        }
    
    def get_action(self, state, training=True):
        """상태에서 행동을 선택하고 행동 확률 반환"""
        self.policy.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_probs, _ = self.policy(state_tensor)
            action_probs = action_probs.cpu().numpy()
        
        # 학습 모드: 확률에 따라 행동 샘플링
        if training:
            action = np.random.choice(self.action_dim, p=action_probs)
        # 평가 모드: 가장 높은 확률의 행동 선택
        else:
            action = np.argmax(action_probs)
        
        return action, action_probs[action]
    
    def update(self, state, action, reward, next_state, done=False):
        """경험 데이터를 메모리에 저장하고 학습 수행"""
        # 행동 확률 임시값 (실제로는 get_action 호출 시 반환된 값을 사용해야 함)
        action_prob = 0.33  # 임시 값
        
        # 경험 저장
        self.memory.push(state, action, reward, next_state, done, action_prob)
        
        # 충분한 데이터가 쌓였으면 학습 수행
        if len(self.memory) >= self.batch_size:
            self.train()
            
        # 보상 기록
        self.training_history['rewards'].append(reward)
        
        return
    
    def train(self):
        """경험 데이터로 PPO 알고리즘 학습"""
        if len(self.memory) < self.batch_size:
            return
        
        self.policy.train()
        
        for _ in range(self.epochs):
            # 미니배치 샘플링
            states, actions, rewards, next_states, dones, old_action_probs = self.memory.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            old_action_probs = old_action_probs.to(self.device)
            
            # 현재 정책으로 행동 확률 및 상태 가치 계산
            action_probs, state_values = self.policy(states)
            action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 다음 상태 가치 계산
            _, next_state_values = self.policy(next_states)
            next_state_values = next_state_values.squeeze(1)
            
            # 어드밴티지(Advantage) 계산
            delta = rewards + self.gamma * next_state_values * (1 - dones) - state_values.squeeze(1)
            
            # 확률 비율(probability ratio) 계산
            ratio = action_probs / (old_action_probs + 1e-8)
            
            # PPO 클리핑 목표 함수
            advantages = delta.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 크리틱 손실 (MSE)
            critic_loss = F.mse_loss(state_values.squeeze(1), rewards + self.gamma * next_state_values * (1 - dones))
            
            # 전체 손실 (액터 + 크리틱)
            loss = actor_loss + 0.5 * critic_loss
            
            # 최적화
            self.optimizer.zero_grad()
            loss.backward()
            # 기울기 클리핑으로 안정성 확보
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            # 손실 기록
            self.training_history['actor_loss'].append(actor_loss.item())
            self.training_history['critic_loss'].append(critic_loss.item())
            self.training_history['total_loss'].append(loss.item())
        
        # 메모리 비우기
        self.memory.clear()
    
    def save_model(self, path='rl_models'):
        """모델 저장"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, 'ppo_trader.pth'))
        
        # 학습 기록 저장
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f)
        
        self.logger.info(f"모델이 {path}에 저장되었습니다.")
    
    def load_model(self, path='rl_models'):
        """모델 로드"""
        model_path = os.path.join(path, 'ppo_trader.pth')
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"모델이 {model_path}에서 로드되었습니다.")
            
            # 학습 기록 로드
            history_path = os.path.join(path, 'training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
        else:
            self.logger.warning(f"모델 파일 {model_path}이 존재하지 않습니다.")

def convert_to_state(pattern_signal, sentiment_signal, price_change, position, portfolio_value):
    """거래 관련 데이터를 RL 상태 벡터로 변환"""
    try:
        # 입력값 안전성 확보
        # 신호 정규화 및 안전 처리
        if isinstance(pattern_signal, dict):
            pattern_signal = pattern_signal.get('score', 0)
        else:
            try:
                pattern_signal = float(pattern_signal) if pattern_signal is not None else 0
            except (TypeError, ValueError):
                pattern_signal = 0
                
        if isinstance(sentiment_signal, dict):
            sentiment_signal = sentiment_signal.get('score', 0)
        else:
            try:
                sentiment_signal = float(sentiment_signal) if sentiment_signal is not None else 0
            except (TypeError, ValueError):
                sentiment_signal = 0
                
        # 가격 변화율 정규화
        try:
            price_change = float(price_change) if price_change is not None else 0
            # 이상치 클리핑
            price_change = max(min(price_change, 1.0), -1.0)
        except (TypeError, ValueError):
            price_change = 0
        
        # 포트폴리오 가치 정규화
        try:
            portfolio_value = float(portfolio_value) if portfolio_value is not None else 0
            # 음수 값 방지
            portfolio_value = max(portfolio_value, 0)
        except (TypeError, ValueError):
            portfolio_value = 0
        
        # 포지션을 원-핫 인코딩 (롱:1,0,0 / 숏:0,1,0 / 중립:0,0,1)
        position_onehot = [0, 0, 0]
        
        # 포지션 문자열 처리
        if isinstance(position, str):
            position = position.lower()
            if position in ['long', 'buy']:
                position_onehot[0] = 1
            elif position in ['short', 'sell']:
                position_onehot[1] = 1
            else:  # neutral, cash 등
                position_onehot[2] = 1
        else:
            # 기본값: 중립
            position_onehot[2] = 1
        
        # 상태 벡터 구성: [패턴 신호, 감성 신호, 가격 변화율, 포지션(원핫), 포트폴리오 가치]
        state = [
            pattern_signal,
            sentiment_signal,
            price_change,
            position_onehot[0],
            position_onehot[1],
            position_onehot[2],
            portfolio_value / 10000.0  # 정규화
        ]
        
        return state
    except Exception as e:
        # 오류 발생 시 안전한 기본값 반환
        logger = logging.getLogger('RL_Utils')
        logger.warning(f"상태 변환 중 오류 발생: {str(e)}, 기본 상태 반환")
        return [0, 0, 0, 0, 0, 1, 0]  # 기본 상태

def action_to_position(action):
    """액션 인덱스를 포지션으로 변환"""
    try:
        # 정수로 변환 시도
        action = int(action) if action is not None else 2
        
        if action == 0:
            return 'long'  # 매수
        elif action == 1:
            return 'short'  # 매도
        else:  # action == 2 또는 다른 값
            return 'cash'  # 현금 보유
    except (TypeError, ValueError):
        # 정수가 아닌 값이 전달된 경우
        return 'cash'  # 기본값: 현금 보유

# 모델 사용 예시
def test_rl_trader():
    # 상태 차원: 패턴 신호, 감성 신호, 가격 변화, 포지션(3), 포트폴리오 가치
    state_dim = 7
    # 행동 차원: 매수, 매도, 홀딩
    action_dim = 3
    
    # RL 트레이더 초기화
    trader = PPOTrader(state_dim=state_dim, action_dim=action_dim)
    
    # 예시 상태
    pattern_signal = 0.7  # 캔들스틱 패턴 신호
    sentiment_signal = 0.3  # 뉴스 감성 신호
    price_change = 0.01  # 가격 변화율
    position = 'neutral'  # 현재 포지션
    portfolio_value = 10000  # 포트폴리오 가치
    
    state = convert_to_state(
        pattern_signal, sentiment_signal, price_change, position, portfolio_value
    )
    
    # 행동 예측
    action, action_prob = trader.get_action(state, training=False)
    new_position = action_to_position(action)
    
    print(f"현재 상태: 패턴={pattern_signal:.2f}, 감성={sentiment_signal:.2f}, 가격변화={price_change:.2f}%, 포지션={position}")
    print(f"예측 행동: {new_position} (확률: {action_prob:.4f})")

if __name__ == "__main__":
    test_rl_trader() 