import torch
import numpy as np
import pandas as pd
import os
import logging
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class MultiTimeframeEnsemble:
    """
    다중 시간프레임 앙상블 모듈
    - 여러 시간프레임(5m, 15m, 30m, 1h, 4h, 1d)의 신호를 통합
    - 투표, 가중평균, 또는 다양한 앙상블 기법 적용
    """
    def __init__(self, timeframes=['5m', '15m', '30m', '1h', '4h', '1d'], 
                 weights=None, method='weighted_average'):
        self.timeframes = timeframes
        self.method = method  # 'voting', 'weighted_average', 'boosting'
        
        # 시간프레임별 가중치 설정 (기본: 장기간 시간프레임에 더 높은 가중치)
        if weights is None:
            # 시간프레임 길이에 비례하는 가중치 설정
            weights = {}
            base_weights = {
                '5m': 1.0,
                '15m': 1.5,
                '30m': 2.0,
                '1h': 2.5,
                '4h': 3.0,
                '1d': 4.0
            }
            total_weight = sum(base_weights[tf] for tf in timeframes if tf in base_weights)
            for tf in timeframes:
                if tf in base_weights:
                    weights[tf] = base_weights[tf] / total_weight
        
        self.weights = weights
        
        # 로깅 설정
        self.logger = logging.getLogger('MultiTimeframeEnsemble')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.logger.info(f"다중 시간프레임 앙상블 초기화: 방법={method}, 시간프레임={timeframes}")
        self.logger.info(f"시간프레임 가중치: {weights}")
    
    def combine_signals(self, signals):
        """
        여러 시간프레임의 신호를 통합
        
        Args:
            signals: dict - {timeframe: signal_value} 형태의 딕셔너리
                     signal_value는 -1.0 ~ 1.0 사이의 값 (-1: 강한 매도, 1: 강한 매수)
        
        Returns:
            float: 통합된 신호 (-1.0 ~ 1.0)
        """
        if not signals:
            return 0.0
        
        available_timeframes = [tf for tf in signals.keys() if tf in self.timeframes]
        if not available_timeframes:
            return 0.0
        
        if self.method == 'voting':
            # 투표 방식: 매수/매도/중립 카운트
            buy_votes = sum(1 for tf in available_timeframes if signals[tf] > 0.2)
            sell_votes = sum(1 for tf in available_timeframes if signals[tf] < -0.2)
            
            if buy_votes > sell_votes:
                # 매수 투표가 많으면 양수 신호
                return 0.3 + 0.7 * (buy_votes / len(available_timeframes))
            elif sell_votes > buy_votes:
                # 매도 투표가 많으면 음수 신호
                return -0.3 - 0.7 * (sell_votes / len(available_timeframes))
            else:
                # 동점이면 중립
                return 0.0
        
        elif self.method == 'weighted_average':
            # 가중 평균: 각 시간프레임에 가중치 적용
            total_weight = 0
            weighted_sum = 0
            
            for tf in available_timeframes:
                if tf in self.weights:
                    weight = self.weights[tf]
                    weighted_sum += signals[tf] * weight
                    total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        elif self.method == 'boosting':
            # 부스팅: 최근 성과가 좋은 시간프레임에 더 높은 가중치 부여
            # (이 예시에서는 단순화를 위해 가중 평균과 동일하게 구현)
            return self.combine_signals(signals, method='weighted_average')
        
        else:
            # 기본 방식: 단순 평균
            return sum(signals[tf] for tf in available_timeframes) / len(available_timeframes)
    
    def update_weights(self, performance_metrics):
        """
        성능 지표에 따라 시간프레임 가중치 동적 업데이트 (부스팅)
        
        Args:
            performance_metrics: dict - {timeframe: performance_score} 형태의 딕셔너리
        """
        if self.method != 'boosting':
            return
        
        # 성능 지표 정규화
        total_perf = sum(max(0.1, score) for score in performance_metrics.values())
        
        # 가중치 업데이트
        for tf in self.timeframes:
            if tf in performance_metrics:
                # 성능이 좋을수록 가중치 증가
                self.weights[tf] = max(0.1, performance_metrics[tf]) / total_perf
        
        self.logger.info(f"시간프레임 가중치 업데이트: {self.weights}")

class ModelEnsemble:
    """
    다양한 모델 앙상블 모듈
    - CNN, LSTM, Transformer 등 다양한 모델의 예측을 통합
    - 투표, 가중평균, 스태킹 등 다양한 앙상블 기법 적용
    """
    def __init__(self, models=None, model_weights=None, method='weighted_average'):
        self.models = models if models else {}  # 모델 딕셔너리: {model_name: model_obj}
        self.method = method  # 'voting', 'weighted_average', 'stacking'
        
        # 모델별 가중치 (기본: 동일 가중치)
        if model_weights is None and models:
            model_weights = {name: 1.0 / len(models) for name in models.keys()}
        
        self.weights = model_weights if model_weights else {}
        
        # 스태킹용 메타 모델 (단순 선형 모델)
        self.meta_model = torch.nn.Linear(len(self.models) if self.models else 1, 1)
        self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)
        
        # 로깅 설정
        self.logger = logging.getLogger('ModelEnsemble')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.logger.info(f"모델 앙상블 초기화: 방법={method}, 모델 수={len(self.models) if self.models else 0}")
        if models:
            self.logger.info(f"모델 가중치: {self.weights}")
    
    def predict(self, predictions):
        """
        combine_predictions 메소드의 별칭으로, 모델 예측값을 통합합니다.
        
        Args:
            predictions: dict - {model_name: prediction_value} 형태의 딕셔너리
            
        Returns:
            float 또는 int: 통합된 예측 값
        """
        return self.combine_predictions(predictions)
    
    def add_model(self, name, model, weight=None):
        """새 모델 추가"""
        self.models[name] = model
        
        # 가중치 업데이트
        if weight is None:
            # 모든 모델에 동일 가중치 부여
            weight = 1.0 / len(self.models)
            self.weights = {model_name: weight for model_name in self.models.keys()}
        else:
            # 기존 가중치 비율 유지하면서 새 모델 가중치 추가
            total_existing_weight = sum(self.weights.values()) if self.weights else 0
            if total_existing_weight > 0:
                for model_name in self.weights:
                    self.weights[model_name] *= (1 - weight)
            self.weights[name] = weight
        
        # 스태킹 메타 모델 재초기화
        if self.method == 'stacking':
            self.meta_model = torch.nn.Linear(len(self.models), 1)
            self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)
        
        self.logger.info(f"모델 추가: {name}, 가중치: {weight}")
    
    def remove_model(self, name):
        """모델 제거"""
        if name in self.models:
            del self.models[name]
            
            # 가중치 재조정
            if name in self.weights:
                del self.weights[name]
                if self.models:
                    total_weight = sum(self.weights.values())
                    for model_name in self.weights:
                        self.weights[model_name] /= total_weight
            
            # 스태킹 메타 모델 재초기화
            if self.method == 'stacking' and self.models:
                self.meta_model = torch.nn.Linear(len(self.models), 1)
                self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)
            
            self.logger.info(f"모델 제거: {name}")
    
    def combine_predictions(self, predictions):
        """
        여러 모델의 예측을 통합
        
        Args:
            predictions: dict - {model_name: prediction_value} 형태의 딕셔너리
                         prediction_value는 -1.0 ~ 1.0 사이의 값 (-1: 강한 매도, 1: 강한 매수)
        
        Returns:
            float: 통합된 예측 (-1.0 ~ 1.0)
        """
        if not predictions or not self.models:
            return 0.0
        
        available_models = [name for name in predictions.keys() if name in self.models]
        if not available_models:
            return 0.0
        
        if self.method == 'voting':
            # 투표 방식: 매수/매도/중립 카운트
            buy_votes = sum(1 for name in available_models if predictions[name] > 0.2)
            sell_votes = sum(1 for name in available_models if predictions[name] < -0.2)
            
            if buy_votes > sell_votes:
                # 매수 투표가 많으면 양수 신호
                return 0.3 + 0.7 * (buy_votes / len(available_models))
            elif sell_votes > buy_votes:
                # 매도 투표가 많으면 음수 신호
                return -0.3 - 0.7 * (sell_votes / len(available_models))
            else:
                # 동점이면 중립
                return 0.0
        
        elif self.method == 'weighted_average':
            # 가중 평균: 각 모델에 가중치 적용
            total_weight = 0
            weighted_sum = 0
            
            for name in available_models:
                if name in self.weights:
                    weight = self.weights[name]
                    weighted_sum += predictions[name] * weight
                    total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        elif self.method == 'stacking':
            # 스태킹: 메타 모델로 예측 통합
            inputs = torch.FloatTensor([predictions[name] for name in available_models])
            with torch.no_grad():
                return self.meta_model(inputs).item()
        
        else:
            # 기본 방식: 단순 평균
            return sum(predictions[name] for name in available_models) / len(available_models)
    
    def train_stacking(self, predictions_history, actual_returns):
        """
        스태킹 메타 모델 학습
        
        Args:
            predictions_history: list of dict - 각 시점별 모델 예측 기록
            actual_returns: list of float - 각 시점별 실제 수익률
        """
        if self.method != 'stacking' or not self.models:
            return
        
        # 데이터 준비
        X = []
        y = []
        
        for preds, ret in zip(predictions_history, actual_returns):
            if all(name in preds for name in self.models):
                X.append([preds[name] for name in self.models])
                y.append(ret)
        
        if not X:
            return
        
        # 텐서 변환
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        # 학습
        epochs = 100
        for epoch in range(epochs):
            self.meta_optimizer.zero_grad()
            outputs = self.meta_model(X_tensor)
            loss = torch.nn.functional.mse_loss(outputs, y_tensor)
            loss.backward()
            self.meta_optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f"스태킹 메타 모델 학습: 에포크 {epoch+1}/{epochs}, 손실 {loss.item():.6f}")
    
    def update_weights(self, performance_metrics):
        """
        성능 지표에 따라 모델 가중치 동적 업데이트
        
        Args:
            performance_metrics: dict - {model_name: performance_score} 형태의 딕셔너리
        """
        if not performance_metrics or not self.models:
            return
        
        # 성능 지표 정규화
        total_perf = sum(max(0.1, score) for name, score in performance_metrics.items() 
                         if name in self.models)
        
        if total_perf <= 0:
            return
        
        # 가중치 업데이트
        for name in self.models:
            if name in performance_metrics:
                # 성능이 좋을수록 가중치 증가
                self.weights[name] = max(0.1, performance_metrics[name]) / total_perf
        
        self.logger.info(f"모델 가중치 업데이트: {self.weights}")

class EnsembleTrader:
    """
    멀티모달 앙상블 트레이딩 시스템
    - 다중 시간프레임 앙상블
    - 다양한 모델 앙상블
    - 포트폴리오 최적화
    """
    def __init__(self, config=None):
        self.config = config or {}
        
        # 다중 시간프레임 앙상블 초기화
        timeframes = self.config.get('timeframes', ['5m', '15m', '30m', '1h', '4h', '1d'])
        tf_weights = self.config.get('timeframe_weights', None)
        tf_method = self.config.get('timeframe_ensemble_method', 'weighted_average')
        self.timeframe_ensemble = MultiTimeframeEnsemble(
            timeframes=timeframes,
            weights=tf_weights,
            method=tf_method
        )
        
        # 모델 앙상블 초기화
        model_weights = self.config.get('model_weights', None)
        model_method = self.config.get('model_ensemble_method', 'weighted_average')
        self.model_ensemble = ModelEnsemble(
            model_weights=model_weights,
            method=model_method
        )
        
        # 거래 기록
        self.trades = []
        self.portfolio_values = []
        
        # 균형 조정 파라미터
        self.rebalance_window = self.config.get('rebalance_window', 20)  # 20일마다 포트폴리오 조정
        self.risk_tolerance = self.config.get('risk_tolerance', 0.02)    # 최대 허용 리스크 (변동성)
        
        # 결과 저장 경로
        self.results_dir = self.config.get('results_dir', 'ensemble_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 로깅 설정
        self.logger = logging.getLogger('EnsembleTrader')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.logger.info("멀티모달 앙상블 트레이딩 시스템 초기화")
    
    def add_model(self, name, model, weight=None):
        """모델 앙상블에 새 모델 추가"""
        self.model_ensemble.add_model(name, model, weight)
    
    def execute_trades(self, timeframe_signals, model_predictions, price_data):
        """
        앙상블 신호에 따라 거래 결정 및 실행
        
        Args:
            timeframe_signals: dict - 각 시간프레임별 신호
            model_predictions: dict - 각 모델별 예측
            price_data: DataFrame - 가격 데이터
        
        Returns:
            dict: 거래 결과
        """
        # 다중 시간프레임 앙상블 신호
        tf_ensemble_signal = self.timeframe_ensemble.combine_signals(timeframe_signals)
        
        # 다양한 모델 앙상블 예측
        model_ensemble_signal = self.model_ensemble.combine_predictions(model_predictions)
        
        # 최종 앙상블 신호 (시간프레임:모델 = 6:4 비율 적용)
        final_signal = 0.6 * tf_ensemble_signal + 0.4 * model_ensemble_signal
        
        # 거래 결정
        position = 'neutral'
        if final_signal > 0.3:
            position = 'long'
        elif final_signal < -0.3:
            position = 'short'
        
        # 거래 기록 추가
        trade = {
            'timestamp': price_data.index[-1],
            'price': price_data['close'].iloc[-1],
            'tf_signal': tf_ensemble_signal,
            'model_signal': model_ensemble_signal,
            'final_signal': final_signal,
            'position': position
        }
        self.trades.append(trade)
        
        return trade
    
    def calculate_portfolio_value(self, initial_balance=10000):
        """포트폴리오 가치 계산"""
        if not self.trades:
            return [{'timestamp': None, 'value': initial_balance}]
        
        balance = initial_balance
        position = 'neutral'
        entry_price = 0
        
        portfolio_values = []
        
        for i, trade in enumerate(self.trades):
            timestamp = trade['timestamp']
            price = trade['price']
            new_position = trade['position']
            
            # 포지션 변경 시 손익 계산
            if position != new_position:
                if position == 'long':
                    # 롱 포지션 청산
                    pnl = (price - entry_price) / entry_price
                    balance *= (1 + pnl)
                elif position == 'short':
                    # 숏 포지션 청산
                    pnl = (entry_price - price) / entry_price
                    balance *= (1 + pnl)
                
                # 새 포지션 진입
                position = new_position
                if position != 'neutral':
                    entry_price = price
            
            # 포트폴리오 가치 기록
            portfolio_values.append({
                'timestamp': timestamp,
                'value': balance,
                'position': position
            })
        
        return portfolio_values
    
    def calculate_performance_metrics(self):
        """성능 지표 계산"""
        if not self.portfolio_values:
            return {}
        
        # 데이터프레임 변환
        df = pd.DataFrame(self.portfolio_values)
        
        # 일별 수익률 계산
        df['daily_return'] = df['value'].pct_change().fillna(0)
        
        # 누적 수익률
        total_return = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
        
        # 연간 수익률 (252 거래일 기준)
        num_days = len(df)
        annual_return = (1 + total_return) ** (252 / num_days) - 1
        
        # 변동성 (표준편차)
        volatility = df['daily_return'].std() * np.sqrt(252)
        
        # 샤프 비율
        risk_free_rate = 0.02  # 2% 무위험 수익률 가정
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 최대 낙폭
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        df['cumulative_max'] = df['cumulative_return'].cummax()
        df['drawdown'] = df['cumulative_max'] - df['cumulative_return']
        max_drawdown = df['drawdown'].max()
        
        # 거래 횟수
        position_changes = (df['position'] != df['position'].shift(1)).sum()
        
        # 성공 거래 비율
        df['trade_pnl'] = 0.0
        for i in range(1, len(df)):
            if df['position'].iloc[i] != df['position'].iloc[i-1]:
                # 포지션 변경 시
                if df['position'].iloc[i-1] == 'long':
                    df.loc[df.index[i], 'trade_pnl'] = df['value'].iloc[i] / df['value'].iloc[i-1] - 1
                elif df['position'].iloc[i-1] == 'short':
                    df.loc[df.index[i], 'trade_pnl'] = 1 - df['value'].iloc[i] / df['value'].iloc[i-1]
        
        winning_trades = (df['trade_pnl'] > 0).sum()
        win_rate = winning_trades / position_changes if position_changes > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': position_changes,
            'win_rate': win_rate
        }
    
    def save_results(self):
        """결과 저장"""
        if not self.trades or not self.portfolio_values:
            self.logger.warning("저장할 거래 기록이 없습니다.")
            return
        
        # 거래 기록 저장
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(os.path.join(self.results_dir, 'trades.csv'), index=False)
        
        # 포트폴리오 가치 저장
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.to_csv(os.path.join(self.results_dir, 'portfolio_values.csv'), index=False)
        
        # 성능 지표 계산 및 저장
        metrics = self.calculate_performance_metrics()
        with open(os.path.join(self.results_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 시각화 및 저장
        self.visualize_results()
        
        self.logger.info(f"결과가 {self.results_dir}에 저장되었습니다.")
    
    def visualize_results(self):
        """결과 시각화"""
        if not self.portfolio_values:
            return
        
        # 데이터프레임 변환
        portfolio_df = pd.DataFrame(self.portfolio_values)
        
        # 포트폴리오 가치 변화 그래프
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df['timestamp'], portfolio_df['value'])
        plt.title('포트폴리오 가치 변화')
        plt.xlabel('날짜')
        plt.ylabel('가치')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'portfolio_value.png'))
        
        # 다중 시간프레임 앙상블 신호 히스토그램
        trades_df = pd.DataFrame(self.trades)
        if 'tf_signal' in trades_df.columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(trades_df['tf_signal'], bins=50, kde=True)
            plt.title('다중 시간프레임 앙상블 신호 분포')
            plt.xlabel('신호 강도')
            plt.ylabel('빈도')
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, 'timeframe_signals.png'))
        
        # 모델 앙상블 신호 히스토그램
        if 'model_signal' in trades_df.columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(trades_df['model_signal'], bins=50, kde=True)
            plt.title('모델 앙상블 신호 분포')
            plt.xlabel('신호 강도')
            plt.ylabel('빈도')
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, 'model_signals.png'))
        
        # 최종 신호 히스토그램
        if 'final_signal' in trades_df.columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(trades_df['final_signal'], bins=50, kde=True)
            plt.title('최종 앙상블 신호 분포')
            plt.xlabel('신호 강도')
            plt.ylabel('빈도')
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, 'final_signals.png'))
        
        # 포지션 분포 파이 차트
        if 'position' in trades_df.columns:
            position_counts = trades_df['position'].value_counts()
            plt.figure(figsize=(8, 8))
            plt.pie(position_counts, labels=position_counts.index, autopct='%1.1f%%')
            plt.title('포지션 분포')
            plt.savefig(os.path.join(self.results_dir, 'position_distribution.png'))
        
        plt.close('all')

# 모델 사용 예시
def test_ensemble_trader():
    # 설정
    config = {
        'timeframes': ['5m', '15m', '30m', '1h', '4h', '1d'],
        'timeframe_ensemble_method': 'weighted_average',
        'model_ensemble_method': 'weighted_average',
        'results_dir': 'ensemble_results'
    }
    
    # 앙상블 트레이더 초기화
    trader = EnsembleTrader(config)
    
    # 예시 신호 및 가격 데이터
    timeframe_signals = {
        '5m': 0.2,
        '15m': 0.3,
        '30m': 0.1,
        '1h': -0.1,
        '4h': -0.2,
        '1d': 0.4
    }
    
    model_predictions = {
        'cnn': 0.3,
        'lstm': 0.2,
        'transformer': 0.1,
        'random_forest': -0.1
    }
    
    # 가격 데이터 (간단한 예시)
    dates = pd.date_range(start='2021-01-01', periods=10, freq='D')
    prices = [100, 101, 102, 103, 102, 101, 102, 103, 104, 105]
    price_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * len(prices)
    }, index=dates)
    
    # 거래 실행
    for i in range(len(price_data)):
        # 간단한 예시를 위해 동일한 신호 사용
        trade = trader.execute_trades(timeframe_signals, model_predictions, price_data.iloc[:i+1])
        print(f"날짜: {price_data.index[i]}, 가격: {price_data['close'].iloc[i]}, 신호: {trade['final_signal']:.2f}, 포지션: {trade['position']}")
    
    # 포트폴리오 가치 계산
    portfolio_values = trader.calculate_portfolio_value()
    
    # 성능 지표 계산
    metrics = trader.calculate_performance_metrics()
    print("\n성능 지표:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 결과 저장
    trader.save_results()
    print(f"\n결과가 {config['results_dir']}에 저장되었습니다.")

if __name__ == "__main__":
    test_ensemble_trader() 