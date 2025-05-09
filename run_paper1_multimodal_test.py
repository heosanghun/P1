# paper1 메인 실행 스크립트 예시
from basic_trader import BasicTrader
from candlestick_analyzer import CandlestickAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from ollama_sentiment_analyzer import OllamaSentimentAnalyzer  # 추가: Ollama 감성 분석기
import os
import torch
import tqdm
import time
from datetime import datetime
import sys
import threading

# 실시간 진행률 모니터링을 위한 클래스 추가
class ProgressMonitor:
    def __init__(self, update_interval=1.0):
        self.start_time = time.time()
        self.is_running = True
        self.progress = 0
        self.status = "초기화 중..."
        self.update_interval = update_interval
        self.thread = None
        
    def start(self):
        self.thread = threading.Thread(target=self._monitor_thread)
        self.thread.daemon = True
        self.thread.start()
        
    def _monitor_thread(self):
        while self.is_running:
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            sys.stdout.write(f"\r[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"진행률: {self.progress:.2f}% | "
                            f"경과 시간: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} | "
                            f"상태: {self.status}")
            sys.stdout.flush()
            time.sleep(self.update_interval)
            
    def update(self, progress, status):
        self.progress = progress
        self.status = status
        
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"완료! 총 소요 시간: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

if __name__ == "__main__":
    # 시작 시간 기록
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 멀티모달 트레이딩 시뮬레이션 시작")
    
    # 진행률 모니터링 시작
    progress_monitor = ProgressMonitor()
    progress_monitor.start()
    
    # GPU 확인 및 사용 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 사용 가능 디바이스: {device}")
    progress_monitor.update(5, f"디바이스 설정: {device}")
    
    # 루트 디렉토리 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    data_root_dir = os.path.join(root_dir, 'data')
    progress_monitor.update(10, "데이터 경로 설정 완료")
    
    # 설정 정의
    config = {
        'data': {
            'chart_dir': os.path.join(data_root_dir, 'chart'),  # 차트 데이터 경로 수정
            'news_file': os.path.join(data_root_dir, 'news', 'cryptonews_2021-10-12_2023-12-19.csv'),  # 뉴스 데이터 파일 경로 수정
            'timeframes': ['1d', '4h', '1h', '30m', '15m', '5m']  # 분석할 시간프레임 (1d부터 역순)
        },
        'output': {
            'save_dir': os.path.join(current_dir, 'results')  # 결과 저장 경로
        },
        # 고급 기술 설정 추가
        'fusion': {
            'type': 'attention'  # 'attention' 또는 'transformer'
        },
        'rl': {
            'use_rl': True,  # 강화학습 사용 여부
            'learning_rate': 0.0003,
            'gamma': 0.99
        },
        'ensemble': {
            'use_ensemble': True,  # 앙상블 방법론 사용 여부
            'timeframe_ensemble_method': 'weighted_average',  # 'voting', 'weighted_average', 'boosting'
            'model_ensemble_method': 'weighted_average'  # 'voting', 'weighted_average', 'stacking'
        },
        # Ollama 설정 추가
        'ollama': {
            'use_deepseek': True,  # 딥시크 모델 사용 여부
            'model_name': 'deepseek-llm:latest',  # 모델명
            'use_deepseek_r1': True,  # 딥시크r1(32b) 모델 명시적 사용
            'offline_mode': False  # 오프라인 모드 (False면, 온라인 API 호출)
        },
        # 실시간 모니터링 설정 추가
        'monitoring': {
            'enable': True,
            'update_interval': 1.0  # 초 단위 업데이트 간격
        }
    }
    progress_monitor.update(15, "설정 구성 완료")
    
    # 데이터 디렉토리 확인 및 생성
    os.makedirs(config['data']['chart_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['data']['news_file']), exist_ok=True)
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    progress_monitor.update(20, "데이터 디렉토리 확인 완료")
    
    # 분석기 초기화 (설정 전달)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 캔들스틱 분석기 초기화 중...")
    progress_monitor.update(25, "캔들스틱 분석기 초기화 중...")
    candlestick_analyzer = CandlestickAnalyzer(config)
    progress_monitor.update(30, "캔들스틱 분석기 초기화 완료")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 감성 분석기 초기화 중...")
    progress_monitor.update(35, "감성 분석기 초기화 중...")
    # 기존 감성 분석기
    sentiment_analyzer_vader = SentimentAnalyzer(config)
    progress_monitor.update(40, "VADER 감성 분석기 초기화 완료")
    
    # Ollama 딥시크 감성 분석기 추가
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 딥시크r1(32b) 감성 분석기 초기화 중...")
    progress_monitor.update(45, "딥시크r1(32b) 감성 분석기 초기화 중...")
    ollama_config = {
        'model_name': config['ollama']['model_name'],
        'use_deepseek_r1': config['ollama']['use_deepseek_r1'],
        'offline_mode': config['ollama']['offline_mode'],
        'output': {
            'save_dir': os.path.join(config['output']['save_dir'], 'ollama_results')
        }
    }
    ollama_analyzer = OllamaSentimentAnalyzer(ollama_config)
    progress_monitor.update(50, "딥시크r1(32b) 감성 분석기 초기화 완료")
    
    # 간단한 테스트 실행
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 딥시크r1(32b) 감성 분석 테스트 실행...")
    progress_monitor.update(55, "딥시크 테스트 실행 중...")
    test_text = "Bitcoin has reached a new all-time high as institutional investors continue to show interest in the cryptocurrency."
    test_result = ollama_analyzer.analyze(test_text)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 테스트 결과: 감성={test_result['overall_sentiment']}, 점수={test_result.get('sentiment_score', 0.0)}")
    progress_monitor.update(60, "딥시크 테스트 완료")
    
    # 트레이더 초기화 및 실행
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 트레이더 초기화 중...")
    progress_monitor.update(65, "트레이더 초기화 중...")
    
    # BasicTrader 초기화 시 progress_callback 함수 전달
    trader = BasicTrader(
        candlestick_analyzer, 
        sentiment_analyzer_vader, 
        config,
        progress_callback=progress_monitor.update  # 콜백 함수 추가
    )
    progress_monitor.update(70, "트레이더 초기화 완료")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 트레이딩 시뮬레이션 실행 중...")
    progress_monitor.update(75, "트레이딩 시뮬레이션 실행 중...")
    trader.run()
    progress_monitor.update(80, "트레이딩 시뮬레이션 완료")
    
    # 성능 평가 및 결과 저장
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 성능 평가 중...")
    progress_monitor.update(85, "성능 평가 중...")
    trader.evaluate_performance()
    progress_monitor.update(90, "성능 평가 완료")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 결과 저장 중...")
    progress_monitor.update(95, "결과 저장 중...")
    # 결과 저장 메서드가 없으면 수정
    if hasattr(trader, 'save_results'):
        trader.save_results()
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 주의: trader 객체에 save_results 메서드가 없습니다.")
    
    # 논문 제출용 결과 생성
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 논문 제출용 결과 생성 중...")
    trader.paper_report_for_submission()
    progress_monitor.update(98, "논문 제출용 결과 생성 완료")
    
    # 모니터링 종료
    progress_monitor.update(100, "작업 완료")
    progress_monitor.stop()
    
    # 실행 시간 계산
    elapsed_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 멀티모달 트레이딩 시뮬레이션 완료! (소요 시간: {elapsed_time:.2f}초)")
