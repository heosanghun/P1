# Paper 1 Directory Structure and Operating Principles

## Paper 1 Directory Structure

```
paper1/
├── __init__.py                          # Package initialization file
├── advanced_multimodal_trader.py        # Advanced multimodal trader implementation
├── basic_trader.py                      # Basic trading system implementation
├── ensemble_trader.py                   # Ensemble trading strategies
├── rl_trader.py                         # Reinforcement learning trader
├── candlestick_analyzer.py              # Candlestick pattern analysis module
├── sentiment_analyzer.py                # News sentiment analysis module
├── ollama_sentiment_analyzer.py         # DeepSeek R1 기반 감성 분석기
├── multimodal_fusion.py                 # Multimodal data fusion implementation
├── feature_fusion.py                    # Feature fusion strategies
├── trading_strategy.py                  # Trading strategy implementations
├── backtesting.py                       # Backtesting framework
├── run_paper1_multimodal_test.py        # Paper 1 execution script
├── run_advanced_multimodal_trader.py    # Advanced multimodal trader execution
├── generate_paper_visuals.py            # Result visualization generation
├── generate_paper_visuals_fixed.py      # Fixed timeframe visualization
├── generate_paper_visuals_agg.py        # Aggregated performance visualization
├── generate_research_graphs.py          # Research graphs generation
├── binance_api.py                       # Binance API connection module
├── news_collector.py                    # News data collection module
├── test_data_generator.py               # Test data generation utilities
├── test_ollama_model.py                 # DeepSeek model testing script
├── test_basic_trader.py                 # Basic trader testing utilities
├── tradingview/                         # TradingView integration
├── docs/                                # Documentation
├── requirements.txt                     # Required dependencies
├── run_all.bat                          # Windows execution batch file
├── install.bat                          # Windows installation batch file
├── run_all.sh                           # Linux execution script
├── install.sh                           # Linux installation script
├── LICENSE                              # License file
├── .gitignore                           # Git ignore configuration
└── results/                             # Simulation results directory
    └── run_YYYYMMDD_HHMMSS/             # Individual run results directory
        ├── portfolio_history.csv        # Portfolio value history
        ├── performance_metrics.json     # Performance metrics
        └── trade_history.csv            # Trade history
```

## 딥시크(DeepSeek) 기반 감성 분석기

본 프로젝트는 기존 NLTK VADER 기반 감성 분석기와 함께 더 성능이 좋은 **DeepSeek_r1(32b)** 대규모 언어 모델을 활용한 감성 분석기를 제공합니다:

### 주요 특징
- **Ollama API 연동**: 로컬에서 실행되는 Ollama 서버를 통해 DeepSeek 모델 활용
- **정확한 감성 분석**: 암호화폐 뉴스에 대한 세 가지 감성(bullish, bearish, neutral) 분류
- **정량적 감성 점수**: -1.0에서 1.0 사이의 연속적인 감성 점수 부여
- **JSON 응답 파싱**: 구조화된 형태로 감성 분석 결과 제공
- **대체 로직 제공**: API 연결 실패 시 텍스트 기반 대체 분석 수행
- **배치 처리 기능**: 다수의 뉴스 항목 일괄 처리 지원

### 관련 파일
- `ollama_sentiment_analyzer.py`: 딥시크 모델 기반 감성 분석기 구현
- `run_paper1_multimodal_test.py`: 감성 분석기 통합 및 테스트
- `test_ollama_model.py`: 감성 분석기 단독 테스트

### 사용 방법

```python
from ollama_sentiment_analyzer import OllamaSentimentAnalyzer

# 설정 정의
config = {
    'model_name': 'deepseek-llm:latest',  # 모델명
    'use_deepseek_r1': True,              # 딥시크r1(32b) 모델 사용
    'offline_mode': False                 # 온라인 모드 (API 호출)
}

# 감성 분석기 초기화
analyzer = OllamaSentimentAnalyzer(config)

# 단일 뉴스 분석
news_text = "Bitcoin has reached a new all-time high as institutional investors continue to show interest."
result = analyzer.analyze(news_text)
print(f"감성: {result['overall_sentiment']}, 점수: {result['sentiment_score']}")

# 뉴스 데이터셋 분석
import pandas as pd
news_df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02'],
    'title': ['Bitcoin rises', 'Ethereum falls'],
    'content': ['BTC price increased by 5%', 'ETH price decreased by 3%']
})
results = analyzer.analyze_sentiment(news_df)
print(f"종합 감성: {results['overall_sentiment']}, 점수: {results['sentiment_score']}")
```

## Paper 1 Operating Principles

### 1. Core Components

#### Candlestick Analyzer
- **Purpose**: Extract patterns and signals from candlestick chart images
- **Key Features**:
  - Image-based pattern recognition using CNN models
  - Generation of trading signals based on chart patterns
  - Integration with technical indicators

#### Sentiment Analyzer
- **Purpose**: Analyze news data to extract market sentiment
- **Key Features**:
  - Natural language processing for news sentiment analysis
  - Extraction of bullish/bearish keywords and signals
  - Time-series sentiment tracking and aggregation

#### Multimodal Fusion System
- **Purpose**: Integrate signals from different data sources (candlesticks, news, price)
- **Key Features**:
  - Attention-based fusion mechanisms
  - Transformer architecture for cross-modal interactions
  - Dynamic weighting of different signal sources

#### Reinforcement Learning Trader
- **Purpose**: Learn optimal trading policy through experience
- **Key Features**:
  - Proximal Policy Optimization (PPO) algorithm
  - State representation based on market features
  - Reward function based on portfolio performance

### 2. Data Processing Flow

1. **Data Collection and Preprocessing**:
   - Market data collection (OHLCV prices)
   - Candlestick chart image generation
   - News data collection and preprocessing

2. **Feature Extraction**:
   - Candlestick pattern feature extraction using CNN
   - Sentiment feature extraction from news data
   - Technical indicator calculation from price data

3. **Signal Fusion**:
   - Multimodal fusion of candlestick, sentiment, and price signals
   - Attention-based weighting of different data sources
   - Generation of unified trading signals

4. **Decision Making**:
   - Trading strategy selection and execution
   - Position sizing and risk management
   - Order execution and monitoring

5. **Performance Evaluation**:
   - Portfolio value tracking
   - Calculation of performance metrics (Sharpe ratio, maximum drawdown, returns)
   - Visualization and analysis of results

### 3. Core Algorithms

#### Candlestick Pattern Recognition
- Deep CNN (ResNet50) for image feature extraction
- Pattern classification based on historical price movements
- Signal strength measurement for detected patterns

#### Sentiment Analysis
- Keyword-based sentiment scoring
- Time-series sentiment aggregation and trending
- Sentiment impact analysis on price movements

#### Multimodal Fusion Algorithms
- Attention mechanism for dynamic feature weighting
- Transformer architecture for cross-modal relationships
- Late fusion strategy for decision optimization

#### Reinforcement Learning Strategy
- State representation using market features
- Action space: Buy, Sell, Hold actions
- PPO algorithm for policy optimization
- Reward function based on profit and risk

### 4. Key Features and Advantages

- **Multimodal Analysis**: Integration of visual, textual, and numerical data
- **Adaptive Learning**: Continuous model improvement through reinforcement learning
- **Cross-Validation**: Multiple signal sources for more robust decision making
- **Feature Importance**: Dynamic assessment of which features matter most
- **Risk Management**: Comprehensive position sizing and risk control

### 5. Data Download

프로젝트에 필요한 데이터셋은 다음 Google Drive 링크에서 다운로드할 수 있습니다:
- **Google Drive**: [https://drive.google.com/drive/folders/1vHxKgrkjguXfgmIOUWqbbSdD1XXDnbSK?usp=sharing](https://drive.google.com/drive/folders/1vHxKgrkjguXfgmIOUWqbbSdD1XXDnbSK?usp=sharing)
  
- 캔들 차트 이미지(224X224) 데이터 용량: 8.19GB/369,456장 | 2021-10-12 ~ 2023-12-19
- 암호화폐 뉴스 기사(감성분석) 데이터 용량: 12.6MB/31,038개 |  2021-10-12 ~ 2023-12-19
  
다운로드한 데이터는 `data/` 디렉토리에 배치하세요. 데이터셋에는 다음이 포함됩니다:
- 암호화폐 가격 이력 데이터
- 전처리된 뉴스 데이터
- 샘플 캔들스틱 차트 이미지
- 테스트 및 검증용 데이터셋

### 6. Simulation and Visualization

- Backtesting across various market conditions
- Performance comparison with baseline strategies
- Key metric visualization (returns, drawdowns, win rate)
- Time-series portfolio value tracking

### 7. Limitations and Future Work

- Potential overfitting to historical data patterns
- Computational intensity of real-time image analysis
- News data latency and relevance assessment challenges
- Need for continuous model retraining and adaptation 
