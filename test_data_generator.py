import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TestDataGenerator:
    @staticmethod
    def generate_candlestick_data(num_candles=100):
        # 랜덤 캔들스틱 데이터 생성
        dates = [datetime.now() - timedelta(hours=i) for i in range(num_candles)]
        open_prices = np.random.normal(50000, 1000, num_candles)
        high_prices = open_prices + np.random.uniform(0, 500, num_candles)
        low_prices = open_prices - np.random.uniform(0, 500, num_candles)
        close_prices = (high_prices + low_prices) / 2 + np.random.normal(0, 100, num_candles)
        volumes = np.random.uniform(100, 1000, num_candles)
        
        data = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        return data
    
    @staticmethod
    def generate_news_data(num_news=50):
        # 랜덤 뉴스 데이터 생성
        dates = [datetime.now() - timedelta(hours=i) for i in range(num_news)]
        headlines = [f"Test News {i+1}" for i in range(num_news)]
        contents = [f"This is test news content {i+1}" for i in range(num_news)]
        sentiments = np.random.uniform(-1, 1, num_news)  # -1: 부정, 1: 긍정
        
        data = pd.DataFrame({
            'date': dates,
            'headline': headlines,
            'content': contents,
            'sentiment': sentiments
        })
        return data 