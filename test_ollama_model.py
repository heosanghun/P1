#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
딥시크r1(32b) 모델 테스트
"""

import os
import logging
from datetime import datetime
from ollama_sentiment_analyzer import OllamaSentimentAnalyzer
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("="*80)
    print("딥시크r1(32b) 모델 테스트")
    print("="*80)
    
    # 결과 디렉토리 생성
    results_dir = f"results/ollama_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Ollama 설정
    config = {
        'model_name': 'deepseek-llm:latest',
        'use_deepseek_r1': True,
        'offline_mode': False,
        'output': {
            'save_dir': results_dir
        }
    }
    
    # Ollama 감성 분석기 초기화
    print("\n1. 딥시크r1(32b) 감성 분석기 초기화 중...")
    analyzer = OllamaSentimentAnalyzer(config)
    
    # 테스트 뉴스 항목 생성
    test_news = [
        "Bitcoin has reached a new all-time high as institutional investors continue to show interest in the cryptocurrency.",
        "Technical analysts are pointing to bullish patterns forming in the charts of major crypto assets.",
        "Despite recent gains, Bitcoin is facing strong resistance at the $50,000 level, with some analysts predicting a potential correction.",
        "Recent regulatory announcements have caused significant volatility in the cryptocurrency market.",
        "Traders and investors remain cautious as Bitcoin struggles to break through key resistance levels."
    ]
    
    # 테스트 데이터프레임 생성
    test_df = pd.DataFrame({
        'date': pd.date_range(start='2025-05-01', periods=len(test_news)),
        'title': test_news,
        'content': test_news
    })
    
    # 개별 뉴스 테스트
    print("\n2. 개별 뉴스 항목 테스트:")
    for i, news in enumerate(test_news, 1):
        print(f"\n테스트 {i}: \"{news[:60]}...\"")
        result = analyzer.analyze(news)
        print(f"  → 감성: {result['overall_sentiment']}, 점수: {result.get('sentiment_score', 0.0):.2f}")
    
    # 전체 뉴스 데이터 분석
    print("\n3. 전체 뉴스 데이터 감성 분석:")
    print(f"  총 {len(test_df)}개 뉴스 항목 분석 중...")
    all_results = analyzer.analyze_sentiment(test_df)
    
    # 결과 출력
    print("\n4. 전체 감성 분석 결과:")
    print(f"  종합 감성: {all_results['overall_sentiment']}")
    print(f"  감성 점수: {all_results['sentiment_score']:.2f}")
    print(f"  긍정 비율: {all_results['bullish_ratio']*100:.1f}%")
    print(f"  부정 비율: {all_results['bearish_ratio']*100:.1f}%")
    print(f"  중립 비율: {all_results['neutral_ratio']*100:.1f}%")
    
    # 결과 저장
    analyzer.save_data(all_results, 'sentiment_results.json')
    print(f"\n결과 저장 완료: {results_dir}")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    main() 