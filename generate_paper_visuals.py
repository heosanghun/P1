import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import seaborn as sns
from datetime import datetime

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#E5E5E5'
plt.rcParams['grid.linestyle'] = '--'

# 결과 디렉토리 설정
results_dir = os.path.dirname(os.path.abspath(__file__)) + "/results"
output_dir = results_dir
latest_run_dir = None

# 가장 최신 실행 결과 디렉토리 찾기
run_dirs = [d for d in os.listdir(results_dir) if d.startswith('run_')]
if run_dirs:
    latest_run_dir = max(run_dirs)
    print(f"최신 실행 결과 디렉토리: {latest_run_dir}")
else:
    print("실행 결과 디렉토리를 찾을 수 없습니다.")
    exit(1)

run_path = os.path.join(results_dir, latest_run_dir)

# 포트폴리오 성능 그래프 생성
def generate_portfolio_performance_graph():
    try:
        portfolio_data = pd.read_csv(os.path.join(run_path, 'portfolio_history.csv'))
        
        # 날짜 포맷 변환
        portfolio_data['timestamp'] = pd.to_datetime(portfolio_data['timestamp'])
        
        plt.figure(figsize=(12, 8))
        
        # 포트폴리오 가치
        plt.plot(portfolio_data['timestamp'], portfolio_data['portfolio_value'], 
                 linewidth=2.5, color='#1f77b4', label='포트폴리오 가치')
        
        # 기준선 (Buy & Hold)
        if 'benchmark_value' in portfolio_data.columns:
            plt.plot(portfolio_data['timestamp'], portfolio_data['benchmark_value'], 
                     linewidth=2, color='#ff7f0e', linestyle='--', label='Buy & Hold')
        
        # 그래프 포맷팅
        plt.title('포트폴리오 성능', fontsize=16, fontweight='bold')
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('포트폴리오 가치 (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # x축 날짜 포맷 설정
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # 저장
        output_file = os.path.join(output_dir, 'portfolio_performance_for_paper.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"포트폴리오 성능 그래프가 저장되었습니다: {output_file}")
    except Exception as e:
        print(f"포트폴리오 성능 그래프 생성 중 오류 발생: {str(e)}")

# 드로다운 곡선 생성
def generate_drawdown_curve():
    try:
        portfolio_data = pd.read_csv(os.path.join(run_path, 'portfolio_history.csv'))
        portfolio_data['timestamp'] = pd.to_datetime(portfolio_data['timestamp'])
        
        # 최대 누적 가치 계산
        portfolio_data['cummax'] = portfolio_data['portfolio_value'].cummax()
        
        # 드로다운 계산
        portfolio_data['drawdown'] = (portfolio_data['portfolio_value'] - portfolio_data['cummax']) / portfolio_data['cummax'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_data['timestamp'], portfolio_data['drawdown'], 
                 linewidth=2, color='#d62728')
        plt.fill_between(portfolio_data['timestamp'], portfolio_data['drawdown'], 0, 
                          color='#d62728', alpha=0.3)
        
        plt.title('포트폴리오 드로다운', fontsize=16, fontweight='bold')
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('드로다운 (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # x축 날짜 포맷 설정
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # y축 반전
        plt.gca().invert_yaxis()
        
        # 저장
        output_file = os.path.join(output_dir, 'drawdown_curve_for_paper.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"드로다운 곡선이 저장되었습니다: {output_file}")
    except Exception as e:
        print(f"드로다운 곡선 생성 중 오류 발생: {str(e)}")

# 거래 분포 생성
def generate_trade_distribution():
    try:
        signals_file = os.path.join(run_path, 'signals_history.csv')
        if os.path.exists(signals_file):
            signals_data = pd.read_csv(signals_file)
            
            # 신호 집계
            signals_count = signals_data['signal'].value_counts()
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(signals_count.index, signals_count.values, color=['#2ca02c', '#d62728', '#7f7f7f'])
            
            # 바 레이블 추가
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{height:.0f}', ha='center', va='bottom', fontsize=11)
            
            plt.title('거래 신호 분포', fontsize=16, fontweight='bold')
            plt.xlabel('신호 유형', fontsize=12)
            plt.ylabel('횟수', fontsize=12)
            plt.xticks(rotation=0, fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')
            
            # 저장
            output_file = os.path.join(output_dir, 'trade_distribution_for_paper.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"거래 분포 그래프가 저장되었습니다: {output_file}")
        else:
            print(f"거래 신호 파일을 찾을 수 없습니다: {signals_file}")
    except Exception as e:
        print(f"거래 분포 그래프 생성 중 오류 발생: {str(e)}")

# 성능 지표 생성
def generate_performance_metrics():
    try:
        # JSON 성능 지표 읽기
        metrics_file = os.path.join(run_path, 'performance_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # CSV 형식으로 저장
            metrics_df = pd.DataFrame([metrics])
            csv_output = os.path.join(output_dir, 'performance_metrics_for_paper.csv')
            metrics_df.to_csv(csv_output, index=False)
            
            # LaTeX 테이블 생성
            latex_table = "\\begin{table}[h]\n\\centering\n\\caption{Trading Strategy Performance Metrics}\n\\begin{tabular}{lc}\n\\hline\n"
            latex_table += "Metric & Value \\\\\n\\hline\n"
            
            # 주요 지표 포맷팅 및 추가
            for key, value in metrics.items():
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, float):
                    if 'return' in key.lower() or 'ratio' in key.lower():
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                latex_table += f"{formatted_key} & {formatted_value} \\\\\n"
            
            latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
            
            # LaTeX 저장
            latex_output = os.path.join(output_dir, 'performance_metrics_for_paper.tex')
            with open(latex_output, 'w') as f:
                f.write(latex_table)
            
            print(f"성능 지표가 저장되었습니다: {csv_output}, {latex_output}")
        else:
            print(f"성능 지표 파일을 찾을 수 없습니다: {metrics_file}")
    except Exception as e:
        print(f"성능 지표 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    print("논문용 시각적 이미지 및 성능 지표 결과 생성을 시작합니다...")
    generate_portfolio_performance_graph()
    generate_drawdown_curve()
    generate_trade_distribution()
    generate_performance_metrics()
    print("모든 시각적 이미지 및 성능 지표 결과 생성이 완료되었습니다!") 