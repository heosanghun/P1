import os

# 필수 폴더
for folder in ['data', 'results', 'docs']:
    os.makedirs(folder, exist_ok=True)

# requirements.txt 예시
if not os.path.exists('requirements.txt'):
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write("numpy\npandas\ntorch\nmatplotlib\nscikit-learn\n")

# run_paper1_multimodal_test.py 예시
if not os.path.exists('run_paper1_multimodal_test.py'):
    with open('run_paper1_multimodal_test.py', 'w', encoding='utf-8') as f:
        f.write(
            "# paper1 메인 실행 스크립트 예시\n"
            "from basic_trader import BasicTrader\n\n"
            "if __name__ == \"__main__\":\n"
            "    trader = BasicTrader()\n"
            "    trader.run()\n"
        )

# .gitignore 예시
if not os.path.exists('.gitignore'):
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(
            "# Python cache/bytecode\n"
            "__pycache__/\n*.pyc\n*.pyo\n\n"
            "# Data/results/logs\n"
            "data/\nresults/\nlogs/\n*.log\n\n"
            "# Jupyter notebook checkpoints\n"
            ".ipynb_checkpoints/\n\n"
            "# Large files\n"
            "*.zip\n*.tar.gz\n*.h5\n*.csv\n*.xlsx\n"
        )

print("필수 폴더 및 파일이 모두 정상적으로 생성/복구되었습니다!") 