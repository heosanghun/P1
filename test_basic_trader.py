import pytest
from basic_trader import BasicTrader

def test_trader_init():
    trader = BasicTrader()
    assert trader is not None

def test_trader_run():
    trader = BasicTrader()
    # 예시: run()이 예외 없이 실행되는지 확인
    try:
        trader.run()
    except Exception as e:
        pytest.fail(f"run()에서 예외 발생: {e}") 