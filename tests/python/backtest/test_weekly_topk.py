from types import SimpleNamespace
import pandas as pd
import pytest
from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from backtest.strategy.weekly_topk import WeeklyTopkDropoutStrategy


class Calendar:
    def __init__(self):
        self.date=pd.Timestamp('2025-01-03')
        self.previous=pd.Timestamp('2025-01-02')
    def get_trade_step(self):return 0
    def get_step_time(self,step=None,shift=0):
        value=self.previous if shift else self.date
        return value,value


def build(monkeypatch):
    cal=Calendar()
    monkeypatch.setattr(WeeklyTopkDropoutStrategy,'trade_calendar',property(lambda _:cal))
    calls=[]
    def parent(self,execute_result=None):
        calls.append(cal.date)
        return TradeDecisionWO([],self)
    monkeypatch.setattr(TopkDropoutStrategy,'generate_trade_decision',parent)
    strategy=WeeklyTopkDropoutStrategy(signal=pd.Series(dtype=float),topk=5,n_drop=2)
    queried=[]
    def signal(**kwargs):
        queried.append(kwargs['start_time'])
        return pd.Series([1.0],index=['600000.SH'])
    strategy.signal=SimpleNamespace(get_signal=signal)
    return strategy,cal,calls,queried


def test_first_signal_and_one_review_per_week_with_holiday(monkeypatch):
    strategy,cal,calls,queried=build(monkeypatch)
    for dt,prev in [('2025-01-03','2025-01-02'),('2025-01-06','2025-01-03'),
                    ('2025-01-07','2025-01-06'),('2025-01-10','2025-01-09'),
                    ('2025-01-14','2025-01-10')]:
        cal.date=pd.Timestamp(dt);cal.previous=pd.Timestamp(prev)
        strategy.generate_trade_decision()
    assert [str(x.date()) for x in calls]==['2025-01-03','2025-01-06','2025-01-14']
    assert all(a<b for a,b in zip(queried,calls))
    assert len(strategy.get_review_events())==3


def test_missing_signal_does_not_consume_initial_review(monkeypatch):
    strategy,cal,calls,_=build(monkeypatch)
    strategy.signal=SimpleNamespace(get_signal=lambda **kwargs:None)
    strategy.generate_trade_decision()
    assert not calls and strategy._last_review_week is None


def test_reset_releases_previous_week(monkeypatch):
    strategy,cal,calls,_=build(monkeypatch)
    strategy.generate_trade_decision();strategy.reset();strategy.generate_trade_decision()
    assert len(calls)==2 and len(strategy.get_review_events())==1


@pytest.mark.parametrize('drop',[0,6,-1])
def test_invalid_drop_rejected(drop):
    with pytest.raises(ValueError,match='n_drop'):
        WeeklyTopkDropoutStrategy(topk=5,n_drop=drop)
