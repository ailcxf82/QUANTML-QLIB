"""Once-per-ISO-week TopkDropout using the previous session's saved signal.

The first usable signal opens the portfolio. Subsequent reviews occur on the
first available trading session of each ISO week. No midweek retries, stops,
or reweighting; all order generation and tradability rules remain Qlib's.
"""
import pandas as pd
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import TradeDecisionWO


class WeeklyTopkDropoutStrategy(TopkDropoutStrategy):
    def __init__(self, *, topk=5, n_drop=1, **kwargs):
        if not isinstance(topk,int) or not isinstance(n_drop,int) or not 1 <= n_drop <= topk:
            raise ValueError('Weekly strategy requires 1 <= n_drop <= topk')
        self._last_review_week=None
        self._review_events=[]
        super().__init__(topk=topk,n_drop=n_drop,**kwargs)

    def reset(self, *args, **kwargs):
        super().reset(*args,**kwargs)
        self._last_review_week=None
        self._review_events=[]

    def generate_trade_decision(self, execute_result=None):
        step=self.trade_calendar.get_trade_step()
        start,end=self.trade_calendar.get_step_time(step)
        iso=pd.Timestamp(start).isocalendar()
        week=(iso.year,iso.week)
        if week==self._last_review_week:
            return TradeDecisionWO([],self)
        pred_start,pred_end=self.trade_calendar.get_step_time(step,shift=1)
        score=self.signal.get_signal(start_time=pred_start,end_time=pred_end)
        if score is None or len(score)==0:
            return TradeDecisionWO([],self)
        decision=super().generate_trade_decision(execute_result)
        self._last_review_week=week
        self._review_events.append({'review_date':str(pd.Timestamp(start).date()),
            'signal_date':str(pd.Timestamp(pred_start).date()),'iso_year':iso.year,'iso_week':iso.week,
            'order_count':len(decision.get_decision())})
        return decision

    def get_review_events(self):
        return pd.DataFrame(self._review_events)
