import numpy as np
import pandas as pd
import pytest

from backtest.market_layer import MarketLayer
from backtest.quote_contract import TushareQfqExchange


def make_exchange(monkeypatch, fee_model=None):
    from qlib.data import D
    from qlib.config import C
    monkeypatch.setitem(C._config, 'trade_unit', 100)
    index = pd.MultiIndex.from_product([['A','B','C','D','E'],pd.to_datetime(['2026-02-04'])],names=['instrument','datetime'])
    frame = pd.DataFrame({'$close':[90,11,12,10,9], '$close_qfq':[80,10,11,np.nan,8],
        '$change':[3.63,.10,-1.85,.01,.01], '$pct_chg':[4.1883,10.0,-10.0,1.0,np.nan],
        '$factor':[np.nan]*5, '$volume':[100000]*5},index=index)
    monkeypatch.setattr(D, 'features', lambda codes, fields, *args, **kwargs: frame[fields].copy(), raising=False)
    costs = {'open_cost':.0035,'close_cost':.0045,'min_cost':5} if fee_model is None else {'fee_model':fee_model}
    exchange=TushareQfqExchange(codes=['A','B','C','D','E'],start_time='2026-02-04',end_time='2026-02-04',
        limit_threshold=.095, **costs)
    return exchange,frame


def test_limits_use_percentage_and_unknown_quotes_fail_closed(monkeypatch):
    exchange,source=make_exchange(monkeypatch)
    date=pd.Timestamp('2026-02-04')
    assert exchange.is_stock_tradable('A',date,date)
    assert not exchange.is_stock_tradable('B',date,date)
    assert not exchange.is_stock_tradable('C',date,date)
    assert not exchange.is_stock_tradable('D',date,date)
    assert not exchange.is_stock_tradable('E',date,date)
    assert exchange.quote_df.loc[('A',date),'$change']==pytest.approx(.041883)
    assert source.loc[('A',date),'$change']==3.63


def test_buying_at_close_has_no_artificial_marking_profit(monkeypatch):
    exchange,source=make_exchange(monkeypatch)
    date=pd.Timestamp('2026-02-04')
    deal=exchange.get_deal_price('A',date,date,direction=1)
    mark=exchange.get_close('A',date,date)
    assert deal==mark==80
    assert source.loc[('A',date),'$close']==90
    quantity=57000/deal
    assert quantity*mark-57000==pytest.approx(0)


def test_market_contract_keeps_dates_and_costs():
    config={'quote_contract':'tushare_qfq_v1','exchange_kwargs':{'deal_price':'$close_qfq','open_cost':.0035,'close_cost':.0045,'limit_threshold':.095}}
    built=MarketLayer().build(config,'1day',start_time='2025-01-01',end_time='2026-10-15')
    assert built['exchange']['class']=='TushareQfqExchange'
    assert built['exchange']['kwargs']['end_time']=='2026-10-15'
    assert built['exchange']['kwargs']['open_cost']==.0035
    assert 'start_time' not in config['exchange_kwargs']
    with pytest.raises(ValueError,match='explicit backtest dates'):
        MarketLayer().build(config,'1day')


def test_incompatible_execution_price_rejected():
    with pytest.raises(ValueError,match='daily'):
        TushareQfqExchange(deal_price='$close')


@pytest.mark.parametrize('value, sell, expected', [(60000,False,81.6),(60000,True,111.6),
                                                  (1000,False,6.01),(1000,True,6.51),(0,True,0)])
def test_explicit_fee_formula(value,sell,expected):
    from backtest.fees import AShareFees
    assert AShareFees().total(value,sell)==pytest.approx(expected)


@pytest.mark.parametrize('cash',[4.0,5.0,1006.01,20000,60081.6])
def test_fee_floor_cash_sizing(monkeypatch,cash):
    from backtest.fees import AShareFees
    from qlib.backtest.decision import Order
    from qlib.backtest.position import Position
    from collections import defaultdict
    fees=AShareFees()
    exchange,_=make_exchange(monkeypatch,fees.__dict__)
    dt=pd.Timestamp('2026-02-04')
    order=Order(stock_id='A',amount=100000,start_time=dt,end_time=dt,direction=Order.BUY)
    price,value,cost=exchange._calc_trade_info_by_order(order,Position(cash=cash),defaultdict(float))
    assert value==pytest.approx(fees.affordable_buy_value(cash))
    assert cost==pytest.approx(fees.total(value))
    assert value+cost<=cash+1e-8
    if cash>5:
        assert value+cost==pytest.approx(cash)
    else:
        assert cost==0


def test_selling_charges_tax_and_separate_minimum(monkeypatch):
    from backtest.fees import AShareFees
    from qlib.backtest.decision import Order
    from qlib.backtest.position import Position
    from collections import defaultdict
    exchange,_=make_exchange(monkeypatch,AShareFees().__dict__)
    dt=pd.Timestamp('2026-02-04')
    position=Position(cash=0,position_dict={'A':{'amount':12.5,'price':80}})
    order=Order(stock_id='A',amount=12.5,start_time=dt,end_time=dt,direction=Order.SELL)
    price,value,cost=exchange._calc_trade_info_by_order(order,position,defaultdict(float))
    assert value==1000
    assert cost==pytest.approx(6.51)


def test_explicit_fees_reject_duplicate_legacy_costs():
    with pytest.raises(ValueError,match='mix explicit'):
        MarketLayer().build({'quote_contract':'tushare_qfq_v1','fee_model':{},
                             'exchange_kwargs':{'open_cost':.0035}},'1day',start_time='2025-01-01',end_time='2026-10-15')


def test_cost_components_reject_wrong_deductions():
    from backtest.fees import AShareFees
    trades=pd.DataFrame({'value':[1000,60000],'direction':['BUY','SELL'],'cost':[6.01,111.6]})
    result=AShareFees().annotate(trades)
    assert result.commission.tolist()==[5,21]
    assert result.stamp_tax.tolist()==[0,30]
    trades.loc[0,'cost']=5
    with pytest.raises(ValueError,match='reconcile'):
        AShareFees().annotate(trades)
