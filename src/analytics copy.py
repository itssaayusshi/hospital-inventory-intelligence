import pandas as pd
def expiry_monitor(batches, as_of=None, warning_days=90):
    as_of=pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    x=batches[batches.remaining_qty>0].copy()   # a used-up batch cannot expire; ignore it
    x['days_to_expiry']=(x['expiry_date']-as_of).dt.days
    x['expiry_value_inr']=x['remaining_qty']*x['unit_cost_inr']
    x['expiry_status']='OK'
    x.loc[x.days_to_expiry<0,'expiry_status']='Expired'
    x.loc[x.days_to_expiry.between(0,warning_days),'expiry_status']='Near expiry'
    return x
def consumption_summary(consumption):
    x=consumption.groupby('product_id',as_index=False).agg(total_issued=('quantity_issued','sum'),active_days=('transaction_date','nunique'))
    x['avg_daily_usage']=x.total_issued/x.active_days.clip(lower=1)
    return x
def reorder_recommendations(products,batches,consumption,safety_days=3):
    usage=consumption_summary(consumption)
    stock=batches.groupby('product_id',as_index=False).remaining_qty.sum()
    x=products.merge(usage,on='product_id',how='left').merge(stock,on='product_id',how='left')
    x[['avg_daily_usage','remaining_qty']]=x[['avg_daily_usage','remaining_qty']].fillna(0)
    x['safety_stock']=x.avg_daily_usage*safety_days
    x['reorder_point']=x.avg_daily_usage*x.lead_time_days+x.safety_stock
    x['recommended_order_qty']=(x.reorder_point-x.remaining_qty).clip(lower=0).round().astype(int)
    x['reorder_status']=x.recommended_order_qty.map(lambda n:'Reorder' if n>0 else 'Sufficient')
    return x
