import streamlit as st
import plotly.express as px
from src.data_loader import load_data
from src.analytics import expiry_monitor,consumption_summary,reorder_recommendations
st.set_page_config(page_title='Hospital Inventory Intelligence',layout='wide')
st.title('Hospital Inventory Intelligence & Waste Reduction')
st.caption('Decision-support prototype using synthetic data — not connected to a hospital.')
d=load_data(); products,batches,consumption,procurement=d.values()
expiry=expiry_monitor(batches,warning_days=90); usage=consumption_summary(consumption)
reorders=reorder_recommendations(products,batches,consumption)
a,b,c,e=st.columns(4)
a.metric('Products tracked',products.product_id.nunique())
b.metric('Units in batch snapshot',f'{int(batches.remaining_qty.sum()):,}')
c.metric('Near-expiry batches',int((expiry.expiry_status=='Near expiry').sum()))
e.metric('Products flagged to reorder',int((reorders.recommended_order_qty>0).sum()))
st.subheader('Expiry monitoring')
st.dataframe(expiry[['batch_id','product_id','remaining_qty','expiry_date','days_to_expiry','expiry_value_inr','expiry_status']].sort_values('days_to_expiry'),use_container_width=True)
st.subheader('Consumption by product')
chart=usage.merge(products[['product_id','product_name']],on='product_id')
st.plotly_chart(px.bar(chart,x='product_name',y='total_issued'),use_container_width=True)
st.subheader('Reorder recommendations')
st.dataframe(reorders[['product_id','product_name','remaining_qty','avg_daily_usage','lead_time_days','reorder_point','recommended_order_qty','reorder_status']],use_container_width=True)
procurement['actual_lead_time_days']=(procurement.delivery_date-procurement.order_date).dt.days
st.metric('Average synthetic procurement lead time',f'{procurement.actual_lead_time_days.mean():.1f} days')
