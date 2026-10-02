import pandas as pd, numpy as np
df=pd.read_parquet('audit.parquet')
LOCAL=0  # offset of local time from UTC, seconds
df['day']=pd.to_datetime(df.t+LOCAL,unit='s').dt.date
df['hr']=((df.t+LOCAL)//3600).astype(int)
print("t range",pd.to_datetime(df.t.min()+LOCAL,unit='s'),pd.to_datetime(df.t.max()+LOCAL,unit='s'))
tab=df.pivot_table(index='day',columns='ev',values='t',aggfunc='size',observed=True).fillna(0).astype(int)
hrs=df.groupby('day').hr.nunique(); tab['hours_with_events']=hrs
print(tab.to_string())
# generation present? check key reuse: keys with >1 CREATE
c=df[df.ev=='CREATE'].key.value_counts()
print("keys with >1 CREATE:",(c>1).sum(),"of",len(c))
u=df[df.ev=='UNLINK'].key.value_counts(); print("keys with >1 UNLINK:",(u>1).sum(),"of",len(u))
print("nlink at UNLINK:",df[df.ev=='UNLINK'].nlink.value_counts().head())
print("size at UNLINK quantiles:",df[df.ev=='UNLINK']['size'].quantile([.5,.9,.99,1]).tolist(), "total TB", df[df.ev=='UNLINK']['size'].sum()/1e12)
w=df[df.ev=='WCLOSE'].drop_duplicates('key',keep='last'); print("WCLOSE distinct keys",len(w),"bytes TB",w['size'].sum()/1e12)
