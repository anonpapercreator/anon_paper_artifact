import pandas as pd, numpy as np
df=pd.read_parquet('audit.parquet',columns=['t','ev','key','size','nlink'])
df['ev']=df.ev.astype(str)
LOCAL=0  # offset of local time from UTC, seconds
# hourly counts at the edges to fix the fully covered window
h=((df.t)//3600).astype(int); hc=h.value_counts().sort_index()
T0=pd.Timestamp('2026-08-28').timestamp(); T1=pd.Timestamp('2026-09-27').timestamp()  # UTC day dirs
print("events before T0:",(df.t<T0).sum()," after T1:",(df.t>=T1).sum())
print("hourly counts first 16h after T0-12h:", hc.loc[int(T0//3600)-12:int(T0//3600)+4].tolist())
print("hourly counts around T1:", hc.loc[int(T1//3600)-4:int(T1//3600)+10].tolist())
