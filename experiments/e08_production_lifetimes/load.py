import pandas as pd, glob, numpy as np
cols="t ev key size nlink ext prefix node excl atime mtime".split()
fs=sorted(glob.glob('audit/*.tsv.gz'))
df=pd.concat([pd.read_csv(f,sep='\t',names=cols,header=None,dtype={'key':str,'prefix':str,'ext':str,'node':str},
      na_values=['','-'],keep_default_na=False) for f in fs],ignore_index=True)
df['ev']=df.ev.astype('category')
df.to_parquet('audit.parquet')
print(df.shape); print(df.dtypes); print(df.node.value_counts()); print(df.excl.value_counts())
