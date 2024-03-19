import pandas as pd
import numpy as np
import re
url = "https://raw.githubusercontent.com/js05212/citeulike-a/master/mult.dat"
df = pd.read_csv(url,header=None)
df = df.rename(columns = {0:'bow'})
df = df['bow'].str.split(' ')
df = pd.DataFrame(df)
df = pd.DataFrame(df.bow.values.tolist()).add_prefix('col_')
nb_vec, df = df.iloc[:,0],df.iloc[:,1:]

df_corpus = df.applymap(lambda x: float(re.split(':',x)[0]) if isinstance(x,str) else np.nan) #python 2.7 uses df.applymap, else python3 uses df.map
df_corpus_nb = df.applymap(lambda x: float(re.split(':',x)[1]) if isinstance(x,str) else np.nan)
mult_nor = pd.DataFrame(np.nan, index=list(range(df_corpus.shape[0])), columns=list(range(int(df_corpus.max().max())+1)))


for ii in range(mult_nor.shape[0]):
    ind_list = df_corpus.iloc[ii].dropna().astype(int)
    mult_nor.loc[ii,ind_list]=list(df_corpus_nb.iloc[ii].dropna().astype(int))
mult_nor = mult_nor.fillna(0)
mult_nor = mult_nor.astype(int)

max_series = mult_nor.max(axis=1)
mult_nor = np.array(mult_nor.divide(max_series, axis=0))

# Doesn't seem like it is dividing by the maximum cooccurrence across all articles (as mentioned in https://github.com/eelxpeng/CollaborativeVAE/issues/4) but maximum cooccurrence in the article itself
# https://datascience.stackexchange.com/questions/26653/cant-interpret-the-text-information-and-ratings-matrix-imported-to-nn