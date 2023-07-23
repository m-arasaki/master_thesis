import pandas as pd
import glob

for f in glob.glob('iemocap_*.val.csv'):
    print('checking ' + f + ':')
    df = pd.read_csv(f, index_col=0)
    spk_set = sorted(set(df['speaker']))
    print(spk_set)

for f in glob.glob('iemocap_*.test.csv'):
    print('checking ' + f + ':')
    df = pd.read_csv(f, index_col=0)
    spk_set = sorted(set(df['speaker']))
    print(spk_set)

for f in glob.glob('iemocap_*.train.csv'):
    print('checking ' + f + ':')
    df = pd.read_csv(f, index_col=0)
    spk_set = sorted(set(df['speaker']))
    print(spk_set)