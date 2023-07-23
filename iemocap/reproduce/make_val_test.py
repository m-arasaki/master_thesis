import pandas as pd
import glob

# 女性話者(F)をval, 男性話者(M)をtestに
for f in glob.glob('_iemocap_*.csv', recursive=False):
    df = pd.read_csv(f, index_col=0)

    session = int(f.split('_')[2].split('.')[0])
    print('session:' + str(session))
    df_val = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)

    for r in range(len(df)):
        if df.iloc[r]['speaker'][-1] == 'M':
            df_test = pd.concat([df_test, pd.DataFrame(df.iloc[r]).T])
        elif df.iloc[r]['speaker'][-1] == 'F':
            df_val = pd.concat([df_val, pd.DataFrame(df.iloc[r]).T])
        
    df_val.to_csv('iemocap_' + str(session) + '.val.csv')
    df_test.to_csv('iemocap_' + str(session) + '.test.csv')