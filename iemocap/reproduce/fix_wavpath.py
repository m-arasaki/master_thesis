import pandas as pd
import glob 

for f in glob.glob('*.csv'):
    
    df = pd.read_csv(f)
    df['file'] = df['file'].map(lambda x: './wav_path/' + x)

    df.to_csv(f)