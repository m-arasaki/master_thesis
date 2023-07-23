import pandas as pd
import glob

def filename_to_spk(filename: str) -> str:
    parts = filename.split('_')
    return parts[0] + parts[-1][0]

def edit_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    # filename を抽出
    filenames = []
    for f in df['file']:
        filenames.append(f.split('/')[-1])
    
    # speaker を抽出
    spk = []
    for f in filenames:
        spk.append(filename_to_spk(f))

    # dataframe に insert
    spk_series = pd.Series(data=spk)
    df.insert(loc=2, column='speaker', value=spk_series)

    # 編集したcsvで元のデータを上書き
    df.to_csv(csv_path)

def main():
    for path in glob.glob('*.csv', recursive=False):
        edit_csv(path)

if __name__ == "__main__":
    main()
