# このリポジトリは何？
修論の研究で使ったコードをおいている場所です．記録及び希望者向けの再現用です．

# 使い方

1. 必要なライブラリをインストールする
   ```
   pip install -r requirement.txt
   ```
2. 話者認識器を訓練する
   ```
   bash run_speaker.sh <split_id> (1~5の数字)
   ```
   `split_id`は5-fold交差検証の分割の番号です．

3. 感情認識器と特徴量抽出器を訓練する
   ```
   bash run_emotion.sh <split_id> 
   ```   
4. 結果を確認する
   
   デフォルトでは`outputs/emotion_<split_id>/results_<split_id>.txt`に
   - 検証データにおけるメトリクス
   - テストデータにおけるメトリクス
   - 検証正答率とテスト正答率の差
   - 検証/テストデータ全体における正答率

   が記録されるはずです．

