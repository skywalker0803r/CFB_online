'''
此腳本為 Y2 的線上學習模擬範例 (重構版)。

功能：
1.  使用 Y2/inference.py 中的 OnlinePredictor。
2.  模擬即時數據流入，進行線上學習。
3.  在每個時間點上印出 Y2 (DeSOx_2nd) 的預測值與 Y3 (MLUT4_AT-240) 的反推值。
4.  採用穩健的路徑處理，與執行位置無關。
'''
import pandas as pd
import time
import pickle
import warnings
import os
from collections import OrderedDict

# 由於此腳本和 inference.py 在同一資料夾，可以直接匯入
from inference import OnlinePredictor

warnings.filterwarnings('ignore')
print("--- Y2 線上學習預測器 (OnlinePredictor) 使用範例 (v2) ---")

# --- 1. 路徑與常數設定 (穩健版) ---
print("\n步驟 1: 設定路徑與常數...")

# __file__ 是本腳本的絕對路徑
# SCRIPT_DIR 就是 Y2 資料夾的路徑
SCRIPT_DIR = "./"#os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT 是 Y2 的上一層，也就是專案根目錄
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

FEATURES_PATH = os.path.join(SCRIPT_DIR, "features2.pkl")
DATA_SOURCE = os.path.join(PROJECT_ROOT, 'test0609-CFB2脫硫劑優化改善.feather')
TARGET_COL = 'DeSOx_2nd'
Y3_DEPENDENCY_COL = 'MLUT4_AIC-232B'
DEBUG_FILE = os.path.join(SCRIPT_DIR, "inference_debug_data_y2.csv")

print(f"專案根目錄: {PROJECT_ROOT}")
print(f"Y2 資料夾: {SCRIPT_DIR}")
print(f"資料來源: {DATA_SOURCE}")

# --- 2. 準備模擬的即時數據流 ---
print("\n步驟 2: 載入並準備模擬數據流...")
try:
    full_df = pd.read_feather(DATA_SOURCE).dropna().tail(1000)
    with open(FEATURES_PATH, "rb") as f:
        features_to_keep = pickle.load(f)

    # 建立一個包含所有必需欄位的列表
    required_cols = features_to_keep + [TARGET_COL, Y3_DEPENDENCY_COL]
    # 使用 OrderedDict.fromkeys 移除重複項，同時保持原始順序
    required_cols = list(OrderedDict.fromkeys(required_cols))

    # 再次確認這些欄位真的存在於讀取的 DataFrame 中，避免 KeyError
    existing_cols = [col for col in required_cols if col in full_df.columns]
    if len(existing_cols) != len(required_cols):
        missing_cols = set(required_cols) - set(existing_cols)
        print(f"警告: 原始資料中缺少以下欄位，將被忽略: {missing_cols}")

    simulation_df = full_df[existing_cols]
    print(f"已載入 {len(simulation_df)} 筆數據用於模擬。")

    num_simulation_steps = len(simulation_df)
    print(f"將進行 {num_simulation_steps} 個時間步驟的模擬。")

except FileNotFoundError:
    print(f"錯誤: 找不到資料來源 {DATA_SOURCE} 或特徵檔案 {FEATURES_PATH}。")
    print("請確認檔案是否存在，或執行 Y2/train.py 來產生必要檔案。")
    exit()

# --- 3. 初始化 OnlinePredictor ---
print("\n步驟 3: 初始化 Y2 的 OnlinePredictor...")
predictor = OnlinePredictor()

# --- 4. 模擬滾動預測的迴圈 ---
print("\n步驟 4: 開始模擬滾動預測迴圈...")

last_true_target = None

for i in range(num_simulation_steps):
    print(f"\n================== 時間點 {i} ==================")
    # a. 從數據流中取得當前的感測器數據 (不包含Y2的答案)
    current_features_df = simulation_df.iloc[[i]].drop(columns=[TARGET_COL], errors='ignore')

    # b. 呼叫 predict_and_learn
    predictions = predictor.predict_and_learn(current_features_df, last_true_target)

    # c. 處理並印出結果
    if predictions and predictions['DeSOx_2nd_pred'] is not None:
        y2_pred = predictions['DeSOx_2nd_pred']
        y3_pred = predictions['Y3_pred']
        print(f"---> Y2 預測 (DeSOx_2nd): {y2_pred:.4f}")
        if y3_pred is not None:
            print(f"---> Y3 反推 (MLUT4_AT-240): {y3_pred:.4f}")
    else:
        print(f"---> 時間點 {i} 無法預測 (模型可能尚未訓練或正在暖機)")

    # d. 準備下一次迴圈的資料 (Y2的真實答案)
    last_true_target = simulation_df[TARGET_COL].iloc[i]
    print(f"(真實世界中，我們稍後會得知時間點 {i} 的 Y2 真實答案是: {last_true_target:.4f})")

    # 模擬時間間隔
    # time.sleep(0.01)

print("\n================== 模擬結束 ==================")

# --- 5. 生成報告與偵錯檔案 ---
print("\n步驟 5: 生成報告與偵錯檔案...")

# a. 生成視覺化HTML報告
predictor.generate_usage_report()

# b. 生成詳細的CSV偵錯檔案
print(f"[DEBUG] 將詳細的線上學習歷史紀錄儲存至 {DEBUG_FILE}...")
history_df = pd.DataFrame(predictor.history)

if not history_df.empty and 'features_used' in history_df.columns:
    # 將 'features_used' 欄位中的字典展開成新的欄位
    features_df = history_df['features_used'].apply(pd.Series)
    features_df = features_df.add_prefix('feature_') # 為特徵欄位加上前綴以避免衝突

    # 合併原始歷史紀錄和展開的特徵欄位
    final_history_df = pd.concat([history_df.drop('features_used', axis=1), features_df], axis=1)
else:
    final_history_df = history_df

final_history_df.to_csv(DEBUG_FILE, index=False)
print(f"[DEBUG] 偵錯檔案已成功儲存。")
