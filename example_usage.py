import pandas as pd
import time
from inference import OnlinePredictor
import pickle
import warnings 
warnings.filterwarnings('ignore')
print("--- 線上學習預測器 (OnlinePredictor) 使用範例 ---")

# --- 1. 準備模擬的即時數據流 ---
# 我們從原始 feather 檔案中讀取一段數據，來模擬一個持續進來的數據流。
print("\n步驟 1: 載入並準備模擬數據流...")
try:
    full_df = pd.read_feather('test0609-CFB2脫硫劑優化改善.feather').dropna().tail(1000)
    # 為了這個範例，我們只取特徵欄位和目標欄位
    with open("features1.pkl", "rb") as f:
        features_to_keep = pickle.load(f)
    simulation_df = full_df[features_to_keep + ['DeSOx_1st']]
    print(f"已載入 {len(simulation_df)} 筆數據用於模擬。")

    # 決定模擬的總步數
    num_simulation_steps = len(simulation_df)
    print(f"將進行 {num_simulation_steps} 個時間步驟的模擬。")
    if num_simulation_steps < 110:
        print("警告: 模擬數據不足110筆，可能無法完整觀察到熱啟動->暖機->新模型訓練的完整過程。")
except FileNotFoundError:
    print("錯誤: 找不到 test0609-CFB2脫硫劑優化改善.feather 或 features1.pkl。請先執行 train.py。")
    exit()

# --- 2. 初始化 OnlinePredictor ---
# 在你的應用程式啟動時，只需要初始化一次。
# 它會自動載入現有模型，並在記憶體中維持自己的狀態。
print("\n步驟 2: 初始化 OnlinePredictor...")
predictor = OnlinePredictor()

# --- 3. 模擬滾動預測的迴圈 ---
# 這模擬了你的系統每 5 秒接收一筆新數據，並進行預測和學習。
print("\n步驟 3: 開始模擬滾動預測迴圈 (每 5 秒一次)...")

last_true_target = None

for i in range(num_simulation_steps):

    print(f"\n================== 時間點 {i} ==================")
    # a. 從數據流中取得當前的感測器數據
    current_features_df = simulation_df.iloc[[i]].drop(columns=['DeSOx_1st'])
    
    # b. 呼叫 predict_and_learn
    #    - 傳入當前的特徵
    #    - 傳入「上一個時間點」的真實答案 (last_true_target)
    prediction = predictor.predict_and_learn(current_features_df, last_true_target)

    if prediction is not None:
        print(f"---> 時間點 {i} 的預測結果為: {prediction:.4f}")
    else:
        print(f"---> 時間點 {i} 無法預測 (模型可能尚未訓練)")

    # c. 準備下一次迴圈的資料
    #    在真實世界中，這是在下一個 5 秒發生前，你從系統中拿到的真實回饋
    last_true_target = simulation_df['DeSOx_1st'].iloc[i]
    print(f"(真實世界中，我們在稍後會得知時間點 {i} 的真實答案是: {last_true_target:.4f}) - 這將在下一步提供給模型學習")

    # d. 模擬 5 秒的間隔
    print("等待 5 秒...")
    time.sleep(0.01)

print("\n================== 模擬結束 ==================")

# --- 4. 生成報告與偵錯檔案 ---
# a. 生成視覺化HTML報告
predictor.generate_usage_report()

# b. 生成詳細的CSV偵錯檔案
DEBUG_FILE = "inference_debug_data.csv"
print(f"[DEBUG] 將詳細的線上學習歷史紀錄儲存至 {DEBUG_FILE}...")
history_df = pd.DataFrame(predictor.history)

# 將 'features_used' 欄位中的字典展開成新的欄位
features_df = history_df['features_used'].apply(pd.Series)
features_df = features_df.add_prefix('feature_') # 為特徵欄位加上前綴以避免衝突

# 合併原始歷史紀錄和展開的特徵欄位
final_history_df = pd.concat([history_df.drop('features_used', axis=1), features_df], axis=1)

final_history_df.to_csv(DEBUG_FILE, index=False)
print(f"[DEBUG] 偵錯檔案已成功儲存。")
