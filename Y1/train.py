import pandas as pd
import xgboost as xgb
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import pickle
from pykalman import KalmanFilter
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# --- Data Loading and Preprocessing ---
df = pd.read_feather('test0609-CFB2脫硫劑優化改善.feather').dropna().tail(10000)
print(f'df讀取完成shape:{df.shape}')
df.index.name = 'datetime'

# select by rule
coal_low = df[df["MLUT4_FIQ-2BTCF"] < 20]
sox = df["MLUT4_AT-240"]
constant_sox_indices = sox[sox.shift(1) == sox][(sox.shift(2) == sox) & (sox.shift(-1) == sox)].index
constant_sox = df.loc[constant_sox_indices]
common_index = coal_low.index.union(constant_sox.index)
select_df = df.loc[~df.index.isin(common_index), :]

# select features
with open("features1.pkl", "rb") as f:
    features = pickle.load(f)

# define y_col
y_col = 'DeSOx_1st'
select_df = select_df[features+[y_col]]

# --- Modeling Code ---
def train_model(train_X, train_y, sample_weight=None):
    model = xgb.XGBRegressor(
        n_estimators=900,
        random_state=42,
        n_jobs=-1,
        learning_rate=0.028,
        max_depth=8,
        subsample=0.75,
        colsample_bytree=0.75,
        objective='reg:squarederror',
        tree_method='hist'
    )
    return model.fit(train_X, train_y, sample_weight=sample_weight)

# 1. 時間排序
select_df = select_df.sort_index().reset_index(drop=True)

# 2. 設定
target_col = y_col
time_windows_len = 100

# 加入前一期目標欄位當特徵
select_df["prev_target"] = select_df[target_col].shift(1)

# 效率不應該低於0
select_df = select_df[select_df["prev_target"]>0]
select_df = select_df[select_df[y_col]>0]

def apply_sequential_kalman_filter(series, transition_covariance=0.3, observation_covariance=1.0, initial_state_covariance=1.0):
    """
    以序列方式、一步步地應用卡爾曼濾波，模擬真實線上情境。
    這可以確保訓練和推論之間的一致性。
    """
    series = series.fillna(method='ffill').dropna()
    if series.empty:
        return pd.Series(dtype=float)

    kf = KalmanFilter(
        transition_matrices=1.0,
        observation_matrices=1.0,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance
    )

    # 初始化狀態
    current_mean = series.iloc[0]
    current_cov = initial_state_covariance

    filtered_means = []
    for value in series.values:
        current_mean, current_cov = kf.filter_update(
            filtered_state_mean=current_mean,
            filtered_state_covariance=current_cov,
            observation=value
        )
        filtered_means.append(current_mean[0, 0])
    
    return pd.Series(filtered_means, index=series.index)

# 套用序列化的卡爾曼濾波，以模擬真實線上情境
print("正在以序列化方式應用卡爾曼濾波，這可能會需要一些時間...")
select_df["prev_target"] = apply_sequential_kalman_filter(select_df["prev_target"])
select_df[y_col] = apply_sequential_kalman_filter(select_df[y_col])

# 刪除因 shift 產生的 NaN 資料
select_df = select_df.dropna().reset_index(drop=True)

# 定義特徵欄位
feature_cols = [col for col in select_df.columns if col not in ['timestamp', target_col]]

# --- DEBUG: 儲存用於訓練的資料 --- 
DEBUG_FILE = "train_debug_data.csv"
print(f"[DEBUG] 將前處理後的訓練資料儲存至 {DEBUG_FILE}...")
select_df.to_csv(DEBUG_FILE, index=False)


# 開始訓練的索引
start_idx = time_windows_len

# 計算 total_steps
total_steps = len(select_df) - start_idx

# 3. 儲存預測結果
predictions = []
abs_errors = []
thresholds = []
indices = []

current_model_recent_errors = []
threshold_update_window = 100
percentile_for_threshold = 90

# 4. 初始化進度條與模型重建計數器
pbar = tqdm(total=total_steps, desc="動態預測進度")
rebuild_count = 0 # <-- 新增計數器
current_model = None

# 5. 動態預測迴圈
i = start_idx
error_exceeded_threshold = True

while i < len(select_df):
    if pbar.n >= 1000:
        print("達到預設的最大步數，停止迴圈。")
        break

    # 判斷是否需要建模
    if current_model is None or error_exceeded_threshold:
        rebuild_count += 1 # <-- 每次重建，計數器加1
        
        train_df = select_df.iloc[i - time_windows_len:i]
        train_X = train_df[feature_cols]
        train_y = train_df[target_col]

        # 根據你的要求，為樣本添加權重，越新的樣本權重越高
        sample_weights = np.linspace(0.1, 1.0, num=len(train_X))

        # train model
        current_model = train_model(train_X, train_y, sample_weight=sample_weights)
        current_model_recent_errors = []
        current_threshold = np.inf
        error_exceeded_threshold = False

    # 預測
    test_row = select_df.iloc[i]
    test_X = test_row[feature_cols].values.reshape(1, -1)
    true_y = test_row[target_col]
    pred_y = current_model.predict(test_X)[0]
    error = abs(pred_y - true_y)
    current_model_recent_errors.append(error)

    if len(current_model_recent_errors) >= threshold_update_window:
        current_threshold = np.percentile(current_model_recent_errors[-threshold_update_window:], percentile_for_threshold)
    elif len(current_model_recent_errors) > 0:
        current_threshold = np.percentile(current_model_recent_errors, percentile_for_threshold)
    else:
        current_threshold = np.inf

    # 紀錄結果
    predictions.append(pred_y)
    abs_errors.append(error)
    thresholds.append(current_threshold)
    indices.append(i)

    # 更新進度條
    pbar.update(1)

    # 判斷是否需要下次重建
    if error > current_threshold:
        error_exceeded_threshold = True
    else:
        error_exceeded_threshold = False

    i += 1

# 結束訓練
pbar.close()
print("\n預測迴圈結束。")

# --- 6. 輸出結果與指標評估 ---
result_df = select_df.loc[indices].copy()
result_df['prediction'] = predictions
result_df['abs_error'] = abs_errors
result_df['threshold'] = thresholds

y_true = result_df[y_col]
y_pred = result_df['prediction']

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true[y_true != 0])) * 100 if np.any(y_true != 0) else 0

metrics = {
    'Metric': ['R-squared', 'RMSE', 'MAPE'],
    'Value': [r2, rmse, mape],
    'Unit': ['', '', '%']
}
df_metrics = pd.DataFrame(metrics)

# --- 7. 保存模型 ---
MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if current_model:
    model_path = os.path.join(MODEL_DIR, "xgb_model.json")
    current_model.save_model(model_path)
    print(f"\n模型已儲存至 {model_path}")
else:
    print("\n沒有模型被訓練，無法儲存。")

# --- 8. 生成HTML報告 ---
print("正在生成HTML報告...")

# a. 生成圖表並轉換為Base64
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_true.values, label='Actual Values', color='blue', alpha=0.7)
ax.plot(y_pred.values, label='Predicted Values', color='red', linestyle='--')
ax.set_title('Actual vs. Predicted Values')
ax.set_xlabel('Time Steps')
ax.set_ylabel('DeSOx_1st')
ax.legend()
ax.grid(True)

# 將圖表保存到記憶體中
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image_base64 = base64.b64encode(buf.read()).decode('utf-8')
buf.close()
plt.close(fig)

# b. 總結模型重建頻率
if total_steps > 0:
    initial_builds = 1  # The first one is always a build
    rebuilds_after_initial = rebuild_count - initial_builds

    # Number of opportunities for a rebuild
    opportunities_for_rebuild = total_steps - 1

    if opportunities_for_rebuild > 0:
        rebuild_frequency = (rebuilds_after_initial / opportunities_for_rebuild) * 100
        rebuild_summary = (f"總共執行 {total_steps} 個預測步驟。"
                         f"除了第1次的必要建置，後續在 {opportunities_for_rebuild} 次預測中，額外觸發了 {rebuilds_after_initial} 次模型重建。"
                         f"模型「再」建置頻率約為 {rebuild_frequency:.2f}%。")
    else:
        # This happens if total_steps is 1
        rebuild_summary = f"總共執行 1 個預測步驟，進行了 1 次初始模型建置。"
else:
    rebuild_summary = "沒有足夠的資料進行預測。"

# c. 組合HTML內容
html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>模型訓練報告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 60%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .container {{ max-width: 1200px; margin: auto; }}
        .summary {{ background-color: #eef; padding: 15px; border-left: 5px solid #66f; margin-top: 20px;}}
        img {{ max-width: 100%; height: auto; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>模型訓練報告</h1>
        
        <h2>模型評估指標</h2>
        {df_metrics.to_html(index=False, classes='table')}
        
        <h2>模型重建總結</h2>
        <div class="summary">
            <p>{rebuild_summary}</p>
        </div>
        
        <h2>實際值 vs. 預測值</h2>
        <img src="data:image/png;base64,{image_base64}" alt="Actual vs. Predicted Values Plot">
        
        <h2>預測結果預覽 (前10筆)</h2>
        {result_df.head(10).to_html(index=False, classes='table')}
        
        <h2>詳細預測數據</h2>
        {result_df.to_html(index=False, classes='table')}
    </div>
</body>
</html>
"""

# d. 寫入HTML文件
report_path = "training_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"報告已儲存至 {report_path}")
