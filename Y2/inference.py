'''
此為 Y2/inference.py 的完整重寫版本。

以 Y1/inference.py 為藍本，進行以下修改：
1.  改為 Y2 的路徑、特徵、模型與目標變數 (DeSOx_2nd)。
2.  新增 Y3 (MLUT4_AT-240) 的計算函式與邏輯。
3.  完整實現了 predict_and_learn 的所有線上學習功能。
4.  完整實現了 generate_usage_report 的報告生成功能，並新增 Y3 趨勢圖。
'''
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import os
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.metrics import r2_score, mean_squared_error

# --- 路徑設定 (穩健版) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model_y2.json")
FEATURES_PATH = os.path.join(SCRIPT_DIR, "features2.pkl")

# --- Y3 計算函式 ---
def y2_2_y3(DeSOx_2nd, MLUT4_AIC_232B):
  """根據 Y2 的預測結果和一個額外特徵，計算 Y3 的值。"""
  return -DeSOx_2nd * MLUT4_AIC_232B + MLUT4_AIC_232B

# --- 輔助函式 (與 train.py 同步) ---
def train_model(train_X, train_y, sample_weight=None):
    model = xgb.XGBRegressor(
        n_estimators=900, random_state=42, n_jobs=-1, learning_rate=0.028,
        max_depth=8, subsample=0.75, colsample_bytree=0.75,
        objective='reg:squarederror', tree_method='hist'
    )
    return model.fit(train_X, train_y, sample_weight=sample_weight)

def apply_sequential_kalman_filter(series, transition_covariance=0.3, observation_covariance=1.0, initial_state_covariance=1.0):
    series = series.fillna(method='ffill').dropna()
    if series.empty: return pd.Series(dtype=float)
    kf = KalmanFilter(transition_matrices=1.0, observation_matrices=1.0, transition_covariance=transition_covariance, observation_covariance=observation_covariance)
    current_mean, current_cov = series.iloc[0], initial_state_covariance
    filtered_means = []
    for value in series.values:
        current_mean, current_cov = kf.filter_update(filtered_state_mean=current_mean, filtered_state_covariance=current_cov, observation=value)
        filtered_means.append(current_mean[0, 0])
    return pd.Series(filtered_means, index=series.index)

# --- 線上學習預測器 (Y2 完整版) ---
class OnlinePredictor:
    def __init__(self, time_windows_len=100, threshold_update_window=100, percentile_for_threshold=90, min_train_samples=10):
        print("初始化 OnlinePredictor (Y2 完整版)...")
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"特徵檔案不存在: {FEATURES_PATH}")
        with open(FEATURES_PATH, "rb") as f:
            training_features = pickle.load(f)
        
        self.target_col = 'DeSOx_2nd'
        self.y3_dependency_col = 'MLUT4_AIC-232B'
        self.feature_cols = [f for f in training_features if f != self.target_col] + ['prev_target']
        
        self.model = None
        if os.path.exists(MODEL_PATH):
            print(f"找到初始模型，正在載入: {MODEL_PATH}")
            self.model = xgb.XGBRegressor()
            self.model.load_model(MODEL_PATH)
        else:
            print(f"未找到初始模型，將在收到足夠數據後進行首次訓練。 ({MODEL_PATH})")
            
        self.time_windows_len = time_windows_len
        self.threshold_update_window = threshold_update_window
        self.percentile_for_threshold = percentile_for_threshold
        self.min_train_samples = min_train_samples
        self.recent_errors = []
        self.threshold = np.inf
        self.data_buffer = pd.DataFrame()
        self.history = []
        self.step_counter = 0
        self.online_learning_started = False
        
        print("正在初始化持久化的卡爾曼濾波器狀態...")
        self.kf = KalmanFilter(transition_matrices=1.0, observation_matrices=1.0, transition_covariance=0.3, observation_covariance=1.0)
        self.kf_mean = np.array([[0.85]])
        self.kf_cov = np.array([[1.0]])

    def predict_and_learn(self, current_features_df, last_true_target=None):
        self.step_counter += 1
        log_entry = {'step': self.step_counter, 'rebuild_triggered': False}
        print(f"\n--- [Y2 步驟 {self.step_counter}] 開始新一輪預測與學習循環 ---")
        error_exceeded_threshold = False
        log_entry['last_true_target'] = last_true_target

        if last_true_target is not None:
            if not self.online_learning_started:
                print("*** 收到第一筆真實數據，觸發熱啟動！拋棄歷史模型和數據。 ***")
                self.model = None
                self.data_buffer = pd.DataFrame()
                self.recent_errors = []
                self.online_learning_started = True
                self.kf_mean = np.array([[last_true_target]])
                self.kf_cov = np.array([[1.0]])

            self.kf_mean, self.kf_cov = self.kf.filter_update(filtered_state_mean=self.kf_mean, filtered_state_covariance=self.kf_cov, observation=last_true_target)
            print(f"收到上一步真實值 {last_true_target:.4f}，更新KF狀態，新平均值: {self.kf_mean[0, 0]:.4f}")

            if not self.data_buffer.empty:
                last_step_data = self.data_buffer.iloc[[-1]]
                if 'prediction_y2' in last_step_data and not pd.isna(last_step_data['prediction_y2'].iloc[0]):
                    last_prediction = last_step_data['prediction_y2'].iloc[0]
                    error = abs(last_prediction - last_true_target)
                    self.recent_errors.append(error)
                    log_entry['error'] = error
                    
                    if len(self.recent_errors) >= self.threshold_update_window:
                        self.threshold = np.percentile(self.recent_errors[-self.threshold_update_window:], self.percentile_for_threshold)
                    else:
                        self.threshold = np.percentile(self.recent_errors, self.percentile_for_threshold)
                    log_entry['threshold'] = self.threshold

                    if error > self.threshold:
                        error_exceeded_threshold = True
                        log_entry['rebuild_triggered'] = True
                        print("*** 誤差超過門檻，觸發模型再訓練！ ***")
                self.data_buffer.at[last_step_data.index[0], self.target_col] = last_true_target

        if self.model is None or error_exceeded_threshold:
            if self.model is None: print("模型不存在或被重置，嘗試進行首次線上訓練...")
            if not self.data_buffer.empty and self.target_col in self.data_buffer.columns:
                train_ready_df = self.data_buffer.dropna(subset=[self.target_col])
                if len(train_ready_df) >= self.min_train_samples:
                    print(f"數據量達標(>{self.min_train_samples})，使用最近 {len(train_ready_df)} 筆資料進行模型訓練...")
                    train_df = train_ready_df.copy()
                    train_df["prev_target"] = apply_sequential_kalman_filter(train_df[self.target_col].shift(1))
                    train_df[self.target_col] = apply_sequential_kalman_filter(train_df[self.target_col])
                    train_df = train_df.dropna(subset=self.feature_cols)
                    train_X = train_df[self.feature_cols]
                    train_y = train_df[self.target_col]
                    sample_weights = np.linspace(0.1, 1.0, num=len(train_X))
                    self.model = train_model(train_X, train_y, sample_weight=sample_weights)
                    print("模型訓練完成！已更新模型。")
                    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
                    self.model.save_model(MODEL_PATH)
                    print(f"新模型已儲存至: {MODEL_PATH}")
                    self.recent_errors = []
                    self.threshold = np.inf
                else:
                    print(f"數據不足 ({len(train_ready_df)}/{self.min_train_samples})，跳過訓練。")
            else:
                print(f"數據不足 (0/{self.min_train_samples})，跳過訓練。")

        prediction_y2 = None
        prediction_y3 = None
        if self.model is not None:
            print("開始為當前資料進行預測...")
            predict_input_df = current_features_df.copy()
            current_kf_mean_float = self.kf_mean[0, 0]
            predict_input_df['prev_target'] = current_kf_mean_float
            predict_X = predict_input_df[self.feature_cols]
            log_entry['features_used'] = predict_X.to_dict('records')[0]
            prediction_y2 = self.model.predict(predict_X)[0]
            print(f"當前 Y2 預測值 (DeSOx_2nd): {prediction_y2:.4f}")

            if self.y3_dependency_col in predict_input_df.columns:
                y3_dependency_val = predict_input_df[self.y3_dependency_col].iloc[0]
                prediction_y3 = y2_2_y3(prediction_y2, y3_dependency_val)
                print(f"當前 Y3 反推值 (MLUT4_AT-240): {prediction_y3:.4f}")
            else:
                print(f"警告: 找不到計算 Y3 所需的欄位 '{self.y3_dependency_col}'。 সন")
        else:
            print("模型尚未訓練或正在暖機，無法進行預測。")

        log_entry['prediction_y2'] = prediction_y2
        log_entry['prediction_y3'] = prediction_y3
        new_row = current_features_df.copy()
        new_row['prediction_y2'] = prediction_y2
        new_row['prediction_y3'] = prediction_y3
        new_row[self.target_col] = np.nan
        self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
        if len(self.data_buffer) > self.time_windows_len:
            self.data_buffer = self.data_buffer.iloc[-self.time_windows_len:]
        print(f"資料緩衝區大小: {len(self.data_buffer)}/{self.time_windows_len}")
        self.history.append(log_entry)
        return {
            'DeSOx_2nd_pred': float(prediction_y2) if prediction_y2 is not None else None,
            'Y3_pred': float(prediction_y3) if prediction_y3 is not None else None
        }

    def generate_usage_report(self, report_path=os.path.join(SCRIPT_DIR, "usage_report_y2.html")):
        print(f"\n正在生成 Y2 使用報告至 {report_path}...")
        if not self.history:
            print("沒有歷史紀錄可供生成報告。\n")
            return

        history_df = pd.DataFrame(self.history)
        history_df['actual_target'] = history_df['last_true_target'].shift(-1)

        metrics_calc_df = history_df.dropna(subset=['prediction_y2', 'actual_target'])
        metrics_html = "<p>沒有足夠的數據來計算指標。</p>"
        if len(metrics_calc_df) > 1:
            y_true = metrics_calc_df['actual_target']
            y_pred = metrics_calc_df['prediction_y2']
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true[y_true != 0])) * 100 if np.any(y_true != 0) else 0
            metrics = {'Metric': ['R-squared', 'RMSE', 'MAPE'], 'Value': [r2, rmse, mape], 'Unit': ['', '', '%']}
            df_metrics = pd.DataFrame(metrics)
            metrics_html = df_metrics.to_html(index=False, classes='table')

        plt.style.use('seaborn-v0_8-whitegrid')
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(history_df['step'], history_df['actual_target'], label='Y2 Actual (DeSOx_2nd)', marker='o', linestyle='-')
        ax1.plot(history_df['step'], history_df['prediction_y2'].astype(float), label='Y2 Predicted', marker='x', linestyle='--')
        ax1.set_title('Y2 Online Learning: Actual vs. Predicted Values')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel(self.target_col)
        ax1.legend()
        buf1 = BytesIO()
        fig1.savefig(buf1, format='png')
        image_base64_1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(history_df['step'], history_df['error'], label='Prediction Error', marker='o', linestyle='-')
        ax2.plot(history_df['step'], history_df['threshold'], label='Rebuild Threshold', color='red', linestyle='--')
        ax2.set_title('Prediction Error vs. Rebuild Threshold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Absolute Error')
        ax2.legend()
        buf2 = BytesIO()
        fig2.savefig(buf2, format='png')
        image_base64_2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(history_df['step'], history_df['prediction_y3'].astype(float), label='Calculated Y3 (MLUT4_AT-240)', marker='.', linestyle='-', color='green')
        ax3.set_title('Calculated Y3 Value Over Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Y3 Value')
        ax3.legend()
        buf3 = BytesIO()
        fig3.savefig(buf3, format='png')
        image_base64_3 = base64.b64encode(buf3.getvalue()).decode('utf-8')
        plt.close(fig3)

        total_rebuilds = history_df['rebuild_triggered'].sum()
        summary_text = f"在 {len(history_df)} 個模擬步驟中，總共觸發了 {total_rebuilds} 次模型重建。"

        html_content = f"""
        <!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Y2 線上學習使用報告</title>
        <style>body{{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:20px}}h1,h2{{color:#333}}table{{border-collapse:collapse;width:100%;margin-top:20px;font-size:.9em}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background-color:#f2f2f2}}.container{{max-width:1200px;margin:auto}}.summary{{background-color:#eef;padding:15px;border-left:5px solid #66f;margin-top:20px}}img{{max-width:100%;height:auto;margin-top:20px;border:1px solid #ccc}}</style>
        </head><body><div class="container"><h1>Y2 線上學習使用報告</h1>
        <h2>Y2 預測評估指標 ({self.target_col})</h2>{metrics_html}
        <h2>運行總結</h2><div class="summary"><p>{summary_text}</p></div>
        <h2>Y2 實際值 vs. 預測值</h2><img src="data:image/png;base64,{image_base64_1}" alt="Y2 Actual vs. Predicted Plot">
        <h2>Y2 預測誤差 vs. 重建門檻</h2><img src="data:image/png;base64,{image_base64_2}" alt="Error vs. Threshold Plot">
        <h2>Y3 反推值 (MLUT4_AT-240) 趨勢</h2><img src="data:image/png;base64,{image_base64_3}" alt="Calculated Y3 Plot">
        <h2>詳細歷史紀錄</h2>{history_df.to_html(index=False,classes='table',float_format='%.4f')}</div></body></html>
        """
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"報告已成功儲存至: {report_path}")