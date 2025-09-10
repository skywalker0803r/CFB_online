
'''
此為 Y2/inference.py 的完整重寫版本 (v2)。

以 Y1/inference.py 為藍本，進行以下修改：
1.  改為 Y2 的路徑、特徵、模型與目標變數 (DeSOx_2nd)。
2.  新增 Y3 (MLUT4_AT-240) 的計算函式與邏輯。
3.  完整實現了 predict_and_learn 的所有線上學習功能。
4.  完整實現了 generate_usage_report 的報告生成功能，並新增 Y3 趨勢圖與評估指標。
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
SCRIPT_DIR = "./"#os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model_y2.json")
FEATURES_PATH = os.path.join(SCRIPT_DIR, "features2.pkl")

# --- Y3 計算函式 ---
def y2_2_y3(DeSOx_2nd, MLUT4_AIC_232B):
  return -DeSOx_2nd * MLUT4_AIC_232B + MLUT4_AIC_232B

# --- 輔助函式 ---
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

# --- 線上學習預測器 (Y2 完整版 v2) ---
class OnlinePredictor:
    def __init__(self, time_windows_len=100, threshold_update_window=100, percentile_for_threshold=90, min_train_samples=10):
        print("初始化 OnlinePredictor (Y2 完整版 v2)...")
        with open(FEATURES_PATH, "rb") as f: training_features = pickle.load(f)
        self.target_col = 'DeSOx_2nd'
        self.y3_true_col = 'MLUT4_AT-240'
        self.y3_dependency_col = 'MLUT4_AIC-232B'
        self.feature_cols = [f for f in training_features if f != self.target_col] + ['prev_target']
        self.model = None
        if os.path.exists(MODEL_PATH): 
            print(f"找到初始模型，正在載入: {MODEL_PATH}")
            self.model = xgb.XGBRegressor()
            self.model.load_model(MODEL_PATH)
        else: print(f"未找到初始模型，將在收到足夠數據後進行首次訓練。 ({MODEL_PATH})")
        self.time_windows_len, self.threshold_update_window, self.percentile_for_threshold, self.min_train_samples = time_windows_len, threshold_update_window, percentile_for_threshold, min_train_samples
        self.recent_errors, self.history = [], []
        self.threshold, self.step_counter, self.online_learning_started = np.inf, 0, False
        self.data_buffer = pd.DataFrame()
        self.kf = KalmanFilter(transition_matrices=1.0, observation_matrices=1.0, transition_covariance=0.3, observation_covariance=1.0)
        self.kf_mean, self.kf_cov = np.array([[0.85]]), np.array([[1.0]])

    def predict_and_learn(self, current_features_df, last_true_target=None):
        self.step_counter += 1
        log_entry = {'step': self.step_counter, 'rebuild_triggered': False, 'last_true_target': last_true_target}
        if self.y3_true_col in current_features_df.columns:
            log_entry['actual_y3'] = current_features_df[self.y3_true_col].iloc[0]
        else:
            log_entry['actual_y3'] = np.nan

        if last_true_target is not None:
            if not self.online_learning_started:
                print("*** 收到第一筆真實數據，觸發熱啟動！ ***")
                self.model, self.data_buffer, self.recent_errors, self.online_learning_started = None, pd.DataFrame(), [], True
                self.kf_mean, self.kf_cov = np.array([[last_true_target]]), np.array([[1.0]])
            self.kf_mean, self.kf_cov = self.kf.filter_update(self.kf_mean, self.kf_cov, last_true_target)
            if not self.data_buffer.empty:
                last_prediction = self.data_buffer.iloc[-1]['prediction_y2']
                if not pd.isna(last_prediction):
                    error = abs(last_prediction - last_true_target)
                    self.recent_errors.append(error)
                    log_entry['error'] = error
                    self.threshold = np.percentile(self.recent_errors[-self.threshold_update_window:], self.percentile_for_threshold)
                    log_entry['threshold'] = self.threshold
                    if error > self.threshold:
                        error_exceeded_threshold = True
                        log_entry['rebuild_triggered'] = True
                        print("*** 誤差超過門檻，觸發模型再訓練！ ***")
                self.data_buffer.at[self.data_buffer.index[-1], self.target_col] = last_true_target

        if self.model is None or log_entry['rebuild_triggered']:
            if not self.data_buffer.empty and self.target_col in self.data_buffer.columns:
                train_ready_df = self.data_buffer.dropna(subset=[self.target_col])
                if len(train_ready_df) >= self.min_train_samples:
                    print(f"數據量達標(>{self.min_train_samples})，使用最近 {len(train_ready_df)} 筆資料進行模型訓練...")
                    train_df = train_ready_df.copy()
                    train_df["prev_target"] = apply_sequential_kalman_filter(train_df[self.target_col].shift(1))
                    train_df[self.target_col] = apply_sequential_kalman_filter(train_df[self.target_col])
                    train_df = train_df.dropna(subset=self.feature_cols)
                    self.model = train_model(train_df[self.feature_cols], train_df[self.target_col], np.linspace(0.1, 1.0, num=len(train_df)))
                    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
                    self.model.save_model(MODEL_PATH)
                    self.recent_errors, self.threshold = [], np.inf

        prediction_y2, prediction_y3 = None, None
        if self.model is not None:
            predict_input_df = current_features_df.copy()
            predict_input_df['prev_target'] = self.kf_mean[0, 0]
            predict_X = predict_input_df[self.feature_cols]
            log_entry['features_used'] = predict_X.to_dict('records')[0]
            prediction_y2 = self.model.predict(predict_X)[0]
            if self.y3_dependency_col in predict_input_df.columns:
                y3_dependency_val = predict_input_df[self.y3_dependency_col].iloc[0]
                prediction_y3 = y2_2_y3(prediction_y2, y3_dependency_val)
        
        log_entry['prediction_y2'], log_entry['prediction_y3'] = prediction_y2, prediction_y3
        new_row = current_features_df.copy()
        new_row['prediction_y2'], new_row['prediction_y3'], new_row[self.target_col] = prediction_y2, prediction_y3, np.nan
        self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True).iloc[-self.time_windows_len:]
        self.history.append(log_entry)
        return {'DeSOx_2nd_pred': prediction_y2, 'Y3_pred': prediction_y3}

    def generate_usage_report(self, report_path=os.path.join(SCRIPT_DIR, "usage_report_y2.html")):
        print(f"\n正在生成 Y2/Y3 使用報告至 {report_path}...")
        if not self.history: return print("沒有歷史紀錄可供生成報告。\n")
        history_df = pd.DataFrame(self.history)
        history_df['actual_target'] = history_df['last_true_target'].shift(-1)

        def get_metrics_html(df, pred_col, true_col, title):
            calc_df = df.dropna(subset=[pred_col, true_col])
            if len(calc_df) < 2: return f"<h2>{title}</h2><p>沒有足夠的數據來計算指標。</p>"
            y_true, y_pred = calc_df[true_col], calc_df[pred_col]
            r2, rmse, mape = r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), np.mean(np.abs((y_true - y_pred) / y_true[y_true != 0])) * 100
            return f"<h2>{title}</h2>" + pd.DataFrame({'Metric':['R-squared','RMSE','MAPE'],'Value':[r2,rmse,mape]}).to_html(index=False, classes='table')

        metrics_html_y2 = get_metrics_html(history_df, 'prediction_y2', 'actual_target', f"Y2 預測評估指標 ({self.target_col})")
        metrics_html_y3 = get_metrics_html(history_df, 'prediction_y3', 'actual_y3', f"Y3 反推評估指標 ({self.y3_true_col})")

        def get_plot_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        plt.style.use('seaborn-v0_8-whitegrid')
        fig1, ax1 = plt.subplots(figsize=(12, 6)); ax1.plot(history_df['step'], history_df['actual_target'], 'o-', label=f'Y2 Actual ({self.target_col})'); ax1.plot(history_df['step'], history_df['prediction_y2'], 'x--', label='Y2 Predicted'); ax1.set_title('Y2 Actual vs. Predicted'); ax1.legend()
        fig2, ax2 = plt.subplots(figsize=(12, 6)); ax2.plot(history_df['step'], history_df['error'], 'o-', label='Prediction Error'); ax2.plot(history_df['step'], history_df['threshold'], 'r--', label='Rebuild Threshold'); ax2.set_title('Y2 Prediction Error vs. Threshold'); ax2.legend()
        fig3, ax3 = plt.subplots(figsize=(12, 6)); ax3.plot(history_df['step'], history_df['actual_y3'], 'o-', label=f'Y3 Actual ({self.y3_true_col})'); ax3.plot(history_df['step'], history_df['prediction_y3'], 'x--', color='green', label='Y3 Calculated'); ax3.set_title('Y3 Actual vs. Calculated'); ax3.legend()
        
        images = [get_plot_base64(f) for f in [fig1, fig2, fig3]]
        summary_text = f"在 {len(history_df)} 個模擬步驟中，總共觸發了 {history_df['rebuild_triggered'].sum()} 次模型重建。"

        html_content = f'''<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Y2/Y3 線上學習使用報告</title><style>body{{font-family:sans-serif;margin:20px}}h1,h2{{color:#333}}table{{border-collapse:collapse;width:100%;margin-top:20px}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background-color:#f2f2f2}}.container{{max-width:1200px;margin:auto}}.summary{{background-color:#eef;padding:15px;border-left:5px solid #66f;margin-top:20px}}img{{max-width:100%;height:auto;margin-top:20px;border:1px solid #ccc}}</style></head><body><div class="container"><h1>Y2/Y3 線上學習使用報告</h1>{metrics_html_y2}{metrics_html_y3}<h2>運行總結</h2><div class="summary"><p>{summary_text}</p></div><h2>Y2 實際值 vs. 預測值</h2><img src="data:image/png;base64,{images[0]}"><h2>Y2 預測誤差 vs. 重建門檻</h2><img src="data:image/png;base64,{images[1]}"><h2>Y3 實際值 vs. 反推值</h2><img src="data:image/png;base64,{images[2]}"><h2>詳細歷史紀錄</h2>{history_df.to_html(index=False,classes='table',float_format='%.4f')}</div></body></html>'''
        with open(report_path, "w", encoding="utf-8") as f: f.write(html_content)
        print(f"報告已成功儲存至: {report_path}")
