# CFB 脫硫劑效率線上學習預測專案 - 簡報生成素材

## 專案概述 (Project Overview)

### 專案背景
- **專案名稱**: CFB (循環流化床) 脫硫劑效率線上學習預測系統
- **核心目標**: 建立一個能夠即時預測脫硫劑效率的機器學習系統
- **應用場景**: 工業級CFB鍋爐脫硫程序優化
- **技術特色**: 線上學習、即時預測、自適應模型更新

### 系統架構圖建議
```
數據來源 → 特徵工程 → 線上學習預測器 → 預測輸出
    ↓           ↓            ↓           ↓
PI Server   Kalman濾波   XGBoost模型    Y1/Y2/Y3預測
歷史數據    特徵選擇     動態更新       效率優化
```

---

## 核心技術模組分析

## 1. Y1 預測模組 (DeSOx_1st 預測)

### 1.1 功能描述
- **目標變數**: DeSOx_1st (第一階段脫硫效率)
- **預測模型**: XGBoost 回歸器
- **學習方式**: 線上增量學習
- **特色**: 熱啟動機制 + 動態閾值調整

### 1.2 數學原理

#### XGBoost 數學基礎
```
目標函數: L = Σ l(yi, ŷi) + Σ Ω(fk)
其中:
- l(yi, ŷi): 損失函數 (平方誤差)
- Ω(fk): 正則化項
- fk: 第k棵樹的函數
```

#### 線上學習更新策略
```
誤差計算: error = |prediction - actual|
動態閾值: threshold = percentile(errors, 90%)
觸發條件: if error > threshold → 重新訓練
```

#### Kalman濾波平滑
```
狀態更新方程:
x_k = A * x_{k-1} + w_k
z_k = H * x_k + v_k

其中:
- A: 轉移矩陣 (設為1.0)
- H: 觀測矩陣 (設為1.0)
- w_k: 過程噪聲 (協方差=0.3)
- v_k: 觀測噪聲 (協方差=1.0)
```

### 1.3 關鍵程式碼結構

```python
class OnlinePredictor:
    def __init__(self, time_windows_len=100, threshold_update_window=100):
        # 模型初始化
        self.model = xgb.XGBRegressor(
            n_estimators=900, 
            learning_rate=0.028,
            max_depth=8,
            subsample=0.75,
            colsample_bytree=0.75
        )
        
    def predict_and_learn(self, current_features_df, last_true_target=None):
        # 核心預測與學習邏輯
        if self.model is None or self.rebuild_triggered:
            # 模型重訓練邏輯
            self.model = train_model(train_X, train_y, sample_weight)
        
        # 預測執行
        prediction = self.model.predict(predict_X)[0]
        return prediction
```

### 1.4 特徵工程策略
- **特徵選擇**: 基於重要性排序的自動特徵選擇
- **前一目標值**: `prev_target` 特徵，使用Kalman濾波平滑
- **時間窗口**: 滑動窗口機制，保持最近100筆數據

---

## 2. Y2 預測模組 (DeSOx_2nd 預測)

### 2.1 功能描述
- **目標變數**: DeSOx_2nd (第二階段脫硫效率)
- **衍生計算**: Y3 (MLUT4_AT-240) 反推計算
- **模型檔案**: `xgb_model_y2.json`

### 2.2 Y3 反推數學公式
```python
def y2_2_y3(DeSOx_2nd, MLUT4_AIC_232B):
    return -DeSOx_2nd * MLUT4_AIC_232B + MLUT4_AIC_232B
```

#### 數學解釋
```
Y3 = MLUT4_AIC_232B * (1 - DeSOx_2nd)

這個公式反映了:
- DeSOx_2nd: 第二階段脫硫效率 (0-1之間)
- MLUT4_AIC_232B: 基準參考值
- Y3: 經過脫硫處理後的最終輸出值
```

### 2.3 雙目標評估機制
```python
# Y2 評估指標
metrics_html_y2 = get_metrics_html(history_df, 'prediction_y2', 'actual_target', 
                                   "Y2 預測評估指標")

# Y3 評估指標  
metrics_html_y3 = get_metrics_html(history_df, 'prediction_y3', 'actual_y3', 
                                   "Y3 反推評估指標")
```

---

## 3. 線上學習核心算法

### 3.1 熱啟動機制 (Hot Start)
```python
if last_true_target is not None:
    if not self.online_learning_started:
        print("*** 收到第一筆真實數據，觸發熱啟動！ ***")
        # 重置所有狀態
        self.model = None
        self.data_buffer = pd.DataFrame()
        self.recent_errors = []
        self.online_learning_started = True
```

### 3.2 動態閾值調整
```python
# 誤差追蹤
error = abs(last_prediction - last_true_target)
self.recent_errors.append(error)

# 動態閾值計算（90th百分位數）
self.threshold = np.percentile(
    self.recent_errors[-self.threshold_update_window:], 
    self.percentile_for_threshold
)

# 模型重建觸發條件
if error > self.threshold:
    log_entry['rebuild_triggered'] = True
    print("*** 誤差超過門檻，觸發模型再訓練！ ***")
```

### 3.3 樣本權重策略
```python
# 線性遞增權重，越新的數據權重越高
sample_weight = np.linspace(0.1, 1.0, num=len(train_df))
```

### 3.4 時間序列特徵處理
```python
def apply_sequential_kalman_filter(series, transition_covariance=0.3, 
                                   observation_covariance=1.0, 
                                   initial_state_covariance=1.0):
    kf = KalmanFilter(
        transition_matrices=1.0, 
        observation_matrices=1.0,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance
    )
    # 序列式Kalman濾波處理
```

---

## 4. 數據歪斜分析模組

### 4.1 功能描述
- **檔案**: `Y1/analyze_skew.py`
- **目的**: 檢測訓練數據與推理數據之間的分佈差異
- **方法**: 統計比較 + 視覺化分析

### 4.2 分析方法
```python
# 統計指標比較
for feature in common_features:
    train_stats = train_df[feature].describe()
    inference_stats = inference_features_df[feature].describe()
    
    stats_dict = {
        'feature': feature,
        'train_mean': train_stats['mean'],
        'inference_mean': inference_stats['mean'],
        'train_std': train_stats['std'],
        'inference_std': inference_stats['std'],
        # ... 更多統計指標
    }
```

### 4.3 視覺化比較
```python
# KDE分佈比較圖
sns.kdeplot(train_df[feature_to_plot], label='Train Data Distribution', 
            fill=True, alpha=0.5)
sns.kdeplot(inference_features_df[feature_to_plot], 
            label='Inference Data Distribution', fill=True, alpha=0.5)
```

---

## 5. 工業整合橋接器

### 5.1 PI Server 整合架構
```python
class PIServerBridge:
    def __init__(self, config):
        self.config = config
        self.predictor_y1 = OnlinePredictorY1()
        self.predictor_y2 = OnlinePredictorY2()
        
    def collect_and_predict(self):
        # PI Server 數據收集
        current_data = self.collect_pi_data()
        
        # 雙模型預測
        y1_pred = self.predictor_y1.predict_and_learn(current_data)
        y2_pred = self.predictor_y2.predict_and_learn(current_data)
        
        return {
            'Y1_prediction': y1_pred,
            'Y2_prediction': y2_pred['DeSOx_2nd_pred'],
            'Y3_calculation': y2_pred['Y3_pred']
        }
```

### 5.2 配置管理系統
```python
class Config:
    def __init__(self):
        # PI Server 設定
        self.PI_SERVER_CONFIG = {
            'enabled': True,
            'server_address': 'PI-SERVER-01',
            'username': 'piuser',
            'collection_interval': 5,  # 秒
            'tag_mappings': {...}
        }
        
        # 預測模型設定
        self.MODEL_CONFIG = {
            'time_window_length': 100,
            'threshold_percentile': 90,
            'min_training_samples': 10
        }
```

---

## 6. 評估指標與報告系統

### 6.1 核心評估指標

#### R-squared (決定係數)
```
R² = 1 - (SS_res / SS_tot)
其中:
- SS_res = Σ(yi - ŷi)²  (殘差平方和)
- SS_tot = Σ(yi - ȳ)²   (總平方和)
```

#### RMSE (均方根誤差)
```
RMSE = √(Σ(yi - ŷi)² / n)
```

#### MAPE (平均絕對百分比誤差)
```
MAPE = (1/n) * Σ|((yi - ŷi) / yi)| * 100%
```

### 6.2 報告生成系統
```python
def generate_usage_report(self, report_path):
    # 1. 計算評估指標
    metrics_html = get_metrics_html(history_df, pred_col, true_col, title)
    
    # 2. 生成趨勢圖
    fig1 = plot_actual_vs_predicted()
    fig2 = plot_error_vs_threshold()
    fig3 = plot_y3_comparison()
    
    # 3. HTML報告組裝
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head><title>線上學習使用報告</title></head>
    <body>
        {metrics_html}
        <img src="data:image/png;base64,{image_base64}">
    </body>
    </html>
    '''
```

---

## 7. 測試與驗證框架

### 7.1 整合測試架構
```python
class IntegrationTester:
    def test_end_to_end_prediction(self):
        # 端到端預測測試
        
    def test_online_learning_convergence(self):
        # 線上學習收斂性測試
        
    def test_pi_server_integration(self):
        # PI Server整合測試
```

### 7.2 模擬數據流測試
```python
# 模擬即時數據流
for i in range(num_simulation_steps):
    current_features = simulation_df.iloc[[i]]
    prediction = predictor.predict_and_learn(current_features, last_true_target)
    
    # 驗證預測結果
    assert prediction is not None
    print(f"時間點 {i}: 預測值 = {prediction:.4f}")
```

---

## 8. 技術亮點與創新特色

### 8.1 線上學習創新
1. **熱啟動機制**: 首次接收真實數據時自動重置並開始學習
2. **動態閾值**: 基於歷史誤差的自適應重建門檻
3. **漸進式權重**: 新數據具有更高的學習權重

### 8.2 數據處理創新
1. **Kalman濾波**: 對時間序列特徵進行噪聲過濾
2. **特徵追蹤**: `prev_target` 特徵增強時間依賴性
3. **數據歪斜檢測**: 主動監控訓練與推理數據分佈差異

### 8.3 工業應用創新
1. **PI Server無縫整合**: 支援工業級數據收集系統
2. **多目標預測**: Y1, Y2, Y3 三層次預測體系
3. **即時報告**: 自動生成HTML格式的使用報告

---

## 9. 部署與維護指南

### 9.1 系統需求
```python
# requirements.txt
pandas
xgboost
numpy
pykalman
scikit-learn
matplotlib
seaborn
```

### 9.2 啟動流程
```bash
# 1. 模型訓練
python Y1/train.py
python Y2/train.py

# 2. 線上預測服務
python Y1/example_usage.py
python Y2/example_usage.py

# 3. PI Server整合
python integration_bridge.py
```

### 9.3 監控與維護
- **模型效能監控**: 通過 `usage_report.html` 定期檢查
- **數據歪斜監控**: 定期執行 `analyze_skew.py`
- **系統健康檢查**: 使用 `test_integration.py` 進行驗證

---

## 10. 數學公式總結

### 10.1 核心算法公式
```
1. XGBoost目標函數:
   L = Σ l(yi, ŷi) + Σ Ω(fk)

2. Kalman濾波方程:
   x_k = A·x_{k-1} + w_k
   z_k = H·x_k + v_k

3. Y3反推公式:
   Y3 = MLUT4_AIC_232B × (1 - DeSOx_2nd)

4. 動態閾值:
   threshold = percentile(errors, 90%)

5. 評估指標:
   R² = 1 - SS_res/SS_tot
   RMSE = √(Σ(yi-ŷi)²/n)
   MAPE = (1/n)×Σ|((yi-ŷi)/yi)|×100%
```

---

## 結論與未來發展

### 技術成就
1. 實現了完整的工業級線上學習系統
2. 建立了多層次的脫硫效率預測體系
3. 提供了完整的數據歪斜檢測與報告機制

### 應用價值
1. **提升脫硫效率**: 即時預測與優化建議
2. **降低操作成本**: 自動化的預測與調整
3. **增強系統穩定性**: 線上學習適應環境變化

### 未來發展方向
1. **深度學習整合**: 引入LSTM、Transformer等先進模型
2. **多站點部署**: 支援多個CFB鍋爐同時監控
3. **預測性維護**: 整合設備健康監控功能

---

*此簡報素材涵蓋了CFB脫硫劑效率線上學習預測專案的所有核心技術、數學原理、程式碼架構和應用價值，適合用於製作詳細的技術簡報。*