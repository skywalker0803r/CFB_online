# CFB 脫硫劑效率線上學習預測專案

這是一個用於預測 CFB（循環流化床）鍋爐脫硫效率（`DeSOx_1st`）的機器學習專案。其核心是一個能夠進行線上學習（Online Learning）的 XGBoost 模型，能夠根據即時數據不斷調整和優化自身，以適應生產環境中可能發生的動態變化。

---

## 核心功能

本專案從一個基本的批次訓練腳本，逐步演進為一個具備多種先進策略的智慧預測系統：

- **動態模型再訓練**：能夠根據預測誤差，自動判斷是否需要使用最新的數據重新訓練模型。
- **訓練/服務一致性**：透過統一的序列化卡爾曼濾波（Sequential Kalman Filter）處理流程，確保了訓練和線上推論時的數據一致性，解決了嚴重的分佈歪斜問題。
- **樣本加權**：在訓練時給予時間上較新的樣本更高的權重，使模型能更快地適應近期的工況變化。
- **熱啟動 (Warm-up & Reset)**：當真實的線上數據首次進入系統時，能果斷拋棄可能已過時的歷史模型，完全從零開始學習，避免歷史數據的污染。
- **漸進式暖機 (Progressive Warm-up)**：在熱啟動後，將模型的「冷卻時間」從 100 次預測大幅縮短至 10 次，一旦收集到少量新數據便立刻產生可用的「迷你模型」，確保服務近乎不中斷。
- **自動化報告**：能夠為批次訓練和線上模擬分別生成詳細的視覺化報告（`training_report.html` 和 `usage_report.html`），包含 R2、RMSE、MAPE 等關鍵指標和圖表，方便對模型性能進行評估。

---

## 檔案結構

```
.
├── model/                    # 存放訓練好的模型檔案
│   └── xgb_model.json
├── .gitignore                # Git 忽略規則
├── analyze_skew.py           # (偵錯用) 分析訓練與推論數據分佈差異的腳本
├── example_usage.py          # 線上學習與預測的模擬範例
├── features1.pkl             # 模型使用的特徵列表
├── inference.py              # 核心推論與線上學習邏輯 (OnlinePredictor)
├── README.md                 # 專案說明文件 (就是本檔案)
├── requirements.txt          # Python 套件依賴列表
├── skew_analysis_plot.png    # (偵錯用) 數據分佈歪斜的視覺化圖表
├── test0609-CFB2脫硫劑優化改善.feather  # (被忽略) 原始數據範例
├── train.py                  # 初始模型批次訓練腳本
├── train_debug_data.csv      # (偵錯用) 訓練數據日誌
└── inference_debug_data.csv  # (偵錯用) 推論數據日誌
```

---

## 安裝與設定

1. **克隆專案**
   ```bash
   git clone https://github.com/skywalker0803r/CFB_online.git
   cd CFB_online
   ```

2. **(建議) 建立虛擬環境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上是 `venv\Scripts\activate`
   ```

3. **安裝依賴套件**
   ```bash
   pip install -r requirements.txt
   ```

---

## 使用流程

本專案的使用分為兩個主要階段：

### 階段一：初始模型訓練

首先，執行 `train.py` 來訓練一個初始的基準模型。這個模型將作為線上服務啟動時的基礎。

```bash
python train.py
```

此步驟會讀取 `.feather` 數據，進行一次完整的批次訓練，並產生：
- `model/xgb_model.json`：初始模型檔案。
- `training_report.html`：一份關於模型在歷史數據上表現的詳細報告。

### 階段二：線上學習與預測

這是系統上線後，持續運行的核心部分。`example_usage.py` 檔案是一個完整的模擬範例。

```bash
python example_usage.py
```

此腳本會：
1. 初始化 `inference.py` 中的 `OnlinePredictor`，並載入初始模型。
2. 進入一個迴圈，模擬即時數據不斷傳入的場景。
3. 在迴圈中，持續進行「預測 -> 接收真實回饋 -> 學習 -> 判斷是否再訓練」的智慧循環。
4. 模擬結束後，會產生一份 `usage_report.html`，詳細記錄本次線上模擬的所有動態過程與最終的模型表現。
