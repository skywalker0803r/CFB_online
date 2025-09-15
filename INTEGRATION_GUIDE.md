# CFB 線上學習系統 - 業主整合指南

## 整合概述

本指南說明如何將您的線上學習預測系統與業主工程師的 PI Server 數據收集代碼整合。

## 整合架構

```
業主 PI Server → TEST0911/crawl_pi_server → integration_bridge.py → Y1/Y2 預測器 → 整合報告
```

## 安裝步驟

### 1. 環境準備

確保已安裝所有必要套件：

```bash
pip install -r requirements.txt
pip install openpyxl  # 用於處理業主的 Excel 輸出
```

### 2. 整合業主代碼

將業主工程師提供的 `crawl_pi_server.py` 模組放入 `TEST0911/` 資料夾：

```
TEST0911/
├── TEST0911.ipynb          # 業主原始代碼
├── crawl_pi_server.py      # 業主的 PI Server 模組 (需要添加)
└── __init__.py             # Python 套件標記檔案 (需要創建)
```

### 3. 創建初始化檔案

```bash
touch TEST0911/__init__.py
```

### 4. 配置 PI Server 連線

根據業主的 PI Server 設定，可能需要配置：
- PI Server 地址
- 認證資訊
- 網路權限

### 5. 測試整合

```bash
python integration_bridge.py
```

## 使用流程

### 方式一：歷史數據訓練模式

1. 運行整合橋接器收集歷史數據
2. 生成 Feather 格式訓練檔案
3. 重新訓練 Y1 和 Y2 模型

```bash
python integration_bridge.py
# 選擇選項 1: 收集歷史數據並訓練模型
```

### 方式二：即時預測模式

直接啟動即時預測迴圈，系統會：
1. 每 5 分鐘從 PI Server 獲取最新數據
2. 執行 Y1/Y2 線上學習預測
3. 根據預測誤差動態調整模型
4. 生成持續的預測記錄

```bash
python integration_bridge.py
# 選擇選項 2: 執行即時預測迴圈
```

### 方式三：單次測試模式

進行單次預測測試並生成報告：

```bash
python integration_bridge.py
# 選擇選項 3: 單次預測測試
```

## 關鍵整合點

### 1. 數據格式轉換

- **業主輸入**: PI Server Tags → Excel 檔案
- **您的系統**: Feather 格式 → 線上學習預測器
- **橋接器**: 自動處理格式轉換

### 2. 標籤對應

CFB1 和 CFB2 使用不同的感測器標籤：

| 功能 | CFB1 | CFB2 |
|------|------|------|
| ECO SOx | MLUT4_AIC-132B | MLUT4_AIC-232B |
| 煙囪 SOx | MLUT4_AT-140 | MLUT4_AT-240 |
| 總飼煤量 | MLUT4_FIQ-1BTCF | MLUT4_FIQ-2BTCF |
| 煙氣流量 | MLUT4_FQ-139 | MLUT4_FQ-239 |
| 石灰石 | MLUT4_RQ-1BTLS | MLUT4_RQ-2BTLS |

### 3. 計算公式整合

業主工程師的計算公式已整合到橋接器中：

```python
# 前爐SOx濃度計算
前爐SOx濃度 = 總飼煤量 * 8 / 煙氣流量 * 24.5 * 1000 / 32.065

# 鈣硫比計算
鈣硫比 = 石灰石 / 總飼煤量 * 0.32 / 0.008

# DeSOx_1st 計算 (第一段脫硫效率)
DeSOx_1st = (前爐SOx濃度 - ECO_SOx) / 前爐SOx濃度

# DeSOx_2nd 計算 (第二段脫硫效率)
DeSOx_2nd = (ECO_SOx - 煙囪SOx) / ECO_SOx
```

## 故障排除

### 1. PI Server 連線問題

如果無法連線到 PI Server：
- 檢查網路連線
- 確認 PI Server 地址和權限
- 系統會自動切換到模擬數據模式

### 2. 模組載入錯誤

```bash
# 確認 TEST0911 目錄結構
ls -la TEST0911/
# 應該包含：
# - __init__.py
# - crawl_pi_server.py
# - TEST0911.ipynb
```

### 3. 數據格式問題

整合橋接器會自動處理：
- 數值轉換錯誤
- 缺失值處理
- 時間戳格式統一

## 系統監控

### 即時狀態監控

```bash
# 查看預測歷史
ls -la *integration_history*.csv

# 查看模型報告
ls -la Y1/usage_report.html Y2/usage_report_y2.html
```

### 日誌訊息說明

- `✅` 成功操作
- `⚠️` 警告訊息
- `❌` 錯誤訊息
- `🎯` 預測結果
- `📡` 數據收集
- `💾` 檔案儲存

## 效能調優

### 1. 預測間隔調整

```python
# 在 integration_bridge.py 中調整
loop_interval = 300  # 5 分鐘 (建議)
# loop_interval = 60   # 1 分鐘 (高頻)
# loop_interval = 900  # 15 分鐘 (低頻)
```

### 2. 歷史數據範圍

```python
# 調整歷史數據收集天數
history_days = 50    # 預設 50 天
# history_days = 30  # 較短期間
# history_days = 100 # 較長期間
```

### 3. 模型參數調整

保留您原有的線上學習參數設定，橋接器不會修改這些設定。

## 生產部署建議

### 1. 服務化部署

建議將整合系統包裝為 Windows 服務或 Linux 守護程序：

```python
# 可以使用 python-windows-service 或 systemd
# 實現自動重啟和錯誤恢復
```

### 2. 監控與告警

- 設定 PI Server 連線監控
- 模型預測精度告警
- 系統資源使用監控

### 3. 備份與恢復

- 定期備份訓練好的模型
- 保留重要的預測歷史記錄
- 建立緊急模式運行機制

## 聯絡支援

如果在整合過程中遇到問題，請提供：
1. 錯誤訊息截圖
2. 系統環境資訊
3. PI Server 連線狀態
4. 數據樣本檔案

---

*此整合指南由 Rovo Dev 協助創建*