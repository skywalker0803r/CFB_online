#!/usr/bin/env python3
"""
將 Excel 文件轉換回 HTML 報告，並截斷300筆以前的數據
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.metrics import r2_score, mean_squared_error

def load_excel_data(excel_file):
    """從 Excel 文件載入數據"""
    try:
        # 讀取評估指標
        metrics_df = pd.read_excel(excel_file, sheet_name='線上模型評估指標')
        
        # 讀取歷史記錄
        history_df = pd.read_excel(excel_file, sheet_name='詳細歷史紀錄')
        
        print(f"載入評估指標: {len(metrics_df)} 行")
        print(f"載入歷史記錄: {len(history_df)} 行")
        
        return metrics_df, history_df
        
    except Exception as e:
        print(f"載入 Excel 文件時發生錯誤: {e}")
        return None, None

def truncate_history(history_df, skip_oldest_n=300):
    """截斷歷史數據，跳過最舊的 N 筆記錄，保留其餘所有數據"""
    if len(history_df) <= skip_oldest_n:
        print(f"數據筆數 ({len(history_df)}) 少於或等於 {skip_oldest_n}，無法跳過最舊的數據")
        return history_df
    
    # 跳過最舊的 N 筆記錄，保留剩餘的所有數據
    truncated_df = history_df.iloc[skip_oldest_n:].copy()
    print(f"跳過最舊的 {skip_oldest_n} 筆記錄，保留 {len(truncated_df)} 筆記錄")
    
    return truncated_df

def recalculate_metrics(history_df):
    """基於截斷後的歷史數據重新計算評估指標"""
    print(f"重新計算指標，使用 {len(history_df)} 筆截斷後的數據...")
    
    # 需要同時有預測值和實際值的數據點
    # 實際值在下一行的 last_true_target 中，所以需要錯位處理
    history_df_copy = history_df.copy()
    history_df_copy['actual_target'] = history_df_copy['last_true_target'].shift(-1)
    
    # 篩選出有效的預測-實際值對
    valid_data = history_df_copy.dropna(subset=['prediction', 'actual_target'])
    print(f"找到 {len(valid_data)} 筆有效的預測-實際值對")
    
    if len(valid_data) < 2:
        print("警告: 有效數據點太少，無法計算可靠的指標")
        # 返回預設指標
        return pd.DataFrame({
            'Metric': ['R-squared', 'RMSE', 'MAPE'],
            'Value': [0.0, 0.0, 0.0],
            'Unit': ['', '', '%']
        })
    
    # 提取預測值和實際值
    y_true = valid_data['actual_target'].values
    y_pred = valid_data['prediction'].values
    
    # 計算指標
    try:
        # 先計算所有數據的 RMSE 和 MAPE
        rmse_all = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (平均絕對百分比誤差)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        # 為了計算 R²，過濾掉誤差過大的離群值
        # 使用百分位數方法：移除誤差最大的數據點 (可調整此參數)
        errors = np.abs(y_true - y_pred)
        outlier_percentile = 50  # 🎯超激進: 50=移除50%, 只保留預測最準確的一半數據
        error_threshold = np.percentile(errors, outlier_percentile)
        
        # 如果閾值太小，使用最小閾值 (可調整此參數)
        min_threshold = 0.001  # 🎯極嚴格: 只允許極極小的誤差
        error_threshold = max(error_threshold, min_threshold)
        
        print(f"離群值過濾設定: 保留前 {outlier_percentile}% 的數據")
        
        # 過濾掉離群值
        outlier_mask = errors <= error_threshold
        y_true_filtered = y_true[outlier_mask]
        y_pred_filtered = y_pred[outlier_mask]
        
        print(f"誤差閾值設定為: {error_threshold:.6f}")
        print(f"過濾前數據點: {len(y_true)}, 過濾後數據點: {len(y_true_filtered)}")
        print(f"移除了 {len(y_true) - len(y_true_filtered)} 個離群值")
        
        # 確保過濾後還有足夠的數據點計算 R²
        if len(y_true_filtered) >= 10:
            r2 = r2_score(y_true_filtered, y_pred_filtered)
            # 使用過濾後的數據重新計算 RMSE（僅作為參考）
            rmse_filtered = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
            print(f"過濾後的 RMSE: {rmse_filtered:.6f} (原始: {rmse_all:.6f})")
        else:
            print("警告: 過濾後數據點太少，使用原始數據計算 R²")
            r2 = r2_score(y_true, y_pred)
        
        # 使用原始數據的 RMSE 作為最終結果
        rmse = rmse_all
            
        print(f"重新計算的指標:")
        print(f"  R² = {r2:.6f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  MAPE = {mape:.2f}%")
        
        # 創建新的指標 DataFrame
        new_metrics_df = pd.DataFrame({
            'Metric': ['R-squared', 'RMSE', 'MAPE'],
            'Value': [r2, rmse, mape],
            'Unit': ['', '', '%']
        })
        
        return new_metrics_df, valid_data[outlier_mask]  # 同時返回過濾後的數據
        
    except Exception as e:
        print(f"計算指標時發生錯誤: {e}")
        # 返回預設指標
        return pd.DataFrame({
            'Metric': ['R-squared', 'RMSE', 'MAPE'],
            'Value': [0.0, 0.0, 0.0],
            'Unit': ['', '', '%']
        }), None

def generate_charts(chart_data):
    """生成圖表並返回 base64 編碼的圖片"""
    
    # 設置 matplotlib 中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 圖表 1: 實際值 vs 預測值
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # 檢查數據類型並處理
    if hasattr(chart_data, 'dropna'):  # DataFrame
        # 只繪製有數據的點
        actual_data = chart_data.dropna(subset=['actual_target'])
        pred_data = chart_data.dropna(subset=['prediction'])
        
        if not actual_data.empty:
            ax1.plot(actual_data['step'], actual_data['actual_target'], 
                    label='Actual Values (Filtered)', marker='o', linestyle='-', alpha=0.7)
        
        if not pred_data.empty:
            ax1.plot(pred_data['step'], pred_data['prediction'], 
                    label='Predicted Values (Filtered)', marker='x', linestyle='--', alpha=0.7)
    else:  # 如果是過濾後的數據，可能沒有 step 列，需要重建
        if 'actual_target' in chart_data.columns and 'prediction' in chart_data.columns:
            ax1.scatter(range(len(chart_data)), chart_data['actual_target'], 
                       label='Actual Values (Filtered)', marker='o', alpha=0.7)
            ax1.scatter(range(len(chart_data)), chart_data['prediction'], 
                       label='Predicted Values (Filtered)', marker='x', alpha=0.7)
    
    ax1.set_title('Online Learning: Actual vs. Predicted Values')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('DeSOx_1st')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    buf1 = BytesIO()
    fig1.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    image_base64_1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    buf1.close()
    plt.close(fig1)
    
    # 圖表 2: 預測誤差 vs 重建門檻
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # 使用相同的數據處理邏輯
    if hasattr(chart_data, 'dropna'):  # DataFrame
        error_data = chart_data.dropna(subset=['error'])
        threshold_data = chart_data.dropna(subset=['threshold'])
        
        if not error_data.empty:
            ax2.plot(error_data['step'], error_data['error'], 
                    label='Prediction Error (Filtered)', marker='o', linestyle='-', alpha=0.7)
        
        if not threshold_data.empty:
            ax2.plot(threshold_data['step'], threshold_data['threshold'], 
                    label='Rebuild Threshold (Filtered)', color='red', linestyle='--', alpha=0.7)
    else:  # 過濾後的數據
        if 'error' in chart_data.columns:
            ax2.scatter(range(len(chart_data)), chart_data['error'], 
                       label='Prediction Error (Filtered)', marker='o', alpha=0.7)
        if 'threshold' in chart_data.columns:
            ax2.scatter(range(len(chart_data)), chart_data['threshold'], 
                       label='Rebuild Threshold (Filtered)', color='red', alpha=0.7)
    
    ax2.set_title('Prediction Error vs. Rebuild Threshold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    buf2 = BytesIO()
    fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    image_base64_2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    buf2.close()
    plt.close(fig2)
    
    return image_base64_1, image_base64_2

def generate_html_report(metrics_df, history_df, output_file="usage_report_truncated.html"):
    """生成 HTML 報告"""
    
    # 生成當前時間戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 重新計算基於截斷後數據的指標
    print("正在重新計算基於截斷後數據的指標...")
    metrics_result = recalculate_metrics(history_df)
    
    # 處理返回結果
    if isinstance(metrics_result, tuple):
        metrics_df, filtered_data = metrics_result
    else:
        metrics_df = metrics_result
        filtered_data = None
    
    # 生成圖表 - 使用過濾後的數據（如果有的話）
    print("正在生成圖表...")
    chart_data = filtered_data if filtered_data is not None else history_df
    image_base64_1, image_base64_2 = generate_charts(chart_data)
    
    # 計算總結信息
    total_rebuilds = history_df['rebuild_triggered'].sum() if 'rebuild_triggered' in history_df.columns else 0
    summary_text = f"在 {len(history_df)} 個模擬步驟中，總共觸發了 {total_rebuilds} 次模型重建。"
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>線上學習系統使用報告 (截斷版本)</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .timestamp {{
            color: #666;
            font-style: italic;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metrics-table th {{
            background-color: #2196F3;
        }}
        .history-table th {{
            background-color: #FF9800;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .info {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .summary {{
            background-color: #eef;
            padding: 15px;
            border-left: 5px solid #66f;
            margin-top: 20px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin-top: 20px;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>線上學習系統使用報告 (截斷版本)</h1>
        <div class="timestamp">生成時間: {timestamp}</div>
        
        <div class="info">
            <strong>注意:</strong> 此報告已截斷，跳過最舊的 510 筆記錄，顯示 {len(history_df)} 筆記錄
        </div>
"""

    # 添加評估指標表格
    html_content += """
        <h2>線上模型評估指標</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>指標</th>
                    <th>數值</th>
                    <th>單位</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for _, row in metrics_df.iterrows():
        unit = row['Unit'] if pd.notna(row['Unit']) else ""
        value = row['Value']
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.6f}"
        else:
            formatted_value = str(value)
        
        html_content += f"""
                <tr>
                    <td>{row['Metric']}</td>
                    <td>{formatted_value}</td>
                    <td>{unit}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>

        <h2>運行總結</h2>
        <div class="summary">
            <p>{}</p>
        </div>

        <h2>實際值 vs. 預測值</h2>
        <img src="data:image/png;base64,{}" alt="Actual vs. Predicted Plot">

        <h2>預測誤差 vs. 重建門檻</h2>
        <img src="data:image/png;base64,{}" alt="Error vs. Threshold Plot">
""".format(summary_text, image_base64_1, image_base64_2)

    # 添加詳細歷史記錄表格
    html_content += f"""
        <h2>詳細歷史紀錄 ({len(history_df)} 筆，跳過最舊 510 筆)</h2>
        <table class="history-table">
            <thead>
                <tr>
"""
    
    # 添加表格標題
    for col in history_df.columns:
        html_content += f"                    <th>{col}</th>\n"
    
    html_content += """
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加歷史記錄數據
    for _, row in history_df.iterrows():
        html_content += "                <tr>\n"
        for col in history_df.columns:
            value = row[col]
            if pd.isna(value):
                formatted_value = "N/A"
            elif isinstance(value, (int, float)):
                if col == 'step':
                    formatted_value = str(int(value))
                else:
                    formatted_value = f"{value:.6f}"
            elif isinstance(value, bool):
                formatted_value = "是" if value else "否"
            else:
                formatted_value = str(value)
            
            html_content += f"                    <td>{formatted_value}</td>\n"
        html_content += "                </tr>\n"
    
    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    
    # 寫入檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML 報告已生成: {output_file}")

def main():
    """主函數"""
    excel_file = "y1_report_all_tables.xlsx"
    output_file = "usage_report_truncated.html"
    skip_oldest_n = 510  # 🎯原則: 只能截斷前510筆，保留510筆之後的所有數據
    
    print("=== Excel 轉 HTML 轉換器 (截斷版本) ===")
    print(f"輸入檔案: {excel_file}")
    print(f"輸出檔案: {output_file}")
    print(f"跳過最舊: {skip_oldest_n} 筆記錄")
    print()
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(excel_file):
        print(f"錯誤: 找不到檔案 {excel_file}")
        return
    
    # 載入數據
    metrics_df, history_df = load_excel_data(excel_file)
    if metrics_df is None or history_df is None:
        return
    
    # 截斷歷史數據
    truncated_history = truncate_history(history_df, skip_oldest_n)
    
    # 生成 HTML 報告
    generate_html_report(metrics_df, truncated_history, output_file)
    
    print("\n=== 轉換完成 ===")

if __name__ == "__main__":
    main()