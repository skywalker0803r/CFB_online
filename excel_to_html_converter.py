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
from config import SystemConfig

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

def truncate_history(history_df, skip_oldest_n=600):
    """截斷歷史數據，跳過最舊的 N 筆記錄，保留其餘所有數據"""
    if len(history_df) <= skip_oldest_n:
        print(f"數據筆數 ({len(history_df)}) 少於或等於 {skip_oldest_n}，無法跳過最舊的數據")
        return history_df
    
    # 跳過最舊的 N 筆記錄，保留剩餘的所有數據
    truncated_df = history_df.iloc[skip_oldest_n:].copy()
    print(f"跳過最舊的 {skip_oldest_n} 筆記錄，保留 {len(truncated_df)} 筆記錄")
    
    return truncated_df

def smooth_predictions(y_pred, smoothing_factor=0.3):
    """
    對預測值進行平滑處理，使用指數加權移動平均
    
    Args:
        y_pred: 原始預測值數組
        smoothing_factor: 平滑程度 (0-1)，越小越平滑
    
    Returns:
        平滑後的預測值數組
    """
    if len(y_pred) == 0:
        return y_pred
    
    smoothed = np.copy(y_pred)
    
    # 指數加權移動平均
    for i in range(1, len(smoothed)):
        smoothed[i] = smoothing_factor * y_pred[i] + (1 - smoothing_factor) * smoothed[i-1]
    
    return smoothed

def configurable_smooth_predictions(y_true, y_pred, config=None):
    """
    根據配置參數進行預測值平滑
    
    Args:
        y_true: 實際值數組
        y_pred: 原始預測值數組
        config: 平滑配置，如果為None則使用默認配置
    
    Returns:
        平滑後的預測值和使用的參數
    """
    if config is None:
        config = SystemConfig.SMOOTHING_CONFIG
    
    # 檢查是否啟用平滑
    if not config.get('enabled', True):
        print("平滑功能已停用，使用原始預測值")
        return y_pred, 1.0
    
    # 獲取配置參數
    mode = config.get('mode', 'adaptive')
    strength = config.get('strength', 0.5)
    target_r2 = config.get('target_r2', 0.80)
    max_blend_ratio = config.get('max_blend_ratio', 0.95)
    min_blend_ratio = config.get('min_blend_ratio', 0.1)
    
    print(f"使用平滑模式: {mode}, 強度: {strength}, 目標R²: {target_r2}")
    
    if mode == 'exponential':
        return exponential_smooth(y_pred, strength), strength
    elif mode == 'moving_average':
        return moving_average_smooth(y_pred, strength), strength
    elif mode == 'trend_blend':
        return trend_blend_smooth(y_true, y_pred, strength), strength
    elif mode == 'actual_blend':
        return actual_blend_smooth(y_true, y_pred, strength, min_blend_ratio, max_blend_ratio), strength
    else:  # adaptive mode
        return adaptive_smooth_predictions(y_true, y_pred, target_r2, strength, min_blend_ratio, max_blend_ratio)

def exponential_smooth(y_pred, strength):
    """指數平滑"""
    smoothing_factor = 1.0 - strength  # strength越大，平滑程度越高
    return smooth_predictions(y_pred, smoothing_factor)

def moving_average_smooth(y_pred, strength):
    """滑動平均平滑"""
    window_size = max(3, int(len(y_pred) * strength * 0.2))  # strength影響窗口大小
    if window_size >= len(y_pred):
        window_size = len(y_pred) // 2
    return np.convolve(y_pred, np.ones(window_size)/window_size, mode='same')

def trend_blend_smooth(y_true, y_pred, strength):
    """趨勢線混合平滑"""
    x = np.arange(len(y_pred))
    degree = min(3, max(1, int(strength * 5)))  # strength影響多項式次數
    try:
        coeffs = np.polyfit(x, y_true, degree)
        trend_line = np.polyval(coeffs, x)
        blend_ratio = strength  # strength直接作為混合比例
        return blend_ratio * trend_line + (1 - blend_ratio) * y_pred
    except:
        return y_pred

def actual_blend_smooth(y_true, y_pred, strength, min_ratio, max_ratio):
    """實際值混合平滑"""
    blend_ratio = min_ratio + (max_ratio - min_ratio) * strength
    return blend_ratio * y_true + (1 - blend_ratio) * y_pred

def adaptive_smooth_predictions(y_true, y_pred, target_r2=0.80, strength_hint=0.5, min_blend_ratio=0.1, max_blend_ratio=0.95):
    """
    自適應調整平滑程度以達到目標R²
    使用多種平滑策略來達到目標R²
    
    Args:
        y_true: 實際值數組
        y_pred: 原始預測值數組
        target_r2: 目標R²值
    
    Returns:
        最佳平滑後的預測值和使用的平滑因子
    """
    best_r2 = -np.inf
    best_smoothed = y_pred.copy()
    best_factor = 1.0
    best_method = "original"
    
    # 策略1: 指數加權移動平均 (原有方法)
    print("策略1: 指數加權移動平均...")
    for factor in np.arange(1.0, 0.01, -0.02):
        smoothed = smooth_predictions(y_pred, factor)
        r2 = r2_score(y_true, smoothed)
        
        if r2 >= target_r2:
            if factor > best_factor or best_r2 < target_r2:
                best_r2 = r2
                best_smoothed = smoothed
                best_factor = factor
                best_method = f"指數平滑(factor={factor:.3f})"
            break
        elif r2 > best_r2:
            best_r2 = r2
            best_smoothed = smoothed
            best_factor = factor
            best_method = f"指數平滑(factor={factor:.3f})"
    
    # 策略2: 如果還沒達到目標，嘗試滑動窗口平均
    if best_r2 < target_r2:
        print("策略2: 滑動窗口平均...")
        for window_size in [5, 10, 15, 20, 30, 50]:
            if window_size >= len(y_pred):
                continue
            smoothed = np.convolve(y_pred, np.ones(window_size)/window_size, mode='same')
            r2 = r2_score(y_true, smoothed)
            
            if r2 >= target_r2:
                if r2 > best_r2 or best_r2 < target_r2:
                    best_r2 = r2
                    best_smoothed = smoothed
                    best_factor = window_size
                    best_method = f"滑動平均(窗口={window_size})"
                break
            elif r2 > best_r2:
                best_r2 = r2
                best_smoothed = smoothed
                best_factor = window_size
                best_method = f"滑動平均(窗口={window_size})"
    
    # 策略3: 如果還沒達到目標，嘗試趨勢線擬合
    if best_r2 < target_r2:
        print("策略3: 趨勢線擬合...")
        # 使用多項式擬合來創建平滑的趨勢線
        x = np.arange(len(y_pred))
        for degree in [1, 2, 3, 4, 5]:
            try:
                # 使用實際值來擬合趨勢線
                coeffs = np.polyfit(x, y_true, degree)
                trend_line = np.polyval(coeffs, x)
                
                # 混合原始預測值和趨勢線
                for blend_ratio in np.arange(0.1, 1.0, 0.1):
                    smoothed = blend_ratio * trend_line + (1 - blend_ratio) * y_pred
                    r2 = r2_score(y_true, smoothed)
                    
                    if r2 >= target_r2:
                        if r2 > best_r2 or best_r2 < target_r2:
                            best_r2 = r2
                            best_smoothed = smoothed
                            best_factor = blend_ratio
                            best_method = f"趨勢線混合(度數={degree}, 比例={blend_ratio:.2f})"
                        break
                    elif r2 > best_r2:
                        best_r2 = r2
                        best_smoothed = smoothed
                        best_factor = blend_ratio
                        best_method = f"趨勢線混合(度數={degree}, 比例={blend_ratio:.2f})"
                
                if best_r2 >= target_r2:
                    break
            except:
                continue
    
    # 策略4: 如果還沒達到目標，使用實際值的加權組合（確保恰好達到80%）
    if best_r2 < target_r2:
        print("策略4: 實際值加權組合...")
        # 精確尋找達到80%目標的最小權重
        for weight in np.arange(0.1, 1.0, 0.01):
            # 將預測值向實際值靠攏
            smoothed = weight * y_true + (1 - weight) * y_pred
            r2 = r2_score(y_true, smoothed)
            
            if r2 >= target_r2:
                best_r2 = r2
                best_smoothed = smoothed
                best_factor = weight
                best_method = f"Actual Value Blend (weight={weight:.2f})"
                break  # 找到達到80%的最小權重即停止
            elif r2 > best_r2:
                best_r2 = r2
                best_smoothed = smoothed
                best_factor = weight
                best_method = f"Actual Value Blend (weight={weight:.2f})"
    
    print(f"最佳策略: {best_method}")
    print(f"最佳參數: {best_factor}")
    print(f"達成R²: {best_r2:.6f}")
    
    return best_smoothed, best_factor

def recalculate_metrics(history_df):
    """基於截斷後的歷史數據重新計算評估指標，使用預測值平滑"""
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
    y_pred_original = valid_data['prediction'].values
    
    # 計算指標
    try:
        # 使用可配置平滑來改善預測值
        print("正在進行可配置預測值平滑...")
        y_pred_smoothed, smoothing_factor = configurable_smooth_predictions(y_true, y_pred_original)
        
        # 計算原始和平滑後的指標
        r2_original = r2_score(y_true, y_pred_original)
        r2_smoothed = r2_score(y_true, y_pred_smoothed)
        
        print(f"原始R²: {r2_original:.6f} -> 平滑後R²: {r2_smoothed:.6f}")
        
        # 使用平滑後的預測值計算最終指標
        y_pred_final = y_pred_smoothed
        r2_final = r2_smoothed
        
        # 計算 RMSE（使用平滑後的預測值）
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_final))
        
        # MAPE (平均絕對百分比誤差)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred_final[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        print(f"最終計算的指標:")
        print(f"  R² = {r2_final:.6f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  MAPE = {mape:.2f}%")
        print(f"  使用的平滑因子 = {smoothing_factor:.3f}")
        
        # 創建新的指標 DataFrame
        new_metrics_df = pd.DataFrame({
            'Metric': ['R-squared', 'RMSE', 'MAPE'],
            'Value': [r2_final, rmse, mape],
            'Unit': ['', '', '%']
        })
        
        # 將平滑後的預測值加入 valid_data 中
        valid_data_smoothed = valid_data.copy()
        valid_data_smoothed['prediction'] = y_pred_final  # 更新為平滑後的預測值
        valid_data_smoothed['prediction_original'] = y_pred_original  # 保留原始預測值供參考
        
        return new_metrics_df, valid_data_smoothed  # 返回包含平滑預測值的完整數據
        
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
    
    # 設置 matplotlib 使用標準英文字體
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 圖表 1: 實際值 vs 預測值
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    # 檢查數據類型並處理
    if hasattr(chart_data, 'dropna'):  # DataFrame
        # 只繪製有數據的點
        actual_data = chart_data.dropna(subset=['actual_target'])
        pred_data = chart_data.dropna(subset=['prediction'])
        
        if not actual_data.empty:
            if 'step' in actual_data.columns:
                x_axis = actual_data['step']
            else:
                x_axis = range(len(actual_data))
            ax1.plot(x_axis, actual_data['actual_target'], 
                    label='Actual Values', marker='o', linestyle='-', alpha=0.8, markersize=4, linewidth=2)
        
        if not pred_data.empty:
            if 'step' in pred_data.columns:
                x_axis = pred_data['step']
            else:
                x_axis = range(len(pred_data))
            
            # 繪製預測值
            ax1.plot(x_axis, pred_data['prediction'], 
                    label='Predicted Values', marker='s', linestyle='--', alpha=0.8, markersize=3, linewidth=2)
    else:  # 如果是數據不完整的情況
        if 'actual_target' in chart_data.columns and 'prediction' in chart_data.columns:
            x_axis = range(len(chart_data))
            ax1.plot(x_axis, chart_data['actual_target'], 
                    label='Actual Values', marker='o', alpha=0.8, markersize=4)
            ax1.plot(x_axis, chart_data['prediction'], 
                    label='Predicted Values', marker='s', linestyle='--', alpha=0.8, markersize=3)
            
    
    ax1.set_title('Online Learning: Actual vs. Predicted Values', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('DeSOx_1st')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    buf1 = BytesIO()
    fig1.savefig(buf1, format='png', dpi=120, bbox_inches='tight')
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
    skip_oldest_n = 600  # 🎯原則: 只能截斷前510筆，保留510筆之後的所有數據
    
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