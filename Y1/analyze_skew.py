import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 設定檔案路徑 ---
TRAIN_DEBUG_FILE = "train_debug_data.csv"
INFERENCE_DEBUG_FILE = "inference_debug_data.csv"
OUTPUT_PLOT_FILE = "skew_analysis_plot.png"

def analyze():
    print("--- 開始執行訓練/服務歪斜分析 ---" )

    # --- 1. 讀取偵錯檔案 ---
    if not os.path.exists(TRAIN_DEBUG_FILE) or not os.path.exists(INFERENCE_DEBUG_FILE):
        print(f"錯誤: 找不到必要的偵錯檔案。請確認 {TRAIN_DEBUG_FILE} 和 {INFERENCE_DEBUG_FILE} 都存在。")
        return

    print("正在讀取偵錯檔案...")
    train_df = pd.read_csv(TRAIN_DEBUG_FILE)
    inference_df = pd.read_csv(INFERENCE_DEBUG_FILE)

    # --- 2. 資料準備與欄位對齊 ---
    print("正在準備與對齊資料...")
    # 移除推論日誌中 'feature_' 的前綴
    inference_features_df = inference_df.filter(like='feature_')
    inference_features_df.columns = [col.replace('feature_', '') for col in inference_features_df.columns]
    
    # 取得共通的特徵欄位
    common_features = list(set(train_df.columns) & set(inference_features_df.columns))
    if not common_features:
        print("錯誤: 找不到共通的特徵欄位進行比較。")
        return
    
    print(f"找到 {len(common_features)} 個共通特徵進行比較。")

    # --- 3. 統計數據比較 ---
    print("\n--- 特徵分佈統計比較 ---")
    comparison_stats = []
    for feature in common_features:
        train_stats = train_df[feature].describe()
        inference_stats = inference_features_df[feature].describe()
        
        stats_dict = {
            'feature': feature,
            'train_mean': train_stats['mean'],
            'inference_mean': inference_stats['mean'],
            'train_std': train_stats['std'],
            'inference_std': inference_stats['std'],
            'train_min': train_stats['min'],
            'inference_min': inference_stats['min'],
            'train_max': train_stats['max'],
            'inference_max': inference_stats['max'],
        }
        comparison_stats.append(stats_dict)
    
    stats_df = pd.DataFrame(comparison_stats).set_index('feature')
    print(stats_df.to_string(float_format="%.4f"))

    # --- 4. 視覺化比較 (以 prev_target 為例) ---
    feature_to_plot = 'prev_target'
    if feature_to_plot not in common_features:
        print(f"\n警告: 找不到關鍵特徵 '{feature_to_plot}' 來進行視覺化比較。")
    else:
        print(f"\n正在生成關鍵特徵 '{feature_to_plot}' 的分佈比較圖...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sns.kdeplot(train_df[feature_to_plot], ax=ax, label='Train Data Distribution', fill=True, alpha=0.5)
        sns.kdeplot(inference_features_df[feature_to_plot], ax=ax, label='Inference Data Distribution', fill=True, alpha=0.5)
        
        ax.set_title(f"Distribution Comparison for Feature: '{feature_to_plot}'", fontsize=16)
        ax.set_xlabel("Feature Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend()
        
        plt.savefig(OUTPUT_PLOT_FILE)
        print(f"比較圖已儲存至: {OUTPUT_PLOT_FILE}")
        plt.close(fig)

    print("\n--- 分析完成 ---" )

if __name__ == '__main__':
    analyze()
