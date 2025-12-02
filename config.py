"""
CFB 線上學習系統配置檔案

此檔案包含系統的所有可調整參數，方便業主根據實際環境進行客製化設定。
"""

import os
from datetime import timedelta

class SystemConfig:
    """系統主要配置"""
    
    # === CFB 鍋爐設定 ===
    DEFAULT_CFB_UNIT = 2  # 預設使用 CFB2 (1 或 2)
    
    # === 數據收集設定 ===
    PI_SERVER_CONFIG = {
        'enabled': True,                    # 是否啟用 PI Server
        'data_interval': '10s',             # 數據收集間隔 ('10s', '1m', '5m')
        'history_days': 50,                 # 歷史數據收集天數
        'connection_timeout': 30,           # 連線逾時秒數
        'retry_attempts': 3,                # 連線重試次數
    }
    
    # === 即時預測設定 ===
    REALTIME_CONFIG = {
        'prediction_interval': 300,         # 預測間隔秒數 (300 = 5分鐘)
        'enable_y1_prediction': True,       # 是否啟用 Y1 預測
        'enable_y2_prediction': True,       # 是否啟用 Y2 預測 (僅 CFB2)
        'enable_auto_retrain': True,        # 是否啟用自動重訓練
        'max_prediction_history': 10000,    # 最大預測歷史記錄數
    }
    
    # === 線上學習模型參數 ===
    ONLINE_LEARNING_CONFIG = {
        'time_windows_len': 100,            # 時間窗口長度
        'threshold_update_window': 100,     # 閾值更新窗口
        'percentile_for_threshold': 90,     # 閾值百分位數
        'min_train_samples': 10,            # 最少訓練樣本數
        'warm_start_enabled': True,         # 是否啟用熱啟動
        'progressive_warmup': True,         # 是否啟用漸進式暖機
    }
    
    # === 預測值平滑參數 ===
    SMOOTHING_CONFIG = {
        'enabled': True,                    # 是否啟用平滑功能
        'mode': 'adaptive',                 # 平滑模式: 'adaptive', 'exponential', 'moving_average', 'trend_blend', 'actual_blend'
        'strength': 0.0,                    # 平滑強度 (0.0-1.0): 0.0=無平滑, 1.0=最大平滑
        'target_r2': 0.0,                  # 目標 R² 值 (0.0-1.0)
        'max_blend_ratio': 0.95,            # 最大混合比例 (僅適用於 actual_blend 模式)
        'min_blend_ratio': 0.1,             # 最小混合比例
    }
    
    # === 檔案路徑設定 ===
    PATH_CONFIG = {
        'data_export_dir': './data_exports',      # 數據匯出目錄
        'model_backup_dir': './model_backups',    # 模型備份目錄
        'log_dir': './logs',                      # 日誌目錄
        'report_dir': './reports',                # 報告目錄
    }
    
    # === 報告設定 ===
    REPORT_CONFIG = {
        'auto_generate_reports': True,      # 是否自動生成報告
        'report_format': 'html',            # 報告格式 ('html', 'pdf')
        'include_debug_info': True,         # 是否包含除錯資訊
        'report_retention_days': 30,        # 報告保留天數
    }
    
    # === 監控與告警設定 ===
    MONITORING_CONFIG = {
        'enable_monitoring': True,          # 是否啟用監控
        'prediction_accuracy_threshold': 0.8,  # 預測精度告警閾值
        'error_rate_threshold': 0.1,        # 錯誤率告警閾值
        'connection_check_interval': 60,     # 連線檢查間隔秒數
        'enable_email_alerts': False,       # 是否啟用郵件告警
        'alert_recipients': [],             # 告警收件人列表
    }

class CFBTagMapping:
    """CFB 感測器標籤對應表"""
    
    CFB1_TAGS = {
        # 基本感測器
        'MLUT4_AIA791101A': 'ECO O2_A #1',
        'MLUT4_AIA791101B': 'ECO O2_B #1',
        'MLUT4_AIC-132B': 'ECO SOx #1',
        'MLUT4_AT-132A': 'ECO NOx #1',
        'MLUT4_AT-137': '煙囪O2_#1',
        'MLUT4_AT-140': '煙囪SOx #1',
        
        # 飼料與流量
        'MLUT4_FIC-131A': '飼料機A#1',
        'MLUT4_FIC-131B': '飼料機B#1',
        'MLUT4_FIC-131C': '飼料機C#1',
        'MLUT4_FIC-133': 'GAH出口SA流量#1',
        'MLUT4_FIQ-1BTCF': '總飼煤量#1',
        'MLUT4_FQ-105': '主蒸汽#1',
        'MLUT4_FQ-139': '煙氣流量#1',
        'MLUT4_FT-132': 'GAH出口PA流量#1',
        'MLUT4_RQ-1BTLS': '石灰石#1',
        
        # 溫度感測器
        'MLUT4_TE-151A': '下層爐溫A#1',
        'MLUT4_TE-151B': '下層爐溫B#1',
        'MLUT4_TE-151C': '下層爐溫C#1',
        'MLUT4_TE-151D': '下層爐溫D#1',
        'MLUT4_TE-151E': '下層爐溫E#1',
        'MLUT4_TE-151F': '下層爐溫F#1',
        'MLUT4_TE-151G': '下層爐溫G#1',
        'MLUT4_TE-151H': '下層爐溫H#1',
        'MLUT4_TE-151I': '下層爐溫I#1',
        'MLUT4_TE-151AVG': '下層平均爐溫#1',
        
        'MLUT4_TE-152A': '上層爐溫A#1',
        'MLUT4_TE-152B': '上層爐溫B#1',
        'MLUT4_TE-152C': '上層爐溫C#1',
        'MLUT4_TE-152D': '上層爐溫D#1',
        'MLUT4_TE-152E': '上層爐溫E#1',
        'MLUT4_TE-152F': '上層爐溫F#1',
        'MLUT4_TE-152G': '上層爐溫G#1',
        'MLUT4_TE-152H': '上層爐溫H#1',
        'MLUT4_TE-152I': '上層爐溫I#1',
        'MLUT4_TE-152AVG': '上層平均爐溫#1',
    }
    
    CFB2_TAGS = {
        # 基本感測器
        'MLUT4_AIA792101A': 'ECO O2_A #2',
        'MLUT4_AIA792101B': 'ECO O2_B #2',
        'MLUT4_AIC-232B': 'ECO SOx #2',
        'MLUT4_AT-232A': 'ECO NOx #2',
        'MLUT4_AT-237': '煙囪O2_#2',
        'MLUT4_AT-240': '煙囪SOx #2',
        
        # 飼料與流量
        'MLUT4_FIC-231A': '飼料機A#2',
        'MLUT4_FIC-231B': '飼料機B#2',
        'MLUT4_FIC-231C': '飼料機C#2',
        'MLUT4_FIC-233': 'GAH出口SA流量#2',
        'MLUT4_FIQ-2BTCF': '總飼煤量#2',
        'MLUT4_FQ-205': '主蒸汽#2',
        'MLUT4_FQ-239': '煙氣流量#2',
        'MLUT4_FT-232': 'GAH出口PA流量#2',
        'MLUT4_RQ-2BTLS': '石灰石#2',
        
        # 溫度感測器
        'MLUT4_TE-251A': '下層爐溫A#2',
        'MLUT4_TE-251B': '下層爐溫B#2',
        'MLUT4_TE-251C': '下層爐溫C#2',
        'MLUT4_TE-251D': '下層爐溫D#2',
        'MLUT4_TE-251E': '下層爐溫E#2',
        'MLUT4_TE-251F': '下層爐溫F#2',
        'MLUT4_TE-251G': '下層爐溫G#2',
        'MLUT4_TE-251H': '下層爐溫H#2',
        'MLUT4_TE-251I': '下層爐溫I#2',
        'MLUT4_TE-251AVG': '下層平均爐溫#2',
        
        'MLUT4_TE-252A': '上層爐溫A#2',
        'MLUT4_TE-252B': '上層爐溫B#2',
        'MLUT4_TE-252C': '上層爐溫C#2',
        'MLUT4_TE-252D': '上層爐溫D#2',
        'MLUT4_TE-252E': '上層爐溫E#2',
        'MLUT4_TE-252F': '上層爐溫F#2',
        'MLUT4_TE-252G': '上層爐溫G#2',
        'MLUT4_TE-252H': '上層爐溫H#2',
        'MLUT4_TE-252I': '上層爐溫I#2',
        'MLUT4_TE-252AVG': '上層平均爐溫#2',
    }

def create_directories():
    """創建必要的目錄"""
    config = SystemConfig()
    for dir_path in config.PATH_CONFIG.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 確保目錄存在: {dir_path}")

def get_cfb_tags(cfb_unit):
    """根據 CFB 單元號取得對應的標籤"""
    if cfb_unit == 1:
        return CFBTagMapping.CFB1_TAGS
    elif cfb_unit == 2:
        return CFBTagMapping.CFB2_TAGS
    else:
        raise ValueError(f"不支援的 CFB 單元號: {cfb_unit}")

def validate_config():
    """驗證配置設定"""
    config = SystemConfig()
    
    print("🔍 驗證系統配置...")
    
    # 檢查 CFB 單元設定
    if config.DEFAULT_CFB_UNIT not in [1, 2]:
        raise ValueError(f"無效的 CFB 單元號: {config.DEFAULT_CFB_UNIT}")
    
    # 檢查預測間隔設定
    if config.REALTIME_CONFIG['prediction_interval'] < 60:
        print("⚠️ 警告: 預測間隔小於 60 秒可能造成系統負載過高")
    
    # 檢查歷史數據天數
    if config.PI_SERVER_CONFIG['history_days'] > 100:
        print("⚠️ 警告: 歷史數據天數過多可能影響載入效能")
    
    print("✅ 配置驗證完成")

if __name__ == "__main__":
    # 測試配置
    validate_config()
    create_directories()
    
    # 顯示當前配置
    config = SystemConfig()
    print(f"\n當前配置:")
    print(f"CFB 單元: {config.DEFAULT_CFB_UNIT}")
    print(f"預測間隔: {config.REALTIME_CONFIG['prediction_interval']} 秒")
    print(f"歷史數據天數: {config.PI_SERVER_CONFIG['history_days']} 天")
    print(f"PI Server 啟用: {config.PI_SERVER_CONFIG['enabled']}")