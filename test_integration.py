#!/usr/bin/env python3
"""
CFB 線上學習系統 - 整合測試腳本

此腳本用於測試業主 PI Server 代碼與線上學習系統的整合是否成功。
執行完整的端到端測試，確保所有組件正常運作。

使用方法:
    python test_integration.py

作者: Rovo Dev
日期: 2024-09-11
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """測試所有必要模組的載入"""
    print("🔍 測試 1: 模組載入測試")
    print("-" * 40)
    
    test_results = {}
    
    # 測試基本套件
    try:
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        test_results['基本套件'] = "✅ 通過"
        print("✅ 基本套件 (pandas, numpy, xgboost) 載入成功")
    except ImportError as e:
        test_results['基本套件'] = f"❌ 失敗: {e}"
        print(f"❌ 基本套件載入失敗: {e}")
    
    # 測試線上學習模組
    try:
        from Y1.inference import OnlinePredictor as Y1Predictor
        from Y2.inference import OnlinePredictor as Y2Predictor
        test_results['線上學習模組'] = "✅ 通過"
        print("✅ 線上學習模組 (Y1, Y2) 載入成功")
    except ImportError as e:
        test_results['線上學習模組'] = f"❌ 失敗: {e}"
        print(f"❌ 線上學習模組載入失敗: {e}")
    
    # 測試業主 PI Server 模組
    try:
        from TEST0911.crawl_pi_server import get_summary_values, async_get_values, get_tag_snapshot
        test_results['PI Server 模組'] = "✅ 通過"
        print("✅ 業主 PI Server 模組載入成功")
    except ImportError as e:
        test_results['PI Server 模組'] = f"⚠️ 模擬模式: {e}"
        print(f"⚠️ 業主 PI Server 模組載入失敗，將使用模擬模式: {e}")
    
    # 測試整合橋接器
    try:
        from integration_bridge import CFBIntegrationBridge
        test_results['整合橋接器'] = "✅ 通過"
        print("✅ 整合橋接器載入成功")
    except ImportError as e:
        test_results['整合橋接器'] = f"❌ 失敗: {e}"
        print(f"❌ 整合橋接器載入失敗: {e}")
    
    # 測試配置模組
    try:
        from config import SystemConfig, CFBTagMapping
        test_results['配置模組'] = "✅ 通過"
        print("✅ 配置模組載入成功")
    except ImportError as e:
        test_results['配置模組'] = f"❌ 失敗: {e}"
        print(f"❌ 配置模組載入失敗: {e}")
    
    return test_results

def test_model_files():
    """測試模型檔案存在性"""
    print("\n🔍 測試 2: 模型檔案檢查")
    print("-" * 40)
    
    test_results = {}
    
    # 檢查 Y1 模型
    y1_model_path = "Y1/model/xgb_model.json"
    if os.path.exists(y1_model_path):
        test_results['Y1 模型'] = "✅ 存在"
        print(f"✅ Y1 模型檔案存在: {y1_model_path}")
    else:
        test_results['Y1 模型'] = "⚠️ 不存在"
        print(f"⚠️ Y1 模型檔案不存在: {y1_model_path}")
    
    # 檢查 Y2 模型
    y2_model_path = "Y2/model/xgb_model_y2.json"
    if os.path.exists(y2_model_path):
        test_results['Y2 模型'] = "✅ 存在"
        print(f"✅ Y2 模型檔案存在: {y2_model_path}")
    else:
        test_results['Y2 模型'] = "⚠️ 不存在"
        print(f"⚠️ Y2 模型檔案不存在: {y2_model_path}")
    
    # 檢查特徵檔案
    features1_path = "Y1/features1.pkl"
    if os.path.exists(features1_path):
        test_results['Y1 特徵'] = "✅ 存在"
        print(f"✅ Y1 特徵檔案存在: {features1_path}")
    else:
        test_results['Y1 特徵'] = "❌ 不存在"
        print(f"❌ Y1 特徵檔案不存在: {features1_path}")
    
    features2_path = "Y2/features2.pkl"
    if os.path.exists(features2_path):
        test_results['Y2 特徵'] = "✅ 存在"
        print(f"✅ Y2 特徵檔案存在: {features2_path}")
    else:
        test_results['Y2 特徵'] = "❌ 不存在"
        print(f"❌ Y2 特徵檔案不存在: {features2_path}")
    
    return test_results

def test_predictor_initialization():
    """測試預測器初始化"""
    print("\n🔍 測試 3: 預測器初始化")
    print("-" * 40)
    
    test_results = {}
    
    # 測試 Y1 預測器
    try:
        from Y1.inference import OnlinePredictor as Y1Predictor
        y1_predictor = Y1Predictor()
        test_results['Y1 預測器初始化'] = "✅ 成功"
        print("✅ Y1 預測器初始化成功")
    except Exception as e:
        test_results['Y1 預測器初始化'] = f"❌ 失敗: {e}"
        print(f"❌ Y1 預測器初始化失敗: {e}")
    
    # 測試 Y2 預測器
    try:
        from Y2.inference import OnlinePredictor as Y2Predictor
        y2_predictor = Y2Predictor()
        test_results['Y2 預測器初始化'] = "✅ 成功"
        print("✅ Y2 預測器初始化成功")
    except Exception as e:
        test_results['Y2 預測器初始化'] = f"❌ 失敗: {e}"
        print(f"❌ Y2 預測器初始化失敗: {e}")
    
    return test_results

def test_integration_bridge():
    """測試整合橋接器"""
    print("\n🔍 測試 4: 整合橋接器功能")
    print("-" * 40)
    
    test_results = {}
    
    try:
        from integration_bridge import CFBIntegrationBridge
        
        # 初始化橋接器
        bridge = CFBIntegrationBridge(cfb_unit=2)
        test_results['橋接器初始化'] = "✅ 成功"
        print("✅ 整合橋接器初始化成功")
        
        # 測試模擬數據生成
        mock_data = bridge._generate_mock_data()
        if not mock_data.empty:
            test_results['模擬數據生成'] = "✅ 成功"
            print(f"✅ 模擬數據生成成功 ({len(mock_data)} 筆數據)")
        else:
            test_results['模擬數據生成'] = "❌ 失敗"
            print("❌ 模擬數據生成失敗")
        
        # 測試單次預測
        if not mock_data.empty:
            current_data = mock_data.iloc[[-1]]
            predictions = bridge.run_online_prediction(current_data)
            test_results['單次預測'] = "✅ 成功"
            print(f"✅ 單次預測成功: {predictions}")
        else:
            test_results['單次預測'] = "❌ 跳過"
            print("❌ 無數據，跳過單次預測測試")
            
    except Exception as e:
        test_results['橋接器測試'] = f"❌ 失敗: {e}"
        print(f"❌ 整合橋接器測試失敗: {e}")
        traceback.print_exc()
    
    return test_results

def test_data_processing():
    """測試數據處理功能"""
    print("\n🔍 測試 5: 數據處理功能")
    print("-" * 40)
    
    test_results = {}
    
    try:
        from integration_bridge import CFBIntegrationBridge
        from config import CFBTagMapping
        
        bridge = CFBIntegrationBridge(cfb_unit=2)
        
        # 創建測試數據
        tags = CFBTagMapping.CFB2_TAGS
        test_data = {}
        for tag in list(tags.keys())[:10]:  # 只取前 10 個標籤進行測試
            if 'TE-' in tag:  # 溫度
                test_data[tag] = [850.0, 860.0, 840.0]
            elif 'FIQ-' in tag or 'FT-' in tag:  # 流量
                test_data[tag] = [50.0, 55.0, 48.0]
            elif 'AIC-' in tag or 'AT-' in tag:  # 濃度
                test_data[tag] = [100.0, 105.0, 95.0]
            else:
                test_data[tag] = [10.0, 12.0, 8.0]
        
        # 添加必要的計算欄位
        test_data['MLUT4_FIQ-2BTCF'] = [50.0, 55.0, 48.0]  # 總飼煤量
        test_data['MLUT4_FQ-239'] = [1000.0, 1100.0, 950.0]  # 煙氣流量
        test_data['MLUT4_RQ-2BTLS'] = [20.0, 22.0, 18.0]  # 石灰石
        test_data['MLUT4_AIC-232B'] = [100.0, 105.0, 95.0]  # ECO SOx
        test_data['MLUT4_AT-240'] = [80.0, 85.0, 75.0]  # 煙囪 SOx
        
        df = pd.DataFrame(test_data)
        processed_df = bridge._process_raw_data(df)
        
        # 檢查計算欄位是否存在
        required_cols = ['前爐SOx濃度', '鈣硫比', 'DeSOx_1st', 'DeSOx_2nd']
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        
        if not missing_cols:
            test_results['數據處理'] = "✅ 成功"
            print("✅ 數據處理功能正常，所有計算欄位都已生成")
        else:
            test_results['數據處理'] = f"⚠️ 部分成功，缺少: {missing_cols}"
            print(f"⚠️ 數據處理部分成功，缺少欄位: {missing_cols}")
        
    except Exception as e:
        test_results['數據處理'] = f"❌ 失敗: {e}"
        print(f"❌ 數據處理測試失敗: {e}")
        traceback.print_exc()
    
    return test_results

def test_configuration():
    """測試配置模組"""
    print("\n🔍 測試 6: 配置模組")
    print("-" * 40)
    
    test_results = {}
    
    try:
        from config import SystemConfig, CFBTagMapping, validate_config, create_directories
        
        # 測試配置載入
        config = SystemConfig()
        test_results['配置載入'] = "✅ 成功"
        print("✅ 配置載入成功")
        
        # 測試標籤對應
        cfb1_tags = CFBTagMapping.CFB1_TAGS
        cfb2_tags = CFBTagMapping.CFB2_TAGS
        
        if len(cfb1_tags) > 0 and len(cfb2_tags) > 0:
            test_results['標籤對應'] = "✅ 成功"
            print(f"✅ 標籤對應正常 (CFB1: {len(cfb1_tags)}, CFB2: {len(cfb2_tags)})")
        else:
            test_results['標籤對應'] = "❌ 失敗"
            print("❌ 標籤對應異常")
        
        # 測試配置驗證
        validate_config()
        test_results['配置驗證'] = "✅ 成功"
        print("✅ 配置驗證通過")
        
        # 測試目錄創建
        create_directories()
        test_results['目錄創建'] = "✅ 成功"
        print("✅ 目錄創建成功")
        
    except Exception as e:
        test_results['配置測試'] = f"❌ 失敗: {e}"
        print(f"❌ 配置模組測試失敗: {e}")
        traceback.print_exc()
    
    return test_results

def generate_test_report(all_results):
    """生成測試報告"""
    print("\n" + "=" * 60)
    print("📋 整合測試總結報告")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, results in all_results.items():
        print(f"\n📊 {test_name}:")
        for item, status in results.items():
            print(f"  {item}: {status}")
            total_tests += 1
            if status.startswith("✅"):
                passed_tests += 1
    
    print(f"\n🎯 測試統計:")
    print(f"  總測試項目: {total_tests}")
    print(f"  通過項目: {passed_tests}")
    print(f"  通過率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有測試通過！系統整合成功！")
        print("✅ 您可以開始使用整合系統了")
    elif passed_tests >= total_tests * 0.8:
        print("\n⚠️ 大部分測試通過，系統基本可用")
        print("💡 建議檢查並修復失敗的項目")
    else:
        print("\n❌ 多個測試失敗，需要修復問題")
        print("🔧 請參考 INTEGRATION_GUIDE.md 進行故障排除")
    
    # 儲存測試報告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"integration_test_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("CFB 線上學習系統 - 整合測試報告\n")
        f.write("=" * 50 + "\n")
        f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for test_name, results in all_results.items():
            f.write(f"{test_name}:\n")
            for item, status in results.items():
                f.write(f"  {item}: {status}\n")
            f.write("\n")
        
        f.write(f"測試統計:\n")
        f.write(f"  總測試項目: {total_tests}\n")
        f.write(f"  通過項目: {passed_tests}\n")
        f.write(f"  通過率: {passed_tests/total_tests*100:.1f}%\n")
    
    print(f"\n📁 詳細測試報告已儲存至: {report_file}")

def main():
    """主測試流程"""
    print("🏭 CFB 線上學習系統 - 整合測試")
    print("=" * 60)
    print("此測試將驗證業主 PI Server 代碼與線上學習系統的整合狀況")
    print("測試開始時間:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    all_results = {}
    
    # 執行各項測試
    all_results["模組載入測試"] = test_imports()
    all_results["模型檔案檢查"] = test_model_files()
    all_results["預測器初始化"] = test_predictor_initialization()
    all_results["整合橋接器功能"] = test_integration_bridge()
    all_results["數據處理功能"] = test_data_processing()
    all_results["配置模組"] = test_configuration()
    
    # 生成測試報告
    generate_test_report(all_results)

if __name__ == "__main__":
    main()