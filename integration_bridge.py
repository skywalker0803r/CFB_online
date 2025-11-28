#!/usr/bin/env python3
"""
CFB 線上學習系統 - 業主數據整合橋接腳本

此腳本整合業主工程師的 PI Server 數據收集代碼與您的線上學習預測系統。
功能：
1. 從 PI Server 收集即時數據
2. 轉換為您的系統所需的 Feather 格式
3. 觸發線上學習預測流程
4. 生成整合報告

作者: Rovo Dev
日期: 2024-09-11
"""

import pandas as pd
import numpy as np
import datetime
import os
import time
import warnings
from pathlib import Path

# 嘗試匯入業主的 PI Server 模組
try:
    from TEST0911.crawl_pi_server import get_summary_values, async_get_values, get_tag_snapshot
    PI_SERVER_AVAILABLE = True
    print("✅ 成功載入 PI Server 模組")
except ImportError:
    PI_SERVER_AVAILABLE = False
    print("⚠️ 無法載入 PI Server 模組，將使用模擬數據模式")

# 匯入您的線上學習模組
try:
    from Y1.inference import OnlinePredictor as Y1Predictor
    from Y2.inference import OnlinePredictor as Y2Predictor
    print("✅ 成功載入線上學習預測器")
except ImportError as e:
    print(f"❌ 無法載入線上學習模組: {e}")
    exit(1)

warnings.filterwarnings('ignore')

class CFBIntegrationBridge:
    """CFB 線上學習系統整合橋接器"""
    
    def __init__(self, cfb_unit=2, data_interval='10s', history_days=50):
        """
        初始化整合橋接器
        
        Args:
            cfb_unit (int): CFB 鍋爐編號 (1 或 2)
            data_interval (str): 數據收集間隔 ('10s', '1m', '5m')
            history_days (int): 歷史數據天數
        """
        self.cfb_unit = cfb_unit
        self.data_interval = data_interval
        self.history_days = history_days
        
        # 初始化預測器
        if cfb_unit == 1:
            self.predictor_y1 = Y1Predictor()
            self.predictor_y2 = None
            self.tag_dict = self._get_cfb1_tags()
            self.target_col_y1 = 'DeSOx_1st'
            print(f"🏭 初始化 CFB#{cfb_unit} 系統 (僅 Y1 預測)")
        elif cfb_unit == 2:
            self.predictor_y1 = Y1Predictor()  # CFB2 也可能需要 Y1 預測
            self.predictor_y2 = Y2Predictor()
            self.tag_dict = self._get_cfb2_tags()
            self.target_col_y1 = 'DeSOx_1st'
            self.target_col_y2 = 'DeSOx_2nd'
            print(f"🏭 初始化 CFB#{cfb_unit} 系統 (Y1 + Y2 預測)")
        else:
            raise ValueError("CFB 單元編號必須是 1 或 2")
            
        self.last_prediction_time = None
        self.prediction_history = []
        
    def _get_cfb1_tags(self):
        """取得 CFB1 的感測器標籤字典"""
        return {
            'MLUT4_AIA791101A':'ECO O2_A #1',
            'MLUT4_AIA791101B':'ECO O2_B #1',
            'MLUT4_AIC-132B': 'ECO SOx #1',
            'MLUT4_AT-132A': 'ECO NOx #1',
            'MLUT4_AT-137': '煙囪O2_#1',
            'MLUT4_AT-140': '煙囪SOx #1',
            'MLUT4_FIC-131A': '飼料機A#1',
            'MLUT4_FIC-131B': '飼料機B#1',
            'MLUT4_FIC-131C': '飼料機C#1',
            'MLUT4_FIC-133': 'GAH出口SA流量#1',
            'MLUT4_FIQ-1BTCF': '總飼煤量#1',
            'MLUT4_FQ-105': '主蒸汽#1',
            'MLUT4_FQ-139': '煙氣流量#1',
            'MLUT4_FT-132': 'GAH出口PA流量#1',
            'MLUT4_RQ-1BTLS': '石灰石#1',
            # ... 可依需要添加更多標籤
        }
    
    def _get_cfb2_tags(self):
        """取得 CFB2 的感測器標籤字典"""
        return {
            # Original Tags
            'MLUT4_AIA792101A':'ECO O2_A #2',
            'MLUT4_AIA792101B':'ECO O2_B #2',
            'MLUT4_AIC-232B': 'ECO SOx #2',
            'MLUT4_AT-232A': 'ECO NOx #2',
            'MLUT4_AT-237': '煙囪O2_#2',
            'MLUT4_AT-240': '煙囪SOx #2',
            'MLUT4_FIC-231A': '飼料機A#2',
            'MLUT4_FIC-231B': '飼料機B#2',
            'MLUT4_FIC-231C': '飼料機C#2',
            'MLUT4_FIC-233': 'GAH出口SA流量#2',
            'MLUT4_FIQ-2BTCF': '總飼煤量#2',
            'MLUT4_FQ-205': '主蒸汽#2',
            'MLUT4_FQ-239': '煙氣流量#2',
            'MLUT4_FT-232': 'GAH出口PA流量#2',
            'MLUT4_RQ-2BTLS': '石灰石#2',
            
            # Added from Y1 Error
            'MLUT4_TE-252D': 'MLUT4_TE-252D',
            'MLUT4_TE-252F': 'MLUT4_TE-252F',
            'MLUT4_PIC-233': 'MLUT4_PIC-233',
            'MLUT4_TE-252G': 'MLUT4_TE-252G',
            'MLUT4_TE-252I': 'MLUT4_TE-252I',
            'MLUT4_IT-E7925B': 'MLUT4_IT-E7925B',
            'MLUT4_ZT-232': 'MLUT4_ZT-232',
            'MLUT4_FT-792020': 'MLUT4_FT-792020',
            'MLUT4_IT-E7925A': 'MLUT4_IT-E7925A',
            'MLUT4_TE-251A': 'MLUT4_TE-251A',
            'MLUT4_PDT-232': 'MLUT4_PDT-232',
            'MLUT4_TE-251G': 'MLUT4_TE-251G',
            'MLUT4_TE-251D': 'MLUT4_TE-251D',
            'MLUT4_TE-251F': 'MLUT4_TE-251F',
            'MLUT4_TE-251H': 'MLUT4_TE-251H',

            # Added from Y2 Error
            'MLUT4_FT-957': 'MLUT4_FT-957',
            'MLUT4_FT-956': 'MLUT4_FT-956',
            'MLUT4_RT-091': 'MLUT4_RT-091',
            'MLUT4_PT-231': 'MLUT4_PT-231',
            'MLUT4_FT-V004': 'MLUT4_FT-V004',
        }
    
    def collect_pi_data(self, realtime=True):
        """
        從 PI Server 收集數據
        
        Args:
            realtime (bool): 是否為即時模式
            
        Returns:
            pd.DataFrame: 處理後的數據
        """
        if not PI_SERVER_AVAILABLE:
            print("⚠️ PI Server 不可用，使用模擬數據")
            return self._generate_mock_data()
        
        try:
            if realtime:
                # 即時模式：獲取最新的單筆數據
                print("📡 從 PI Server 收集即時數據...")
                snapshot_data = {}
                for tag in self.tag_dict.keys():
                    # 假設 get_tag_snapshot 返回一個包含單個值或(值, 時間戳)元組的結果
                    # 我們只取值的部分，並將其放入列表中以創建 DataFrame
                    try:
                        # 由於不確定返回格式，我們做一個靈活的處理
                        result = get_tag_snapshot(tag)
                        if isinstance(result, (dict,)):
                            # 如果返回字典，取第一個值
                            value = next(iter(result.values()), None)
                        elif isinstance(result, (list, tuple)) and len(result) > 2:
                            # 如果返回列表或元組，取第三個元素 (數值)
                            value = result[2]
                        else:
                            # 否則直接使用返回值
                            value = result
                        
                        # PI SDK 可能返回帶有時間戳的 AFValue 對象，我們需要提取其值
                        if hasattr(value, 'Value'):
                            snapshot_data[tag] = [value.Value]
                        else:
                            snapshot_data[tag] = [value]
                    except Exception as tag_error:
                        print(f"⚠️ 讀取標籤 '{tag}' 失敗: {tag_error}")
                        snapshot_data[tag] = [None]
                data = snapshot_data
            else:
                # 歷史模式：獲取指定時間範圍的數據
                end_time = datetime.datetime.now()
                start_time = end_time - datetime.timedelta(days=self.history_days)
                
                start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
                end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"📡 從 PI Server 收集歷史數據 ({start_time_str} ~ {end_time_str})...")
                data = async_get_values(
                    list(self.tag_dict.keys()),
                    timestart=start_time_str,
                    timeend=end_time_str,
                    interval=self.data_interval
                )
            
            # 數據處理
            df = pd.DataFrame(data)
            df = self._process_raw_data(df)
            
            print(f"✅ 成功收集 {len(df)} 筆數據")
            return df
            
        except Exception as e:
            print(f"❌ PI Server 數據收集失敗: {e}")
            return self._generate_mock_data()
    
    def _process_raw_data(self, df):
        """處理原始 PI 數據"""
        # 轉換數值型態
        for col in df.columns:
            if col in self.tag_dict:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 計算衍生指標（參考業主工程師的計算）
        if self.cfb_unit == 1:
            coal_flow = 'MLUT4_FIQ-1BTCF'
            gas_flow = 'MLUT4_FQ-139'
            limestone = 'MLUT4_RQ-1BTLS'
            eco_sox = 'MLUT4_AIC-132B'
            stack_sox = 'MLUT4_AT-140'
        else:  # CFB2
            coal_flow = 'MLUT4_FIQ-2BTCF'
            gas_flow = 'MLUT4_FQ-239'
            limestone = 'MLUT4_RQ-2BTLS'
            eco_sox = 'MLUT4_AIC-232B'
            stack_sox = 'MLUT4_AT-240'
        
        # 業主工程師的計算公式
        if all(col in df.columns for col in [coal_flow, gas_flow, limestone, eco_sox, stack_sox]):
            df['前爐SOx濃度'] = df[coal_flow] * 8 / df[gas_flow] * 24.5 * 1000 / 32.065
            df['鈣硫比'] = df[limestone] / df[coal_flow] * 0.32 / 0.008
            df['DeSOx_1st'] = (df['前爐SOx濃度'] - df[eco_sox]) / df['前爐SOx濃度']
            df['DeSOx_2nd'] = (df[eco_sox] - df[stack_sox]) / df[eco_sox]
        
        return df
    
    def _generate_mock_data(self):
        """生成模擬數據（當 PI Server 不可用時）"""
        print("🔧 生成模擬數據用於測試...")
        # 載入歷史數據作為模板
        try:
            if self.cfb_unit == 2:
                # 嘗試載入現有的 CFB2 測試數據
                test_data_file = 'test0609-CFB2脫硫劑優化改善.feather'
                if os.path.exists(test_data_file):
                    df = pd.read_feather(test_data_file).tail(100)
                    print(f"📁 載入歷史測試數據 {test_data_file}")
                    return df
        except:
            pass
        
        # 如果無法載入歷史數據，生成基本模擬數據
        np.random.seed(42)
        n_samples = 50
        
        mock_data = {}
        for tag in self.tag_dict.keys():
            if 'TE-' in tag:  # 溫度
                mock_data[tag] = np.random.normal(850, 50, n_samples)
            elif 'FIQ-' in tag or 'FT-' in tag:  # 流量
                mock_data[tag] = np.random.normal(50, 10, n_samples)
            elif 'AIC-' in tag or 'AT-' in tag:  # 濃度
                mock_data[tag] = np.random.normal(100, 20, n_samples)
            else:
                mock_data[tag] = np.random.normal(10, 2, n_samples)
        
        df = pd.DataFrame(mock_data)
        df = self._process_raw_data(df)
        return df
    
    def save_data_for_training(self, df, filename=None):
        """將數據儲存為 Feather 格式供訓練使用"""
        if filename is None:
            today = datetime.date.today().strftime('%Y%m%d')
            filename = f"{today}-CFB{self.cfb_unit}脫硫劑優化改善.feather"
        
        df.reset_index(drop=True).to_feather(filename)
        print(f"💾 數據已儲存至: {filename}")
        return filename
    
    def run_online_prediction(self, current_data, last_true_y1=None, last_true_y2=None):
        """執行線上預測"""
        results = {}
        
        # 提取當前特徵數據
        current_features = {}
        if not current_data.empty:
            # 記錄所有特徵值
            for col in current_data.columns:
                if col in current_data.columns:
                    current_features[f'feature_{col}'] = current_data[col].iloc[-1] if len(current_data) > 0 else None
        
        # Y1 預測 (DeSOx_1st)
        if self.predictor_y1:
            try:
                y1_pred = self.predictor_y1.predict_and_learn(current_data, last_true_y1)
                results['Y1_prediction'] = y1_pred
                results['Y1_true'] = current_data.get('DeSOx_1st', pd.Series([None])).iloc[-1] if not current_data.empty else None
                print(f"🎯 Y1 預測結果: {y1_pred:.4f}" if y1_pred else "⏳ Y1 模型尚未就緒")
            except Exception as e:
                print(f"❌ Y1 預測失敗: {e}")
                results['Y1_prediction'] = None
                results['Y1_true'] = None
        
        # Y2 預測 (DeSOx_2nd) - 僅 CFB2
        if self.predictor_y2 and self.cfb_unit == 2:
            try:
                y2_pred_dict = self.predictor_y2.predict_and_learn(current_data, last_true_y2)
                results['Y2_prediction'] = y2_pred_dict['DeSOx_2nd_pred']
                results['Y3_prediction'] = y2_pred_dict['Y3_pred']
                results['Y2_true'] = current_data.get('DeSOx_2nd', pd.Series([None])).iloc[-1] if not current_data.empty else None
                print(f"🎯 Y2 預測結果: {y2_pred_dict['DeSOx_2nd_pred']:.4f}" if y2_pred_dict['DeSOx_2nd_pred'] else "⏳ Y2 模型尚未就緒")
                print(f"🎯 Y3 反推結果: {y2_pred_dict['Y3_pred']:.4f}" if y2_pred_dict['Y3_pred'] else "⏳ Y3 計算尚未就緒")
            except Exception as e:
                print(f"❌ Y2/Y3 預測失敗: {e}")
                results['Y2_prediction'] = None
                results['Y3_prediction'] = None
                results['Y2_true'] = None
        
        # 記錄完整的預測歷史（包含特徵、預測值、真實值）
        prediction_record = {
            'timestamp': datetime.datetime.now(),
            'cfb_unit': self.cfb_unit,
            **current_features,  # 添加所有特徵數據
            **results
        }
        self.prediction_history.append(prediction_record)
        
        return results
    
    def run_realtime_loop(self, loop_interval=300):
        """執行即時預測迴圈"""
        print(f"🚀 開始 CFB#{self.cfb_unit} 即時預測迴圈 (每 {loop_interval} 秒)")
        print("按 Ctrl+C 停止...")
        
        last_true_y1 = None
        last_true_y2 = None
        
        try:
            while True:
                print(f"\n{'='*50}")
                print(f"⏰ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 收集即時數據
                current_data = self.collect_pi_data(realtime=True)
                
                if current_data is not None and not current_data.empty:
                    # 執行預測
                    predictions = self.run_online_prediction(
                        current_data.iloc[[-1]], 
                        last_true_y1, 
                        last_true_y2
                    )
                    
                    # 更新真實值（模擬延遲獲得真實值的情況）
                    if 'DeSOx_1st' in current_data.columns:
                        last_true_y1 = current_data['DeSOx_1st'].iloc[-1]
                    if 'DeSOx_2nd' in current_data.columns:
                        last_true_y2 = current_data['DeSOx_2nd'].iloc[-1]
                
                # 等待下一次迴圈
                print(f"💤 等待 {loop_interval} 秒...")
                time.sleep(loop_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 使用者中斷，正在生成最終報告...")
            self.generate_integration_report()
    
    def generate_integration_report(self):
        """生成整合報告"""
        if not self.prediction_history:
            print("📋 無預測歷史，跳過報告生成")
            return
        
        print("📊 正在生成整合系統報告...")
        
        # 生成業主系統的使用報告
        if self.predictor_y1:
            self.predictor_y1.generate_usage_report()
        
        if self.predictor_y2:
            self.predictor_y2.generate_usage_report()
        
        # 準備完整的預測歷史數據
        history_df = pd.DataFrame(self.prediction_history)
        
        # 重新整理欄位順序，讓時間戳記和關鍵指標在前面
        if not history_df.empty:
            # 基本資訊欄位
            basic_cols = ['timestamp', 'cfb_unit']
            
            # 預測和真實值欄位
            prediction_cols = [col for col in history_df.columns if col.endswith('_prediction') or col.endswith('_true')]
            
            # 特徵欄位
            feature_cols = [col for col in history_df.columns if col.startswith('feature_')]
            
            # 重新排列欄位順序
            ordered_cols = basic_cols + prediction_cols + feature_cols
            existing_cols = [col for col in ordered_cols if col in history_df.columns]
            history_df = history_df[existing_cols]
        
        # 儲存完整的預測歷史（包含特徵、預測值、真實值）
        history_filename = f"CFB{self.cfb_unit}_integration_history_{datetime.date.today().strftime('%Y%m%d')}.csv"
        history_df.to_csv(history_filename, index=False)
        print(f"📁 完整整合歷史已儲存至: {history_filename}")
        print(f"   包含 {len(history_df)} 筆記錄，{len(history_df.columns)} 個欄位")
        
        # 生成摘要統計
        if len(history_df) > 0:
            print(f"\n📈 預測摘要統計:")
            
            # Y1 統計
            if 'Y1_prediction' in history_df.columns:
                y1_pred = history_df['Y1_prediction'].dropna()
                y1_true = history_df.get('Y1_true', pd.Series()).dropna()
                
                if len(y1_pred) > 0:
                    print(f"   Y1 預測值範圍: {y1_pred.min():.4f} ~ {y1_pred.max():.4f}")
                    print(f"   Y1 預測值平均: {y1_pred.mean():.4f}")
                
                if len(y1_true) > 0:
                    print(f"   Y1 真實值範圍: {y1_true.min():.4f} ~ {y1_true.max():.4f}")
                    print(f"   Y1 真實值平均: {y1_true.mean():.4f}")
                    
                    # 計算預測誤差（如果有足夠的配對數據）
                    paired_data = history_df.dropna(subset=['Y1_prediction', 'Y1_true'])
                    if len(paired_data) > 0:
                        mae = abs(paired_data['Y1_prediction'] - paired_data['Y1_true']).mean()
                        rmse = ((paired_data['Y1_prediction'] - paired_data['Y1_true'])**2).mean()**0.5
                        print(f"   Y1 預測 MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            # Y2 統計（僅CFB2）
            if self.cfb_unit == 2 and 'Y2_prediction' in history_df.columns:
                y2_pred = history_df['Y2_prediction'].dropna()
                y2_true = history_df.get('Y2_true', pd.Series()).dropna()
                
                if len(y2_pred) > 0:
                    print(f"   Y2 預測值範圍: {y2_pred.min():.4f} ~ {y2_pred.max():.4f}")
                    print(f"   Y2 預測值平均: {y2_pred.mean():.4f}")
                
                if len(y2_true) > 0:
                    print(f"   Y2 真實值範圍: {y2_true.min():.4f} ~ {y2_true.max():.4f}")
                    print(f"   Y2 真實值平均: {y2_true.mean():.4f}")
                    
                    # 計算預測誤差
                    paired_data = history_df.dropna(subset=['Y2_prediction', 'Y2_true'])
                    if len(paired_data) > 0:
                        mae = abs(paired_data['Y2_prediction'] - paired_data['Y2_true']).mean()
                        rmse = ((paired_data['Y2_prediction'] - paired_data['Y2_true'])**2).mean()**0.5
                        print(f"   Y2 預測 MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            # 特徵統計摘要
            feature_columns = [col for col in history_df.columns if col.startswith('feature_')]
            if feature_columns:
                print(f"\n🔧 特徵數據摘要 (共 {len(feature_columns)} 個特徵):")
                for col in feature_columns[:5]:  # 只顯示前5個特徵
                    feature_data = history_df[col].dropna()
                    if len(feature_data) > 0:
                        feature_name = col.replace('feature_', '')
                        print(f"   {feature_name}: {feature_data.min():.2f} ~ {feature_data.max():.2f} (平均: {feature_data.mean():.2f})")
                
                if len(feature_columns) > 5:
                    print(f"   ... 以及其他 {len(feature_columns)-5} 個特徵")
        
        print("\n✅ 整合報告生成完成")
        print(f"💡 提示: 可使用以下程式碼分析完整數據:")
        print(f"   import pandas as pd")
        print(f"   df = pd.read_csv('{history_filename}')")
        print(f"   print(df.describe())")
        print(f"   print(df.columns.tolist())")


def main():
    """主程式入口"""
    print("🏭 CFB 脫硫劑效率線上學習系統 - 業主整合版")
    print("=" * 60)
    
    # 設定參數
    CFB_UNIT = 2  # 可改為 1 或 2
    
    # 初始化整合橋接器
    bridge = CFBIntegrationBridge(cfb_unit=CFB_UNIT)
    
    # 選擇運行模式
    print("\n請選擇運行模式:")
    print("1. 收集歷史數據並訓練模型")
    print("2. 執行即時預測迴圈") 
    print("3. 單次預測測試")
    
    choice = input("請輸入選項 (1-3): ").strip()
    
    if choice == "1":
        print("\n🔄 歷史數據收集與模型訓練模式")
        df = bridge.collect_pi_data(realtime=False)
        if df is not None:
            bridge.save_data_for_training(df)
            print("💡 請使用儲存的 Feather 檔案重新訓練 Y1 和 Y2 模型")
            
    elif choice == "2":
        print("\n🔄 即時預測模式")
        loop_interval = int(input("請輸入預測間隔(秒，建議300): ") or "300")
        bridge.run_realtime_loop(loop_interval)
        
    elif choice == "3":
        print("\n🔄 單次測試模式")
        df = bridge.collect_pi_data(realtime=True)
        if df is not None and not df.empty:
            results = bridge.run_online_prediction(df.iloc[[-1]])
            print(f"✅ 測試完成，預測結果: {results}")
        bridge.generate_integration_report()
        
    else:
        print("❌ 無效選項")

if __name__ == "__main__":
    main()