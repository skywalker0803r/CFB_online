#!/usr/bin/env python3
"""
從Y2/usage_report_y2.html萃取表格並生成y2_report_all_tables.xlsx
Extract tables from Y2/usage_report_y2.html and generate y2_report_all_tables.xlsx
"""

import pandas as pd
from bs4 import BeautifulSoup
import re
import os

def extract_tables_from_html(html_file_path):
    """
    從HTML文件中提取所有表格並轉換為DataFrame
    
    Args:
        html_file_path (str): HTML文件路徑
        
    Returns:
        dict: 包含所有表格的字典，key為表格名稱，value為DataFrame
    """
    
    print(f"開始處理文件: {html_file_path}")
    
    # 檢查文件是否存在
    if not os.path.exists(html_file_path):
        print(f"錯誤: 文件 {html_file_path} 不存在")
        return {}
    
    # 讀取HTML文件
    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except Exception as e:
        print(f"讀取文件失敗: {e}")
        return {}
    
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 找到所有表格
    tables = soup.find_all('table')
    print(f"找到 {len(tables)} 個表格")
    
    extracted_tables = {}
    
    for i, table in enumerate(tables):
        print(f"\n處理表格 {i+1}...")
        
        # 獲取表格標題
        title = f"table_{i+1}"
        
        # 尋找表格前面的h2或h1標題
        prev_element = table.find_previous_sibling(['h2', 'h1'])
        if prev_element and prev_element.name in ['h2', 'h1']:
            title = prev_element.get_text(strip=True)
            print(f"  找到標題: {title}")
            # 清理標題作為工作表名稱
            title_clean = re.sub(r'[^\w\s-]', '', title).replace(' ', '_').replace('-', '_')
            if title_clean:
                title = title_clean
        
        # 提取表格數據
        rows = []
        headers = []
        
        # 處理表頭
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                print(f"  表頭: {headers}")
        
        # 如果沒有找到表頭，使用第一行作為表頭
        if not headers:
            first_row = table.find('tr')
            if first_row:
                headers = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
                print(f"  使用第一行作為表頭: {headers}")
        
        # 處理表格主體數據
        tbody = table.find('tbody')
        if tbody:
            data_rows = tbody.find_all('tr')
        else:
            # 如果沒有tbody，獲取所有tr，如果第一行是表頭則跳過
            all_rows = table.find_all('tr')
            if headers and all_rows:
                # 檢查第一行是否為表頭
                first_row_cells = [cell.get_text(strip=True) for cell in all_rows[0].find_all(['th', 'td'])]
                if first_row_cells == headers:
                    data_rows = all_rows[1:]  # 跳過表頭行
                else:
                    data_rows = all_rows
            else:
                data_rows = all_rows
        
        print(f"  數據行數: {len(data_rows)}")
        
        # 提取每行數據
        for row_idx, row in enumerate(data_rows):
            cells = row.find_all(['td', 'th'])
            row_data = []
            
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                
                # 數據類型轉換
                if cell_text == '' or cell_text.lower() in ['nan', 'null']:
                    row_data.append(None)
                elif cell_text.lower() in ['true', 'false']:
                    row_data.append(cell_text.lower() == 'true')
                else:
                    # 嘗試轉換為數值
                    try:
                        # 檢查是否為整數
                        if cell_text.isdigit() or (cell_text.startswith('-') and cell_text[1:].isdigit()):
                            row_data.append(int(cell_text))
                        # 檢查是否為浮點數
                        elif '.' in cell_text or 'e' in cell_text.lower() or 'E' in cell_text:
                            row_data.append(float(cell_text))
                        else:
                            row_data.append(cell_text)
                    except ValueError:
                        row_data.append(cell_text)
            
            if row_data:  # 只添加非空行
                rows.append(row_data)
        
        # 創建DataFrame
        if rows:
            try:
                # 確保列名數量與數據匹配
                if headers:
                    max_cols = max(len(row) for row in rows) if rows else len(headers)
                    # 調整headers長度
                    while len(headers) < max_cols:
                        headers.append(f'column_{len(headers)+1}')
                    headers = headers[:max_cols]
                else:
                    # 創建默認列名
                    max_cols = max(len(row) for row in rows)
                    headers = [f'column_{i+1}' for i in range(max_cols)]
                
                # 確保每行數據長度一致
                for row in rows:
                    while len(row) < len(headers):
                        row.append(None)
                    if len(row) > len(headers):
                        row[:] = row[:len(headers)]
                
                df = pd.DataFrame(rows, columns=headers)
                extracted_tables[title] = df
                print(f"  ✓ 成功創建DataFrame: {title} - 形狀 {df.shape}")
                
            except Exception as e:
                print(f"  ✗ 創建DataFrame失敗: {e}")
                # 備用方案
                try:
                    df = pd.DataFrame(rows)
                    backup_title = f"table_{i+1}_backup"
                    extracted_tables[backup_title] = df
                    print(f"  ✓ 備用方案成功: {backup_title}")
                except Exception as e2:
                    print(f"  ✗ 備用方案也失敗: {e2}")
        else:
            print(f"  ! 表格 {i+1} 沒有數據行")
    
    return extracted_tables

def save_to_excel(tables_dict, output_filename="y2_report_all_tables.xlsx"):
    """
    將所有表格保存到Excel文件
    
    Args:
        tables_dict (dict): 包含DataFrame的字典
        output_filename (str): 輸出Excel文件名
    """
    
    if not tables_dict:
        print("沒有表格可以保存")
        return False
    
    try:
        print(f"\n開始保存到Excel文件: {output_filename}")
        
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            for table_name, df in tables_dict.items():
                # 處理Excel工作表名稱限制
                sheet_name = table_name[:31]  # 最大31字符
                
                # 移除Excel不允許的字符
                invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
                for char in invalid_chars:
                    sheet_name = sheet_name.replace(char, '_')
                
                # 確保工作表名稱唯一
                original_sheet_name = sheet_name
                counter = 1
                while sheet_name in [s for s in writer.sheets.keys()]:
                    sheet_name = f"{original_sheet_name}_{counter}"
                    counter += 1
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  ✓ 已保存工作表: {sheet_name} - {df.shape}")
        
        print(f"✓ Excel文件保存成功: {output_filename}")
        return True
        
    except Exception as e:
        print(f"✗ 保存Excel文件失敗: {e}")
        return False

def main():
    """主程式"""
    
    print("=== HTML表格提取工具 ===")
    
    # 設定輸入和輸出文件
    html_file = "usage_report_y2.html"
    excel_file = "y2_report_all_tables.xlsx"
    
    # 提取表格
    print("\n1. 提取HTML表格...")
    tables = extract_tables_from_html(html_file)
    
    if not tables:
        print("沒有提取到任何表格")
        return
    
    # 顯示提取結果摘要
    print(f"\n2. 提取結果摘要:")
    print(f"   共提取 {len(tables)} 個表格:")
    for name, df in tables.items():
        print(f"     - {name}: {df.shape} (行×列)")
        if len(df.columns) <= 5:  # 只顯示列較少的表格的列名
            print(f"       列名: {list(df.columns)}")
    
    # 保存到Excel
    print(f"\n3. 保存到Excel文件...")
    success = save_to_excel(tables, excel_file)
    
    if success:
        print(f"\n✅ 完成！")
        print(f"   Excel文件: {excel_file}")
        print(f"   包含 {len(tables)} 個工作表")
        
        # 顯示使用範例
        print(f"\n📖 使用範例:")
        print(f"import pandas as pd")
        print(f"")
        print(f"# 載入所有工作表")
        print(f"tables = pd.read_excel('{excel_file}', sheet_name=None)")
        print(f"")
        print(f"# 查看所有工作表名稱")
        print(f"print(list(tables.keys()))")
        print(f"")
        print(f"# 訪問特定表格")
        for i, name in enumerate(tables.keys()):
            sheet_name = name[:31]
            invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
            for char in invalid_chars:
                sheet_name = sheet_name.replace(char, '_')
            print(f"df_{i+1} = tables['{sheet_name}']")
            if i == 0:  # 只顯示第一個示例
                break
        
        return tables
    else:
        print("\n❌ 保存失敗")
        return None

if __name__ == "__main__":
    extracted_data = main()