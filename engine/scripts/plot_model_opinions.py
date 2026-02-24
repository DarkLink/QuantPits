#!/usr/bin/env python3
"""
可视化 model_opinions_*.csv 排名的折线图
"""

import os
import glob
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import env
os.chdir(env.ROOT_DIR)

# 尝试导入 seaborn 优化样式
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    parser = argparse.ArgumentParser(description="绘制 model_opinions 排名折线图")
    parser.add_argument('--input', type=str, help='指定输入的 model_opinions CSV 文件路径 (默认自动查找最新的)')
    args = parser.parse_args()

    # 查找输入文件
    if args.input:
        csv_file = args.input
        if not os.path.exists(csv_file):
            print(f"找不到文件: {csv_file}")
            return
    else:
        # 默认自动查找最新的
        csv_files = glob.glob('output/model_opinions_*.csv')
        if not csv_files:
            print("在 output/ 目录下未找到 model_opinions_*.csv 文件")
            return
        csv_file = sorted(csv_files)[-1]
    
    print(f"正在读取数据: {csv_file}")
    df = pd.read_csv(csv_file, index_col=0)

    # 提取排名数字
    # 格式例如 "BUY (1)" 或者 "-- (150)"
    rank_df = pd.DataFrame(index=df.index, columns=df.columns)
    
    for col in df.columns:
        for idx in df.index:
            val = str(df.loc[idx, col])
            # 正则匹配括号内的数字
            m = re.search(r'\((\d+)\)', val)
            if m:
                rank_df.loc[idx, col] = int(m.group(1))
            else:
                rank_df.loc[idx, col] = None
                
    # 转换为 float，为了画图能处理 NaN
    rank_df = rank_df.astype(float)

    # 画折线图 (Parallel Coordinates 风格)
    # X 轴为各个模型/Combo，Y 轴为排名，每一根线代表一只股票
    plt.figure(figsize=(14, 8))
    
    for instrument in rank_df.index:
        y_values = rank_df.loc[instrument]
        if y_values.isna().all():
            continue
        plt.plot(rank_df.columns, y_values, marker='o', alpha=0.7, label=instrument)
        
    # Y轴刻度反转，使得排名第 1 的在最上面
    plt.gca().invert_yaxis()
    
    # 调整 X 轴标签角度，防止重叠
    plt.xticks(rotation=30, ha='right')
    
    plt.ylabel('模型预测排名 (数字越小越靠前)')
    plt.title(f'模型预测排名对比 - {os.path.basename(csv_file)}')
    
    # 将图例画在外面
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="标的 (Instrument)")
    
    plt.tight_layout()
    
    # 确定输出路径
    basename = os.path.splitext(os.path.basename(csv_file))[0]
    out_img = os.path.join(os.path.dirname(csv_file), f"{basename}_linechart.png")
    
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"折线图已保存至: {out_img}")

if __name__ == "__main__":
    # 切换到项目根目录执行
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    os.chdir(ROOT_DIR)
    
    main()
