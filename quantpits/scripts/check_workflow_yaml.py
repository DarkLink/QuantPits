#!/usr/bin/env python
"""
Workflow YAML Validator (频率模式配置验证器)

检查 config/ 目录下的 workflow_config_*.yaml 文件是否完全符合配置要求：
1. data_handler_config 下的 label 必须符合预期频次
2. port_analysis_config 下的 executor.kwargs.time_per_step 必须与频次一致
3. task.record 下的 SigAnaRecord 的 ann_scaler 必须正确

用法：
    python quantpits/scripts/check_workflow_yaml.py [--fix]
"""

import os
import glob
import re
import yaml
import argparse

import env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
CONFIG_DIR = os.path.join(ROOT_DIR, "config")


def check_yamls(freq="week"):
    """检查所有的 workflow yaml，返回包含异常的字典"""
    
    # 根据频率确定预期值
    if freq == "week":
        expected_label = ["Ref($close, -6) / Ref($close, -1) - 1"]
        expected_time_per_step = "week"
        expected_ann_scaler = 52
    else:
        expected_label = ["Ref($close, -2) / Ref($close, -1) - 1"]
        expected_time_per_step = "day"
        expected_ann_scaler = 252
    files = glob.glob(os.path.join(CONFIG_DIR, "workflow_config_*.yaml"))
    anomalies = {}
    
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except Exception as e:
                anomalies[filename] = [f"Error parsing YAML: {e}"]
                continue
                
        issues = []
        
        # 1. 检查 label
        # 对于多步预测模型(如tcts)，可能包含多个标签，但第一个必须是以-6结尾
        is_tcts = "tcts" in filename.lower()
        
        dh_config = data.get("data_handler_config", {})
        if not isinstance(dh_config, dict):
            dh_config = {}
            
        label = dh_config.get("label", None)
        
        # 尝试深度查找 (有些 yaml 的 handler 放在 kwargs 内部)
        if label is None:
            try:
                handler_kwargs = data["task"]["dataset"]["kwargs"]["handler"]
                if isinstance(handler_kwargs, dict) and "kwargs" in handler_kwargs:
                    label = handler_kwargs["kwargs"].get("label", [])
            except KeyError:
                pass
                
        if isinstance(label, list):
            # 去除字符串内的空格干扰进行比较
            clean_labels = [l.replace(" ", "") for l in label]
            if is_tcts:
                if not any("Ref($close,-6)/Ref($close,-1)-1" in l for l in clean_labels):
                   issues.append(f"LABEL: 多步预测缺失周频步(-6): {str(label)}")
            else:
                expected_clean = expected_label[0].replace(" ", "")
                if len(clean_labels) != 1 or clean_labels[0] != expected_clean:
                   issues.append(f"LABEL: 期望 {expected_label}, 实际 {label}")
        else:
            issues.append(f"LABEL: 格式错误或未找到: {label}")

        # 2. 检查 time_per_step
        pa_config = data.get("port_analysis_config", {})
        time_per_step = None
        try:
            time_per_step = pa_config["executor"]["kwargs"]["time_per_step"]
        except KeyError:
            # 尝试在 task.record 中查找
            try:
                for rec in data["task"]["record"]:
                    if rec["class"] == "PortAnaRecord":
                        time_per_step = rec["kwargs"]["config"]["executor"]["kwargs"]["time_per_step"]
                        break
            except KeyError:
                pass
                
        if time_per_step != expected_time_per_step:
            issues.append(f"TIME_PER_STEP: 期望 '{expected_time_per_step}', 实际 '{time_per_step}'")

        # 3. 检查 ann_scaler
        ann_scaler = None
        try:
            records = data.get("task", {}).get("record", [])
            for rec in records:
                if rec.get("class") == "SigAnaRecord":
                    ann_scaler = rec.get("kwargs", {}).get("ann_scaler", None)
        except AttributeError:
            pass
            
        if ann_scaler != expected_ann_scaler:
            issues.append(f"ANN_SCALER: 期望 {expected_ann_scaler}, 实际 '{ann_scaler}'")

        # 4. 检查 lr 的科学计数法 (直接读取文本)
        with open(filepath, "r", encoding="utf-8") as raw_f:
            raw_content = raw_f.read()
            if re.search(r'^\s*lr:\s*\d+(\.\d+)?[eE]-\d+', raw_content, flags=re.MULTILINE):
                issues.append("LR: 发现科学计数法格式，可能导致后续处理报错")

        if issues:
            anomalies[filename] = issues
            
    return anomalies


def fix_yamls(freq="week"):
    """使用正则等字符串替换模式修复YAML，保留原本的格式和注释"""
    
    if freq == "week":
        target_label = 'label: ["Ref($close, -6) / Ref($close, -1) - 1"]'
        target_ann_scaler = 'ann_scaler: 52'
        target_time_per_step = 'time_per_step: "week"'
    else:
        target_label = 'label: ["Ref($close, -2) / Ref($close, -1) - 1"]'
        target_ann_scaler = 'ann_scaler: 252'
        target_time_per_step = 'time_per_step: "day"'
    files = glob.glob(os.path.join(CONFIG_DIR, "workflow_config_*.yaml"))
    fixed_count = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        file_has_executor = any(re.search(r'^\s*executor:', line) for line in lines)
        new_lines = []
        in_pa_config = False
        modified = False
        
        for line in lines:
            original_line = line
            
            # 1. 替换非周频的预测 label (通用模型)
            if "tcts" not in filename.lower():
                line = re.sub(
                    r'label:\s*\[[\"\']Ref\(\$close,\s*-\d+\)\s*/\s*Ref\(\$close,\s*-1\)\s*-\s*1[\"\']\]', 
                    target_label, 
                    line
                )
            else:
                # TCTS 模型特殊处理 (如果依然是旧的日频配置，主动替换为周频多步)
                if "Ref($close, -2)" in line and "Ref($close, -3)" in line:
                    line = re.sub(
                        r'label:\s*\[.*?\]',
                        'label: ["Ref($close, -6) / Ref($close, -1) - 1", "Ref($close, -11) / Ref($close, -1) - 1", "Ref($close, -16) / Ref($close, -1) - 1"]',
                        line,
                        flags=re.DOTALL
                    )
            
            # 2. 替换 ann_scaler (252 -> 52)
            line = re.sub(r'ann_scaler:\s*\d+', target_ann_scaler, line)
            
            # 3. 替换 time_per_step (day/step -> week)
            line = re.sub(r'time_per_step:\s*[\"\'](day|step|week)[\"\']', target_time_per_step, line)
            
            # 4. 替换科学计数法的 lr
            lr_match = re.search(r'lr:\s*(\d+(\.\d+)?[eE]-\d+)', line)
            if lr_match:
                sci_val = lr_match.group(1)
                try:
                    decimal_val = f"{float(sci_val):.10f}".rstrip('0')
                    if decimal_val.endswith('.'):
                        decimal_val += '0'
                    line = line.replace(sci_val, decimal_val)
                except ValueError:
                    pass
            
            # 5. 插入缺失的 executor (在 port_analysis_config 或 backtest 时机插入)
            if line.strip().startswith('port_analysis_config:'):
                in_pa_config = True
                
            if not file_has_executor and in_pa_config and line.strip().startswith('backtest:'):
                indent = line[:len(line) - len(line.lstrip())]
                if indent == "":
                    indent = "    "
                executor_block = (
                    f"{indent}executor:\n"
                    f"{indent}    class: SimulatorExecutor\n"
                    f"{indent}    module_path: qlib.backtest.executor\n"
                    f"{indent}    kwargs:\n"
                    f"{indent}        time_per_step: \"{freq}\"\n"
                    f"{indent}        generate_portfolio_metrics: true\n"
                    f"{indent}        verbose: false\n"
                )
                new_lines.append(executor_block)
                in_pa_config = False
                modified = True
                
            if original_line != line:
                modified = True
                
            new_lines.append(line)
            
        if modified:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            fixed_count += 1
            
    print(f"🔧 已尝试按 {freq} 频次自动修复 {fixed_count} 个配置文件。")


def main():
    parser = argparse.ArgumentParser(description="Workflow YAML 频率配置验证工具")
    parser.add_argument("--fix", action="store_true", help="尝试自动修复有问题的参数")
    args = parser.parse_args()
    
    import json
    model_config_path = os.path.join(ROOT_DIR, "config", "model_config.json")
    freq = "week"
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            m_cfg = json.load(f)
            freq = m_cfg.get("freq", "week")
            
    print(f"🔍 检测配置频次: {freq}")
    anomalies = check_yamls(freq=freq)
    
    if args.fix:
        if anomalies:
            print(f"开始执行自动修复 ({freq})...")
            fix_yamls(freq=freq)
            # 修复后再检查一次
            anomalies = check_yamls(freq=freq)
        else:
            print(f"所有的工作流配置文件都完美符合 {freq} 模式要求，无需修复！")
            return

    if not anomalies:
        print(f"✅ 所有的工作流配置文件都完美符合 {freq} 频次要求！")
    else:
        print(f"❌ 发现部分配置文件仍不符合 {freq} 要求：")
        print("-" * 50)
        for filename, issues in anomalies.items():
            print(f"📄 {filename}")
            for issue in issues:
                print(f"  - {issue}")
        print("-" * 50)
        if not args.fix:
            print("\n💡 提示：运行 `python quantpits/scripts/check_workflow_yaml.py --fix` 尝试自动修复。")


if __name__ == "__main__":
    main()
