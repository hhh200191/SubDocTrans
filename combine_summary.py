import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.document_loaders import TextLoader


from helpers.gpt_prompts import  CombineSummaryPrompt

# 配置常量
DATA_DIR = Path("")
OUTPUT_BASE_DIR = Path("") 

LANGUAGE_DICT = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "ja": "Japanese",
}

SRC_LANG = "zh"
TGT_LANG = "en"
MODEL = "gpt-4o-mini"
CHUNK_SIZE = 20  
CHUNK_OVERLAP = 3

def validate_language_suffix(src_lang: str, tgt_lang: str, language_dict: dict) -> None:
    """验证语言后缀是否在支持的字典中"""
    if src_lang not in language_dict or tgt_lang not in language_dict:
        raise ValueError("输入的语言后缀不在支持的范围内")

def extract_summarys_from_json(json_path: Path, src_lang: str, tgt_lang: str, language_dict: dict) -> str:
    """从 JSON 文件中提取每隔一个条目的 Chinese summary 和 Theme distribution prediction，并构造成带编号的字符串"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    summaries = []
    themes = []
    for i, item in enumerate(data):
        # if i % 2 == 0:  # 选择第 0, 2, 4... 个条目（即每隔一个）
        summary = item.get(f"{language_dict.get(src_lang, src_lang)} summary", "")
        theme = item.get("Theme distribution prediction", "")
        if summary and theme:  # 确保字段不为空
            # 添加编号，如 Chinese summary 1, Chinese summary 2...
            summaries.append(f"{language_dict.get(src_lang, src_lang)} summary {len(summaries) + 1}: {summary}")
            themes.append(f"Theme distribution prediction {len(themes) + 1}: {theme}")
    
    # 将所有 summaries 和 themes 合并为一个字符串，中间用 \n\n 分隔
    return "\n".join(summaries) + "\n\n" + "\n".join(themes)

def combine_summary(summarys: str, src_lang: str, tgt_lang: str, language_dict: dict, model: str) -> list:
    """合并摘要和主题预测"""
    # 调用 CombineSummaryPrompt 函数，假设其返回一个 JSON 列表
    CombinedSummary = CombineSummaryPrompt(summarys, model=model, src_lang=src_lang, tgt_lang=tgt_lang, dic=language_dict)
    return CombinedSummary

def save_data(CombinedSummary: list, output_dir: Path, file_name: str) -> None:
    """保存 JSON 文件到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 JSON 文件
    json_path = output_dir / f"{file_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(CombinedSummary, f, ensure_ascii=False, indent=4)
    print(f"JSON 保存完成，路径: {json_path}")

def main():
    """主函数，处理文件夹下所有子文件夹中的同名 JSON 文件，提取 summarys 并合并成一个总的 summary，因为直接用大模型处理整篇文档太长了"""
    start_time = time.time()
    print("开始处理所有子文件夹中的 JSON 文件...")

    # 验证语言后缀
    validate_language_suffix(SRC_LANG, TGT_LANG, LANGUAGE_DICT)
    print("语言后缀验证完成")

    # 获取所有子文件夹
    subfolders = [f for f in DATA_DIR.iterdir() if f.is_dir()]
    print(f"找到 {len(subfolders)} 个子文件夹")

    for subfolder in subfolders:
        file_name = subfolder.name  # e.g., "2425_zh"
        json_file = subfolder / f"{file_name}.json"
        
        if not json_file.exists():
            print(f"JSON 文件 {json_file} 不存在，跳过")
            continue

        output_dir = OUTPUT_BASE_DIR / file_name

        # 检查是否已存在同名子文件夹
        if output_dir.exists():
            print(f"输出文件夹 {output_dir} 已存在，跳过文件 {json_file.name}")
            continue

        file_start_time = time.time()
        print(f"\n处理文件: {json_file.name}")

        # 提取 summarys
        step_start = time.time()
        summarys = extract_summarys_from_json(json_file, SRC_LANG, TGT_LANG, LANGUAGE_DICT)
        print(f"summarys 提取完成，耗时: {time.time() - step_start:.2f} 秒")

        # 合并 summary
        step_start = time.time()
        combined_summary = combine_summary(summarys, SRC_LANG, TGT_LANG, LANGUAGE_DICT, MODEL)
        print(f"summary 合并完成，耗时: {time.time() - step_start:.2f} 秒")

        # 保存数据
        step_start = time.time()
        save_data(combined_summary, output_dir, file_name)
        print(f"文件 {json_file.name} 处理完成，耗时: {time.time() - file_start_time:.2f} 秒")

    total_time = time.time() - start_time
    print(f"\n所有文件处理完成，总执行时间: {total_time:.2f} 秒")

    # 将总执行时间追加到 result.log，单位转换为小时
    log_file = "/home/xmt/Doc2subdoc/result.log"
    with open(log_file, "a") as f:  # 以追加模式打开文件
        total_time_hours = total_time / 3600  # 转换为小时
        f.write(f"Execution time for {OUTPUT_BASE_DIR}: {total_time_hours:.2f} hours\n")

if __name__ == "__main__":
    main()