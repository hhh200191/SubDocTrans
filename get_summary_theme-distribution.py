import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.document_loaders import TextLoader

from helpers.ollama_df_helpers import df2Graph, documents2Dataframe, graph2Df, split_by_lines

from helpers.gpt_prompts import df2Summary

# 配置常量
DATA_DIR = Path("")
OUTPUT_BASE_DIR = Path("")  # 输出目录与输入目录相同

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

MAX_RETRIES = 10  # 最大重试次数

def validate_language_suffix(src_lang: str, tgt_lang: str, language_dict: dict) -> None:
    """验证语言后缀是否在支持的字典中"""
    if src_lang not in language_dict or tgt_lang not in language_dict:
        raise ValueError("输入的语言后缀不在支持的范围内")

def load_and_split_document(file_path: Path) -> list:
    """加载并分割文档为指定大小的 chunk"""
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()
    
    print(f"Processing file: {file_path.name}")
    print(f"Number of documents: {len(documents)}")
    print(f"Lines in document: {len(documents[0].page_content.splitlines())}")
    
    pages = split_by_lines(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Number of chunks: {len(pages)}")
    
    return pages


def process_summary(pages: list, src_lang: str, tgt_lang: str, language_dict: dict, model: str) -> tuple:
    """提取summary 和 主题预测"""

    df = documents2Dataframe(pages)
    Summary = df2Summary(df, model=model, src_lang=src_lang, tgt_lang=tgt_lang, dic=language_dict)

    return Summary


def save_data(Summary: list, output_dir: Path, file_name: str, expected_length: int) -> None:
    """保存Summary JSON 文件到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 JSON 文件
    json_path = output_dir / f"{file_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(Summary, f, ensure_ascii=False, indent=4)
    print(f"JSON 保存完成，路径: {json_path}")

     # 检查长度并保存错误日志
    if len(Summary) != expected_length:
        error_log_path = output_dir / "error_log.txt"
        error_message = f"Summary长度 {len(Summary)} 不匹配预期 {expected_length}"
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(error_message)
        print(f"错误日志保存完成，路径: {error_log_path}")

def main():
    """主函数，处理文件夹下所有 _zh.txt 文件，提取双语三元组并保存"""
    start_time = time.time()
    print(f"开始处理所有 _{SRC_LANG}.txt 文件...")

    # 验证语言后缀
    validate_language_suffix(SRC_LANG, TGT_LANG, LANGUAGE_DICT)
    print("语言后缀验证完成")

    # 获取所有 _zh.txt 文件
    txt_files = [f for f in DATA_DIR.glob(f"*_{SRC_LANG}.txt") if f.is_file()]
    print(f"找到 {len(txt_files)} 个 _{SRC_LANG}.txt 文件")

    for txt_file in txt_files:
        # 获取文件名（不含扩展名）
        file_name = txt_file.stem  # e.g., "2425_zh"
        output_dir = OUTPUT_BASE_DIR / file_name

        # 检查是否已存在同名子文件夹
        if output_dir.exists():
            print(f"子文件夹 {output_dir} 已存在，跳过文件 {txt_file.name}")
            continue  # 跳过已处理的文件

        file_start_time = time.time()
        print(f"\n处理文件: {txt_file.name}")

        # 加载并分割文档
        step_start = time.time()
        pages = load_and_split_document(txt_file)
        expected_length = len(pages)
        print(f"文档加载和分割完成，耗时: {time.time() - step_start:.2f} 秒")

        # 提取主题，带重试
        step_start = time.time()
        retries = 0
        while retries <= MAX_RETRIES:
            summary = process_summary(pages, SRC_LANG, TGT_LANG, LANGUAGE_DICT, MODEL)
            
            if len(summary) == expected_length:
                print(f"主题长度 {len(summary)} 匹配预期 {expected_length}")
                break
            else:
                print(f"主题长度 {len(summary)} 不匹配预期 {expected_length}，重试...")
                retries += 1

        print(f"三元组提取完成，耗时: {time.time() - step_start:.2f} 秒")

        # 保存数据
        step_start = time.time()
        save_data(summary, output_dir, file_name, expected_length)
        print(f"文件 {txt_file.name} 处理完成，耗时: {time.time() - file_start_time:.2f} 秒")

    total_time = time.time() - start_time
    print(f"\n所有文件处理完成，总执行时间: {total_time:.2f} 秒")

    # 将总执行时间追加到 result.log，单位转换为小时
    log_file = "/home/xmt/Doc2subdoc/result.log"
    with open(log_file, "a") as f:  # 以追加模式打开文件
        total_time_hours = total_time / 3600  # 转换为小时
        f.write(f"Execution time for {OUTPUT_BASE_DIR}: {total_time_hours:.2f} hours\n")

if __name__ == "__main__":
    main()