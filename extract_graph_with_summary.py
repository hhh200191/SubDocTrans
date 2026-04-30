import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.document_loaders import TextLoader

from helpers.ollama_df_helpers import df2Graph, documents2Dataframe, graph2Df, split_by_lines

from helpers.gpt_prompts import df2Graph_with_summary

# 配置常量
DATA_DIR = Path("")
SUMMARIES_DIR = Path("")
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

def load_json_from_subfolder(base_dir: Path, subfolder_name: str) -> dict:
    """从指定目录的子文件夹中加载 JSON 文件"""
    json_path = base_dir / subfolder_name / f"{subfolder_name}.json"
    if not json_path.exists():
        print(f"未找到文件 {json_path}，使用默认空值")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def process_summaries(json_data: list, src_lang: str, tgt_lang: str, language_dict: dict) -> dict:
    """处理 Summary 数据，返回源语言和目标语言的摘要"""
    if not json_data or len(json_data) < 1:
        return {"source_summary": ""}
    item = json_data[0]  # 只有一个 item
    return {
        f"{language_dict.get(src_lang, src_lang)} summary": item.get(f"Merged {language_dict.get(src_lang, src_lang)} summary", ""),
    }

def clean_and_count_triplets(dfg: pd.DataFrame) -> pd.DataFrame:
    """清理数据，去除自环并统计三元组出现次数"""
    dfg.replace("", np.nan, inplace=True)
    dfg.dropna(subset=["node_1", "node_2", "edge"], inplace=True)
    
    # 移除自环
    dfg = dfg[dfg["node_1"] != dfg["node_2"]]
    
    # 统计三元组出现次数
    dfg = dfg.groupby(["node_1", "node_2", "edge"]).size().reset_index(name="count")
    
    return dfg

def process_triplets(pages: list, summaries: dict, src_lang: str, tgt_lang: str, language_dict: dict, model: str) -> tuple:
    """提取三元组并生成映射字典，为每个 chunk 添加 summary"""
    # 将文档转换为 DataFrame
    df = documents2Dataframe(pages)
    
    # 添加 summary 列，默认使用源语言摘要
    source_summary = summaries.get(f"{language_dict.get(src_lang, src_lang)} summary", "")
    df["summary"] = source_summary  # 为每一行添加相同的摘要
    
    # 提取三元组
    concepts_list = df2Graph_with_summary(df, model=model, src_lang=src_lang, tgt_lang=tgt_lang, dic=language_dict)
    dfg, mapping_dict = graph2Df(concepts_list, src_lang=src_lang, tgt_lang=tgt_lang)
    
    return dfg, mapping_dict

def combine_bilingual_triplets(graph_df_lang1: pd.DataFrame, mapping_dict: dict, src_lang: str, tgt_lang: str) -> list:
    """将 graph_df_lang1 和 mapping_dict 组合成双语三元组列表，符合目标 JSON 格式"""
    bilingual_triplets = []
    for _, row in graph_df_lang1.iterrows():
        node1_src = row['node_1']
        node2_src = row['node_2']
        edge_src = row['edge']
        
        node1_tgt = mapping_dict.get(node1_src)
        node2_tgt = mapping_dict.get(node2_src)
        edge_tgt = mapping_dict.get(edge_src)
        
        if node1_tgt is not None and node2_tgt is not None and edge_tgt is not None:
            triplet = {
                f"node_1_{src_lang}": node1_src,
                f"node_1_{tgt_lang}": node1_tgt,
                f"node_2_{src_lang}": node2_src,
                f"node_2_{tgt_lang}": node2_tgt,
                f"edge_{src_lang}": edge_src,
                f"edge_{tgt_lang}": edge_tgt
            }
            bilingual_triplets.append(triplet)
    
    return bilingual_triplets

def save_data(dfg: pd.DataFrame, bilingual_triplets: list, output_dir: Path, file_name: str) -> None:
    """保存三元组 CSV 和双语三元组 JSON 文件到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 CSV 文件
    csv_path = output_dir / f"{file_name}.csv"
    dfg.to_csv(csv_path, sep="|", index=False)
    print(f"CSV 保存完成，路径: {csv_path}")
    
    # 保存 JSON 文件
    json_path = output_dir / f"{file_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bilingual_triplets, f, ensure_ascii=False, indent=4)
    print(f"JSON 保存完成，路径: {json_path}")

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
        print(f"文档加载和分割完成，耗时: {time.time() - step_start:.2f} 秒")

        summaries_raw = load_json_from_subfolder(SUMMARIES_DIR, file_name)
        summaries = process_summaries(summaries_raw, SRC_LANG, TGT_LANG, LANGUAGE_DICT)

        # 提取三元组
        step_start = time.time()
        graph_df_lang1, mapping_dict = process_triplets(pages, summaries, SRC_LANG, TGT_LANG, LANGUAGE_DICT, MODEL)
        print(f"三元组提取完成，耗时: {time.time() - step_start:.2f} 秒")

        # 清理和统计三元组
        graph_df_lang1 = clean_and_count_triplets(graph_df_lang1)

        # 组合双语三元组
        step_start = time.time()
        bilingual_triplets = combine_bilingual_triplets(graph_df_lang1, mapping_dict, SRC_LANG, TGT_LANG)
        print(f"双语三元组组合完成，耗时: {time.time() - step_start:.2f} 秒")
        
        # 保存数据
        step_start = time.time()
        save_data(graph_df_lang1, bilingual_triplets, output_dir, file_name)
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