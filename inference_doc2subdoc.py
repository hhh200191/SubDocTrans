import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.document_loaders import TextLoader

from helpers.ollama_df_helpers import df2Graph, documents2Dataframe, graph2Df, split_by_lines, df2Summary, df2Topic, split_by_lines_for_inference_get_all_sentences
from helpers.gpt_prompts import InferenceFirstChunkPrompt, InferencePrompt

# CUDA_VISIBLE_DEVICES=1,2

# 配置常量
DATA_DIR = Path("")
OUTPUT_BASE_DIR = Path("")
SUMMARIES_DIR = Path("")
THEMES_DIR = Path("")
HINTS_DIR = Path("")
TOPICS_DIR = Path("")
TRIPLETS_DIR = Path("")

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
    
    pages = split_by_lines_for_inference_get_all_sentences(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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
        return {"source_summary": "", "target_summary": ""}
    item = json_data[0]  # 只有一个 item
    return {
        f"{language_dict.get(src_lang, src_lang)} summary": item.get(f"Merged {language_dict.get(src_lang, src_lang)} summary", ""),
        f"{language_dict.get(tgt_lang, tgt_lang)} summary": item.get(f"Merged {language_dict.get(tgt_lang, tgt_lang)} summary", "")
    }

def process_theme(json_data: list, src_lang: str, tgt_lang: str, language_dict: dict) -> str:
    """处理 Theme distribution prediction 数据，返回主题分布预测字符串"""
    if not json_data or len(json_data) < 1:
        return ""
    return json_data[0].get(f"Merged {language_dict.get(tgt_lang, tgt_lang)} theme distribution prediction", "")

def process_hint(json_data: list, chunk_idx: int, src_lang: str, tgt_lang: str, language_dict: dict) -> str:
    """处理 Transition hint 数据，返回对应 chunk 的 src_lang 提示"""
    if not json_data or chunk_idx >= len(json_data):
        return ""
    if chunk_idx == 0:  # 第一个 chunk 不需要 hint
        return ""
    return json_data[chunk_idx].get(f"{language_dict.get(tgt_lang, tgt_lang)} hint", "")

def process_topic(json_data: list, chunk_idx: int, src_lang: str, tgt_lang: str, language_dict: dict) -> str:
    """处理 Topic 数据，返回对应 chunk tgt_lang topic，用逗号分隔"""
    if not json_data or chunk_idx >= len(json_data):
        return ""
    chunk_topics = json_data[chunk_idx]  # 获取对应 chunk 的 topic 列表
    chinese_topics = [item.get(f"{language_dict.get(tgt_lang, tgt_lang)} topic", "") for item in chunk_topics if item.get(f"{language_dict.get(tgt_lang, tgt_lang)} topic")]
    return ", ".join(chinese_topics)



def process_proper_nouns(json_data: list, chunk_text: str, src_lang: str, tgt_lang: str, language_dict: dict) -> str:
    """处理 triplets 数据，返回出现在 chunk 中的专有名词翻译"""
    if not json_data:
        return ""
    
    history_dict = {}
    for item in json_data:
        proper_nouns = item.get(f"{language_dict.get(src_lang, src_lang)} proper nouns", "")
        proper_nouns_tgt_lang = item.get(f"{language_dict.get(tgt_lang, tgt_lang)} proper nouns", "")
        history_dict[proper_nouns] = proper_nouns_tgt_lang
    
    matched_pairs = []
    for proper_nouns, proper_nouns_tgt_lang in history_dict.items():
        # 判断是否在 chunk_text 中且源语言和目标语言不同
        if proper_nouns in chunk_text:
            matched_pairs.append(f'"{proper_nouns}" - "{proper_nouns_tgt_lang}"')
    
    return ", ".join(matched_pairs)

"""
对英文实体长度进行限制，只保留长度小于四个单词的作为实体
"""
def process_triplets(json_data: list, chunk_text: str, src_lang: str, tgt_lang: str) -> str:
    """处理 triplets 数据，返回出现在 chunk 中的专有名词翻译"""
    if not json_data:
        return ""
    
    history_dict = {}
    for item in json_data:
        node_1_src_lang = item.get(f"node_1_{src_lang}", "")
        node_1_tgt_lang = item.get(f"node_1_{tgt_lang}", "")
        node_2_src_lang = item.get(f"node_2_{src_lang}", "")
        node_2_tgt_lang = item.get(f"node_2_{tgt_lang}", "")
        
        # 检查 node_1_src_lang 是否已存在，若不在则添加
        if node_1_src_lang and node_1_tgt_lang and node_1_src_lang not in history_dict:
            history_dict[node_1_src_lang] = node_1_tgt_lang
        
        # 检查 node_2_src_lang 是否已存在，若不在则添加
        if node_2_src_lang and node_2_tgt_lang and node_2_src_lang not in history_dict:
            history_dict[node_2_src_lang] = node_2_tgt_lang
    
    matched_pairs = []
    for src_lang_node, tgt_lang_node in history_dict.items():
        # 判断是否在 chunk_text 中且源语言和目标语言不同
        if src_lang_node in chunk_text and src_lang_node != tgt_lang_node:
            # 如果 src_lang 是 "en"，检查 src_lang_node 单词数是否超过 3
            if src_lang == "en" and len(src_lang_node.split(" ")) > 3:
                continue
            # 如果 tgt_lang 是 "en"，检查 tgt_lang_node 单词数是否超过 3
            if tgt_lang == "en" and len(tgt_lang_node.split(" ")) > 3:
                continue
            matched_pairs.append(f'"{src_lang_node}" - "{tgt_lang_node}"')
    
    return ", ".join(matched_pairs)

def inference(pages: list, src_lang: str, tgt_lang: str, language_dict: dict, model: str, summaries: dict, themes: list, hints: list, topics: list, triplets_raw: list) -> list:
    """对每个 chunk 进行推理，返回翻译结果，并验证 item 数量"""
    inference_results = []
    error_log = []

    for i, page in enumerate(pages):
        chunk_text = page.page_content.strip()
        chunk_lines = chunk_text.splitlines()  # 获取 chunk 的行数
        expected_items = len(chunk_lines)  # 预期 item 数量
        print(f"待翻译句子数量：{expected_items}")
        
        # 处理上下文信息
        hint = process_hint(hints, i, src_lang, tgt_lang, language_dict)
        topic = process_topic(topics, i, src_lang, tgt_lang, language_dict)
        triplets = process_triplets(triplets_raw, chunk_text, src_lang, tgt_lang)

        if i == 0:
            # 第一个 chunk 不需要 hint
            inputs = (
                f"Summaries:\n{language_dict.get(src_lang, src_lang)} Summary: {summaries.get(f'{language_dict.get(src_lang, src_lang)} summary', '')}\n"
                f"{language_dict.get(tgt_lang, tgt_lang)} Summary: {summaries.get(f'{language_dict.get(tgt_lang, tgt_lang)} summary', '')}\n\n"
                f"Theme distribution prediction: {themes}\n\n"
                f"Topic: {topic}\n\n"
                f"Historical translations of proper nouns: {triplets}\n\n"
                f"Now translate the following {language_dict.get(src_lang, src_lang)} source paragraph into {language_dict.get(tgt_lang, tgt_lang)}.\n"
                f"{language_dict.get(src_lang, src_lang)} paragraph:\n{chunk_text}"
            )
            
            result,is_valid = InferenceFirstChunkPrompt(inputs, model, src_lang, tgt_lang, language_dict, expected_items, chunk_lines)

            if result == chunk_lines:
                error_log.append("生成的内容涉及不正当言论被过滤调了")
                error_log.append(result)
                is_valid = True

            # 验证推理结果中句子的数量
            if len(result) != expected_items:
                error_log.append(f"Chunk {i} 翻译结果数量不匹配: 预期 {expected_items} 个 item，实际得到 {len(result)} 个 item")
                error_log.append(result)
            
            elif any("item解析失败" in item for item in result):
                error_log.append("item解析失败")
                error_log.append(result)

            elif not is_valid:
                error_log.append(f"实际顺序或内容错误")
                error_log.append(result)
        else:
            # 构造 USER_PROMPT
            inputs = (
                f"Summaries:\n{language_dict.get(src_lang, src_lang)} Summary: {summaries.get(f'{language_dict.get(src_lang, src_lang)} summary', '')}\n"
                f"{language_dict.get(tgt_lang, tgt_lang)} Summary: {summaries.get(f'{language_dict.get(tgt_lang, tgt_lang)} summary', '')}\n\n"
                f"Theme distribution prediction: {themes}\n\n"
                f"Transition hint: {hint}\n\n"
                f"Topic: {topic}\n\n"
                f"Historical translations of proper nouns: {triplets}\n\n"
                f"Now translate the following {language_dict.get(src_lang, src_lang)} source paragraph into {language_dict.get(tgt_lang, tgt_lang)}.\n"
                f"{language_dict.get(src_lang, src_lang)} paragraph:\n{chunk_text}"
            )
            result,is_valid = InferencePrompt(inputs, model, src_lang, tgt_lang, language_dict, expected_items, chunk_lines)

            flag = True
            if result == chunk_lines:
                error_log.append("生成的内容涉及不正当言论被过滤调了")
                error_log.append(result)
                is_valid = True
                flag = False

            # 验证推理结果中句子的数量
            if len(result) != expected_items:
                error_log.append(f"Chunk {i} 翻译结果数量不匹配: 预期 {expected_items} 个 item，实际得到 {len(result)} 个 item")
                error_log.append(result)
                flag = False
            
            elif any("item解析失败" in item for item in result):
                error_log.append("item解析失败")
                error_log.append(result)
                flag = False

            elif not is_valid:
                error_log.append(f"实际顺序或内容错误")
                error_log.append(result)
                flag = False

            if flag:
                # 去除前 3 个重叠 item（仅对非第一个 chunk）
                if len(result) > CHUNK_OVERLAP:
                    result = result[CHUNK_OVERLAP:]
                else:
                    result = []  # 如果结果少于重叠部分，认为无效

        inference_results.append(result)

    if not error_log:
        # 展平结果
        inference_results = np.concatenate(inference_results).ravel().tolist()
    return inference_results, error_log


def save_data(data: list, error_log: list, output_dir: Path, file_name: str) -> None:
    """保存 JSON 文件和错误日志到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    inference_json_path = output_dir / f"{file_name}.json"
    error_log_json_path = output_dir / f"{file_name}_error_log.json"

    # 保存 JSON 数据
    with open(inference_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"JSON 保存完成，路径: {inference_json_path}")

    # 保存错误日志（如果有内容）
    if error_log:
        with open(error_log_json_path, "w", encoding="utf-8") as f:
            json.dump(error_log, f, ensure_ascii=False, indent=4)
        print(f"错误日志保存完成，路径: {error_log_json_path}")
    else:
        print("无错误日志，无需保存")

def main():
    """主函数，处理文件夹下所有 _SRC_LANG.txt 文件，提取翻译并保存"""
    start_time = time.time()
    print(f"开始处理所有 _{SRC_LANG}.txt 文件...")

    validate_language_suffix(SRC_LANG, TGT_LANG, LANGUAGE_DICT)
    print("语言后缀验证完成")

    pattern = f"*_{SRC_LANG}.txt"
    txt_files = [f for f in DATA_DIR.glob(pattern) if f.is_file()]
    print(f"找到 {len(txt_files)} 个 _{SRC_LANG}.txt 文件")

    error_files = []  # 记录报错的文件

    for txt_file in txt_files:
        file_name = txt_file.stem
        output_dir = OUTPUT_BASE_DIR / file_name

        if output_dir.exists():
            print(f"子文件夹 {output_dir} 已存在，跳过文件 {txt_file.name}")
            continue

        file_start_time = time.time()
        print(f"\n处理文件: {txt_file.name}")

        # 加载并分割文档
        step_start = time.time()
        pages = load_and_split_document(txt_file)
        print(f"文档加载和分割完成，耗时: {time.time() - step_start:.2f} 秒")

        # 加载上下文信息
        step_start = time.time()
        summaries_raw = load_json_from_subfolder(SUMMARIES_DIR, file_name)
        themes_raw = load_json_from_subfolder(THEMES_DIR, file_name)
        hints_raw = load_json_from_subfolder(HINTS_DIR, file_name)
        topics_raw = load_json_from_subfolder(TOPICS_DIR, file_name)
        triplets_raw = load_json_from_subfolder(TRIPLETS_DIR, file_name)
        summaries = process_summaries(summaries_raw, SRC_LANG, TGT_LANG, LANGUAGE_DICT)
        themes = process_theme(themes_raw, SRC_LANG, TGT_LANG, LANGUAGE_DICT)
        print(f"上下文信息加载完成，耗时: {time.time() - step_start:.2f} 秒")

        # 提取翻译
        step_start = time.time()
        inference_result, error_log = inference(pages, SRC_LANG, TGT_LANG, LANGUAGE_DICT, MODEL, summaries, themes, hints_raw, topics_raw, triplets_raw)
        print(f"翻译提取完成，耗时: {time.time() - step_start:.2f} 秒")

        # 保存数据
        step_start = time.time()
        save_data(inference_result, error_log, output_dir, file_name)
        print(f"文件 {txt_file.name} 处理完成，耗时: {time.time() - file_start_time:.2f} 秒")


    total_time = time.time() - start_time
    print(f"\n所有文件处理完成，总执行时间: {total_time:.2f} 秒")
    if error_files:
        print(f"以下文件处理时发生错误: {', '.join(error_files)}")
    else:
        print("所有文件处理成功，无错误")

    # 将总执行时间追加到 result.log，单位转换为小时
    log_file = "/home/xmt/Doc2subdoc/result.log"
    with open(log_file, "a") as f:  # 以追加模式打开文件
        total_time_hours = total_time / 3600  # 转换为小时
        f.write(f"Execution time for {OUTPUT_BASE_DIR}: {total_time_hours:.2f} hours\n")

if __name__ == "__main__":
    main()