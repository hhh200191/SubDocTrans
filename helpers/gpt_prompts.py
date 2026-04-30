import sys
from yachalk import chalk
import re
from typing import List, Tuple, Dict, Optional, Any
sys.path.append("..")
import asyncio
import aiohttp
import json
import uuid
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import os
from .ollama_prompts import extract_inference_from_text, extract_summary_from_text, extract_topic_from_text, extract_triplets_from_text


api_base = ''
api_key = ''

# MODEL_NAME="gpt-4o-mini"
# MODEL_NAME="gpt-3.5-turbo-0125"

MODEL_NAME="gpt-4o-mini"


client = OpenAI(api_key=api_key, base_url=api_base, timeout=30)


def invoke_chat_api(SYS_PROMPT:str, user_prompt: str, model:str):
    try_cnt = 0
    while try_cnt < 10:
        try:
            completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYS_PROMPT
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.8,
                    top_p=0.8,

                )
            if completion is None:
                raise RuntimeError('Returned None!')
            break
        except Exception as e:
            completion = None
            print(e)
            try_cnt += 1
            print("Retry in 2 seconds")
            time.sleep(2)

    if completion is None:
        print('Error waiting')
        return None
    
    response = completion.choices[0].message.content.strip()
    print("🔍 Raw Response from GPT:\n", response)
    print("Finish reason:", completion.choices[0].finish_reason)

    if completion.choices[0].finish_reason == 'content_filter':
        print(f"警告：响应因内容过滤终止")
        return None, True
    return response, False



def df2Graph_with_summary(dataframe: pd.DataFrame, model=None, src_lang='en', tgt_lang='zh', dic={}) -> list:
    results = dataframe.apply(
        lambda row: graphPrompt_with_summary(row.text, row.summary, model, src_lang, tgt_lang, dic), axis=1
    )
    # 过滤掉非列表项（包括 None 或其他无效类型）
    valid_results = [r for r in results if isinstance(r, list) and r]

    if not valid_results:
        print("Warning: No valid triplets extracted.")
        return []

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(valid_results).ravel().tolist()
    return concept_list


def df2Summary(dataframe: pd.DataFrame, model=None, src_lang='en', tgt_lang='zh', dic={}) -> list:
    results = dataframe.apply(
        lambda row: SummaryPrompt(row.text, model, src_lang, tgt_lang, dic), axis=1
    )
    # 过滤掉非列表项（包括 None 或其他无效类型）
    valid_results = [r for r in results if isinstance(r, list) and r]

    if not valid_results:
        print("Warning: No valid triplets extracted.")
        return []

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(valid_results).ravel().tolist()
    return concept_list

def df2Topic(dataframe: pd.DataFrame, model=None, src_lang='en', tgt_lang='zh', dic={}) -> list:
    results = dataframe.apply(
        lambda row: TopicPrompt(row.text, model, src_lang, tgt_lang, dic), axis=1
    )
    # 过滤掉非列表项（包括 None 或其他无效类型）
    valid_results = [r for r in results if isinstance(r, list) and r]

    if not valid_results:
        print("Warning: No valid triplets extracted.")
        return []

    ## Flatten the list of lists to one single list of entities.
    # concept_list = np.concatenate(valid_results).ravel().tolist()
    return valid_results


def df2Hint(dataframe: pd.DataFrame, model=None, src_lang='en', tgt_lang='zh', dic={}) -> list:
    """
    从 DataFrame 中提取衔接提示，将当前行和前一行传递给 HintPrompt。
    
    Args:
        dataframe (pd.DataFrame): 包含 'text' 列的 DataFrame
        model (str, optional): 使用的模型名称，默认为 None
        src_lang (str): 源语言，默认为 'en'
        tgt_lang (str): 目标语言，默认为 'zh'
        dic (dict): 语言映射字典，默认为空字典
    
    Returns:
        list: 包含所有衔接提示的列表
    """
    results = []
    
    # 遍历 DataFrame 的行
    for i in range(len(dataframe)):
        # 当前行的 text
        current_text = dataframe.iloc[i]['text']
        
        # 前一行的 text，如果是第一行则为空字符串
        previous_text = dataframe.iloc[i-1]['text'] if i > 0 else ""
        
        # 调用 HintPrompt，传入前一行和当前行的 text
        hint_result = HintPrompt(
            input_pre=previous_text,
            input=current_text,
            model=model,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            dic=dic
        )
        
        # 将结果添加到列表中
        results.append(hint_result)

    # 过滤掉非列表项（包括 None 或空列表）
    valid_results = [r for r in results if isinstance(r, list) and r]

    if not valid_results:
        print("Warning: No valid hints extracted.")
        return []

    # 展平列表，将所有衔接提示合并为单一列表
    concept_list = np.concatenate(valid_results).ravel().tolist()
    return concept_list





def CombineSummaryPrompt(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}):

    SYS_PROMPT = (
        "You are a professional article analysis agent. You are provided with a context (delimited by ```) "
        "Below are the summaries and theme distribution predictions for multiple paragraphs in an article. "
        "Please combine them into a single summary (50-80 words) and a single theme distribution prediction to represent the entire article(e.g., politics 60%, economy 30%, technology 10%), retaining as much key information as possible and ensuring that information about the domain, style, and tone are preserved.\n"
        f"And translate the {dic.get(src_lang, src_lang)} summary to {dic.get(tgt_lang, tgt_lang)}.\n"
        "Format your output as a list of JSON. Like the following:\n"
        "[\n"
        "   {\n"
        f'       "Merged {dic.get(src_lang, src_lang)} summary": "A Merged {dic.get(src_lang, src_lang)} summary",\n'
        f'       "Merged {dic.get(tgt_lang, tgt_lang)} summary": "A Merged {dic.get(tgt_lang, tgt_lang)} summary",\n' 
        f'       "Merged {dic.get(tgt_lang, tgt_lang)} theme distribution prediction": "The article’s theme distribution in {dic.get(tgt_lang, tgt_lang)}"\n'
        "   }\n"
        "]"
    )


    USER_PROMPT = f"context: ```\n{input}\n```\n\noutput: "

    print(SYS_PROMPT + "\n" + USER_PROMPT)
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, is_filter = invoke_chat_api(SYS_PROMPT=SYS_PROMPT, user_prompt=USER_PROMPT, model=model)
        
        try:
            result = extract_summary_from_text(response, src_lang=src_lang, tgt_lang=tgt_lang, dic=dic)
            if result == []:  # 检查 item 数量是否匹配
                print(f"生成失败，重新生成...")
            else:
                return result
        except Exception as e:
            print(f"尝试 {attempt + 1}: 提取过程失败，错误: {e}，重新生成...")

        if attempt == max_attempts - 1:  # 最后一次尝试
            print(f"警告: 经过 {max_attempts} 次尝试，仍然生成失败，返回最后一次结果")
            return result if 'result' in locals() else []  # 返回最后一次结果，若无则返回空列表

    return []  # 默认返回空列表（理论上不会到达这里）




def TopicPrompt(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        "You are a professional article analysis agent. You are provided with a context (delimited by ```)"
        f"Use a few words to describe the topics of the following input {dic.get(src_lang, src_lang)} article.\n"
        f"And translate the {dic.get(src_lang, src_lang)} topics into {dic.get(tgt_lang, tgt_lang)}.\n"
        "Format your output as a list of JSON. Like the following:\n"
        "[\n"
        "   {\n"
        f'       "{dic.get(src_lang, src_lang)} topic": "The {dic.get(src_lang, src_lang)} topic",\n'
        f'       "{dic.get(tgt_lang, tgt_lang)} topic": "The {dic.get(tgt_lang, tgt_lang)} topic"\n'
        "   }, {...}\n"
        "]"
    )


    USER_PROMPT = f"context: ```\n{input}\n```\n\noutput: "
    print(SYS_PROMPT + "\n" + USER_PROMPT)

    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, is_filter = invoke_chat_api(SYS_PROMPT=SYS_PROMPT, user_prompt=USER_PROMPT, model=model)
        
        try:
            result = extract_topic_from_text(response, src_lang=src_lang, tgt_lang=tgt_lang, dic=dic)
            if result == []:  # 检查 item 数量是否匹配
                print(f"生成失败，重新生成...")
            else:
                return result
        except Exception as e:
            print(f"尝试 {attempt + 1}: 提取过程失败，错误: {e}，重新生成...")

        if attempt == max_attempts - 1:  # 最后一次尝试
            print(f"警告: 经过 {max_attempts} 次尝试，仍然生成失败，返回最后一次结果")
            return result if 'result' in locals() else []  # 返回最后一次结果，若无则返回空列表

    return []  # 默认返回空列表（理论上不会到达这里）


def HintPrompt(input_pre: str,input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        f"You are a professional article analysis agent. Please generate a hint for the input {dic.get(src_lang, src_lang)} paragraph, "
        "linking its logical relationship with the previous paragraph. The transition hint should be concise (20-30 words) and reflect logical connections (e.g., cause-effect, progression). You are provided with a context (delimited by ```) "
        f"Thought 1: Analyze the logical relationship between the two paragraphs (e.g., cause-effect, progression, contrast). "
        f"Thought 2: Generate a {dic.get(src_lang, src_lang)} language transition hint describing how the current paragraph follows the previous one. "
        f"Thought 3: Self-evaluate whether the transition hint accurately reflects the logical relationship; if inaccurate, optimize it. "
        f"Thought 4: Translate the {dic.get(src_lang, src_lang)} hint into {dic.get(tgt_lang, tgt_lang)}.\n"
        "You are provided with 2 paragraphs (delimited by ```). Format your output as a list of JSON. Like the following:\n"
        "[\n"
        "   {\n"
        f'       "{dic.get(src_lang, src_lang)} hint": "The {dic.get(src_lang, src_lang)} hint",\n'
        f'       "{dic.get(tgt_lang, tgt_lang)} hint": "The {dic.get(tgt_lang, tgt_lang)} hint"\n'
        "   }\n"
        "]"
    )


    USER_PROMPT = f"context: ```\nPrevious paragraph:\n{input_pre}\n\nCurrent paragraph:\n{input}\n```\n\noutput: "

    print(SYS_PROMPT + "\n" + USER_PROMPT)

    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, is_filter = invoke_chat_api(SYS_PROMPT=SYS_PROMPT, user_prompt=USER_PROMPT, model=model)
        
        try:
            result = extract_summary_from_text(response, src_lang=src_lang, tgt_lang=tgt_lang, dic=dic)
            if result == []:  # 检查 item 数量是否匹配
                print(f"生成失败，重新生成...")
            else:
                return result
        except Exception as e:
            print(f"尝试 {attempt + 1}: 提取过程失败，错误: {e}，重新生成...")

        if attempt == max_attempts - 1:  # 最后一次尝试
            print(f"警告: 经过 {max_attempts} 次尝试，仍然生成失败，返回最后一次结果")
            return result if 'result' in locals() else []  # 返回最后一次结果，若无则返回空列表

    return []  # 默认返回空列表（理论上不会到达这里）



def SummaryPrompt(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        "You are a professional article analysis agent. You are provided with a context (delimited by ```) "
        "and you need to summarize the input text into an abstract and a theme distribution prediction.\n"
        f"Thought 1: Generate a {dic.get(src_lang, src_lang)} language summary (30-60 words), including the main contents of these sentences, the overall domain, style, and tone, while preserving key information as much as possible.\n"
        f"Thought 2: translate the {dic.get(src_lang, src_lang)} summary to {dic.get(tgt_lang, tgt_lang)}.\n"
        f"Thought 3: Generate a theme distribution prediction: Output the article’s theme distribution in {dic.get(tgt_lang, tgt_lang)}(e.g., politics 60%, economy 30%, technology 10%).\n"
        "Format your output as a list of JSON. Like the following:\n"
        "[\n"
        "   {\n"
        f'       "{dic.get(src_lang, src_lang)} summary": "A {dic.get(src_lang, src_lang)} summary",\n'
        f'       "{dic.get(tgt_lang, tgt_lang)} summary": "A {dic.get(tgt_lang, tgt_lang)} summary",\n'
        f'       "Theme distribution prediction": "The article’s theme distribution in {dic.get(tgt_lang, tgt_lang)}"\n'
        "   }\n"
        "]"
    )


    USER_PROMPT = f"context: ```\n{input}\n```\n\noutput: "

    print(SYS_PROMPT + "\n" + USER_PROMPT)

    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, is_filter = invoke_chat_api(SYS_PROMPT=SYS_PROMPT, user_prompt=USER_PROMPT, model=model)
        
        try:
            result = extract_summary_from_text(response, src_lang=src_lang, tgt_lang=tgt_lang, dic=dic)
            if result == []:  # 检查 item 数量是否匹配
                print(f"生成失败，重新生成...")
            else:
                return result
        except Exception as e:
            print(f"尝试 {attempt + 1}: 提取过程失败，错误: {e}，重新生成...")

        if attempt == max_attempts - 1:  # 最后一次尝试
            print(f"警告: 经过 {max_attempts} 次尝试，仍然生成失败，返回最后一次结果")
            return result if 'result' in locals() else []  # 返回最后一次结果，若无则返回空列表

    return []  # 默认返回空列表（理论上不会到达这里）



def graphPrompt_with_summary(input: str, summary: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}):
    if model is None:
        model = "deepseek-r1:32b"


    # 使用摘要作为全局把控
    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 0: Use the summary as a global reference to understand the broader context of the entire paragraph. "
        "The summary helps identify key entities (e.g., persons, locations, organizations) or concepts that might be ambiguous or unrecognized in the context.\n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        f"Thought 4: For each extracted pair of terms and their relation, provide both the original {dic.get(src_lang, src_lang)} terms and relation (from the context) "
        f"and their {dic.get(tgt_lang, tgt_lang)} translations. The input context is in {dic.get(src_lang, src_lang)}, and you must preserve the original {dic.get(src_lang, src_lang)} alongside the {dic.get(tgt_lang, tgt_lang)} output.\n\n"
        "Format your output as a list of JSON. Each element of the list contains a pair of terms and their relation between them, "
        f"including both {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} versions, like the following:\n"
        "[\n"
        "   {\n"
        f'       "node_1_{src_lang}": "A concept from extracted ontology",\n'
        f'       "node_1_{tgt_lang}": "Translated concept in {dic.get(tgt_lang, tgt_lang)}",\n'
        f'       "node_2_{src_lang}": "A related concept from extracted ontology",\n'
        f'       "node_2_{tgt_lang}": "Translated concept in {dic.get(tgt_lang, tgt_lang)}",\n'
        f'       "edge_{src_lang}": "relationship between the two concepts"\n'
        f'       "edge_{tgt_lang}": "Translated relationship in {dic.get(tgt_lang, tgt_lang)}"\n'
        "   }, {...}\n"
        "]"
    )


    # 用户提示
    USER_PROMPT = f"context: ```\nSummary: {summary}\n\nContext:\n{input}\n```\n\noutput:"

    print(SYS_PROMPT + "\n" + USER_PROMPT)


    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, is_filter = invoke_chat_api(SYS_PROMPT=SYS_PROMPT, user_prompt=USER_PROMPT, model=model)
        
        try:
            result = extract_triplets_from_text(response, src_lang=src_lang, tgt_lang=tgt_lang)
            if result == []:  # 检查 item 数量是否匹配
                print(f"生成失败，重新生成...")
            else:
                return result
        except Exception as e:
            print(f"尝试 {attempt + 1}: 提取过程失败，错误: {e}，重新生成...")

        if attempt == max_attempts - 1:  # 最后一次尝试
            print(f"警告: 经过 {max_attempts} 次尝试，仍然生成失败，返回最后一次结果")
            return result if 'result' in locals() else []  # 返回最后一次结果，若无则返回空列表

    return []  # 默认返回空列表（理论上不会到达这里）




def InferenceFirstChunkPrompt(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, theme distribution prediction, "
        f"the current paragraph’s topic, and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags.\n"
        "Format your output as follows:\n"
        "[\n"
        "   {\n"
        f'       "{dic.get(src_lang, src_lang)} sentence": "First line",\n'
        f'       "{dic.get(tgt_lang, tgt_lang)} translation": "First line translated to {dic.get(tgt_lang, tgt_lang)}"\n'
        "   },\n"
        "   // ... one object per line\n"
        "]"
    )

    USER_PROMPT = f"context: ```\n{input}\n```\n\noutput:"

    print(SYS_PROMPT + "\n" + USER_PROMPT)
    
    max_attempts = 15  # 最大尝试次数
    for attempt in range(max_attempts):
        response, is_filter = invoke_chat_api(SYS_PROMPT=SYS_PROMPT, user_prompt=USER_PROMPT, model=model)


        if is_filter:
            return chunk_lines,False
        
        # 提取并验证
        expected_sentences = [re.sub(r'^<s>(.*?)</s>$', r'\1', line).strip() for line in chunk_lines] if chunk_lines else None
        result, is_valid = extract_inference_from_text(response, src_lang=src_lang, tgt_lang=tgt_lang, dic=dic, 
                                                       max_attempts=max_attempts, current_attempts=attempt+1, 
                                                       expected_sentences=expected_sentences)
        
        if result and is_valid:
            print(f"尝试 {attempt + 1}: 生成成功，句子顺序和内容匹配 ({len(result)})")
            return result, is_valid
        else:
            print(f"尝试 {attempt + 1}: 生成 {len(result)} 个 item，句子顺序或内容不匹配，重新生成...")
        if attempt == max_attempts - 1:
            print(f"警告: 经过 {max_attempts} 次尝试，仍未完全匹配，返回最后一次结果")
            return result, is_valid

    return [],False  # 默认返回空列表（理论上不会到达这里）





def InferencePrompt(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, theme distribution prediction, "
        f"transition hint between the previous and current paragraphs, the current paragraph’s topic, and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags.\n"
        "Format your output as follows:\n"
        "[\n"
        "   {\n"
        f'       "{dic.get(src_lang, src_lang)} sentence": "First line",\n'
        f'       "{dic.get(tgt_lang, tgt_lang)} translation": "First line translated to {dic.get(tgt_lang, tgt_lang)}"\n'
        "   },\n"
        "   // ... one object per line\n"
        "]"
    )



    USER_PROMPT = f"context: ```\n{input}\n```\n\noutput:"
    print(SYS_PROMPT + "\n" + USER_PROMPT)
    
    max_attempts = 15  # 最大尝试次数
    for attempt in range(max_attempts):
        response, is_filter = invoke_chat_api(SYS_PROMPT=SYS_PROMPT, user_prompt=USER_PROMPT, model=model)

        #生成的内容涉及不正当言论被过滤调了
        if is_filter:
            return chunk_lines,False
        
        # 提取并验证
        expected_sentences = [re.sub(r'^<s>(.*?)</s>$', r'\1', line).strip() for line in chunk_lines] if chunk_lines else None
        result, is_valid = extract_inference_from_text(response, src_lang=src_lang, tgt_lang=tgt_lang, dic=dic, 
                                                       max_attempts=max_attempts, current_attempts=attempt+1, 
                                                       expected_sentences=expected_sentences)
        
        if result and is_valid:
            print(f"尝试 {attempt + 1}: 生成成功，句子顺序和内容匹配 ({len(result)})")
            return result, is_valid
        else:
            print(f"尝试 {attempt + 1}: 生成 {len(result)} 个 item，句子顺序或内容不匹配，重新生成...")
        if attempt == max_attempts - 1:
            print(f"警告: 经过 {max_attempts} 次尝试，仍未完全匹配，返回最后一次结果")
            return result, is_valid

    return [],False  # 默认返回空列表（理论上不会到达这里）





