import sys
from yachalk import chalk
import re
from typing import List, Tuple, Dict, Optional, Any
sys.path.append("..")
import asyncio
import aiohttp
import json
import ollama.client as client

def normalize_sentence_for_comparison(sentence: str) -> str:
    """去除句子中的空格、标点符号和不可见字符，用于比较"""
    # 移除不可见字符（如零宽空格、换行符）
    sentence = re.sub(r'[\u200b-\u200f\u2028-\u202f\n\r]', '', sentence)
    # 去除所有空格
    sentence = re.sub(r'\s+', '', sentence)
    # 统一全角标点为半角，并去除所有标点
    sentence = sentence.replace('，', ',').replace('。', '.').replace('：', ':').replace('；', ';')
    sentence = sentence.replace('（', '(').replace('）', ')').replace('“', '"').replace('”', '"')
    sentence = sentence.replace('‘', "'").replace('’', "'")
    sentence = re.sub(r'[.,!?;:"\'“”‘’，。！？；：()\[\]{}]', '', sentence)
    return sentence

def fix_quotes(value: str, is_chinese: bool = False) -> str:
    """修复引号：补齐最外层引号后，转义内部引号，若为 Chinese sentence 则处理奇数引号"""
    value = value.strip()

    # 第一步：补齐最外层引号
    if not (value.startswith('"') and value.endswith('"')):
        if value.startswith('"'):
            value = value.rstrip('"') + '"'
        elif value.endswith('"'):
            value = '"' + value.lstrip('"')
        else:
            value = f'"{value}"'
    
    # 第二步：获取值（去掉最外层引号）
    inner_content = value[1:-1]
    
    # 第四步：若为 Chinese sentence，处理内部奇数引号
    if is_chinese:
        quote_count = inner_content.count('"')
        if quote_count % 2 == 1:
            last_quote_idx = inner_content.rfind('"')
            if last_quote_idx != -1:
                inner_content = inner_content[:last_quote_idx] + inner_content[last_quote_idx + 1:]
                # print(f"检测到内部奇数引号，移除一个引号，修复为: \"{inner_content}\"")
    
    # 返回带外层引号的结果
    return f'"{inner_content}"'


def extract_inference_from_text(text: str, src_lang: str = 'en', tgt_lang: str = 'zh', 
                               dic: Dict[str, str] = {}, max_attempts: int = 4, 
                               current_attempts: int = 0, expected_sentences: List[str] = None) -> tuple:
    """
    从大模型的返回文本中提取 JSON 数据，并验证内容是否按顺序匹配预期句子。
    比较时去除空格和标点符号。

    参数：
    text -- 模型返回的原始文本
    src_lang -- 源语言
    tgt_lang -- 目标语言
    dic -- 语言映射字典
    max_attempts -- 最大尝试次数
    current_attempts -- 当前尝试次数
    expected_sentences -- 预期的源句子列表（不含 <s></s>）

    返回值：
    (items, is_valid) -- 提取并修复后的 JSON 数据列表，以及是否匹配的标志
    """

    # 去除 <think>...</think> 之间的内容
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 获取动态 key
    src_key = f"{dic.get(src_lang, src_lang)} sentence"
    tgt_key = f"{dic.get(tgt_lang, tgt_lang)} translation"
    
    # 提取 JSON 部分
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if not match:
        print("未找到 JSON 数据")
        return [], False
    
    json_str = match.group(0)

    try:
        json_data = json.loads(json_str)
        if not isinstance(json_data, list):
            print("JSON 数据不是列表格式")
            return [], False
    except json.JSONDecodeError as e:
        print(f"整体 JSON 解析失败: {e}。开始逐项解析并修复！")
        
        items = []
        item_matches = re.findall(r"\{[^{}]*?\}", json_str, re.DOTALL)
        for item_str in item_matches:
            item_str = item_str.strip()
            # 修复 src_key 的值（从 "src_key": 到 ,"tgt_key":）
            item_str = re.sub(
                rf'("{src_key}":\s*)(.*?)(,\s*"{tgt_key}":)',
                lambda m: f'"{src_key}": {fix_quotes(m.group(2), is_chinese=(src_key == "Chinese sentence"))}{m.group(3)}',
                item_str,
                flags=re.DOTALL
            )
            # 修复 tgt_key 的值（从 ,"tgt_key": 到 }）
            item_str = re.sub(
                rf'("{tgt_key}":\s*)(.*?)(?=\s*\}})',
                lambda m: f'{m.group(1)}{fix_quotes(m.group(2), is_chinese=(src_key == "Chinese sentence"))}',
                item_str,
                flags=re.DOTALL
            )

            try:
                item = json.loads(item_str)
                if (isinstance(item, dict) and 
                    src_key in item and 
                    tgt_key in item):
                    items.append(item)
                else:
                    if current_attempts == max_attempts:
                        items.append({"key不匹配": item_str})
            except json.JSONDecodeError as item_e:
                print(f"逐项解析失败: {item_str}, 错误: {item_e}")
                if current_attempts == max_attempts:
                    items.append({"item解析失败": item_str})
        
    else:
        # 修复键名
        items = []
        for item in json_data:
            if isinstance(item, dict) and len(item) == 2:
                current_keys = list(item.keys())
                # 检查键名是否需要替换
                if current_keys[0] != src_key or current_keys[1] != tgt_key:
                    # 创建新字典，替换键名
                    new_item = {
                        src_key: item[current_keys[0]],
                        tgt_key: item[current_keys[1]]
                    }
                    items.append(new_item)
                else:
                    # 键名已正确，直接添加
                    items.append(item)
            else:
                # 非两个键的 item 保持不变
                items.append(item)

    # 移除 items 中 src_key 和 tgt_key 的 <s></s> 标签
    for item in items:
        if src_key in item:
            # 移除开头 <s> 和结尾 </s>
            item[src_key] = re.sub(r'^<s>', '', item[src_key])  # 移除开头 <s>
            item[src_key] = re.sub(r'</s>$', '', item[src_key]).strip()  # 移除结尾 </s> 并清理空格
        if tgt_key in item:
            # 对 tgt 应用相同的处理
            item[tgt_key] = re.sub(r'^<s>', '', item[tgt_key])
            item[tgt_key] = re.sub(r'</s>$', '', item[tgt_key]).strip()


    # 验证句子内容（顺序相关）
    if expected_sentences:
        extracted_sentences = [item.get(src_key, "") for item in items if src_key in item]
        
        # 规范化句子，去除空格和标点符号
        norm_extracted = [normalize_sentence_for_comparison(sent) for sent in extracted_sentences]
        norm_expected = [normalize_sentence_for_comparison(sent) for sent in expected_sentences]
        
        # 第一步：检查长度是否相同
        if len(norm_extracted) != len(norm_expected):
            print(f"句子数量不匹配！预期: {len(norm_expected)}，实际: {len(norm_extracted)}")
            print(f"预期（规范化）: {norm_expected}")
            print(f"实际（规范化）: {norm_extracted}")
            return items, False
        
        # 第二步：计算匹配比例
        total_sentences = len(norm_expected)
        match_count = sum(1 for extracted, expected in zip(norm_extracted, norm_expected) if extracted == expected)
        match_ratio = match_count / total_sentences if total_sentences > 0 else 0.0
        
        if match_ratio < 0.9:  # 90% 阈值
            print(f"句子内容匹配率不足 90%（实际: {match_ratio:.2%}）！")
            print(f"预期（规范化）: {norm_expected}")
            print(f"实际（规范化）: {norm_extracted}")
            return items, False
        
        print(f"句子内容匹配率: {match_ratio:.2%} ≥ 90%，认为列表匹配！")

    return items, True

def extract_topic_from_text(text, src_lang='en', tgt_lang='zh',dic={}):
    """
    从大模型的返回文本中提取 JSON 数据，并筛选出符合严格格式要求的项。
    适用于 deepseek-r1 模型。
    
    参数:
        text: 模型返回的文本
        src_lang: 源语言后缀（例如 'en'）
        tgt_lang: 目标语言后缀（例如 'zh'）
    
    返回:
        符合要求的 JSON 项列表
    """
    # 去除 <think>...</think> 之间的内容（如果存在）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # 提取 JSON 部分（匹配最外层的 `[` 和 `]` 之间的内容）
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if not match:
        print("未找到 JSON 数据")
        return []
    
    json_str = match.group(0)
    
    # 定义必需的键
    required_keys = {
        f"{dic.get(src_lang, src_lang)} topic", f"{dic.get(tgt_lang, tgt_lang)} topic"
    }
    
    def is_valid_item(item):
        """检查 item 是否符合要求：键集合匹配且值非空"""
        if not isinstance(item, dict) or set(item.keys()) != required_keys:
            return False
        # 检查值是否为空（None 或空字符串）
        return all(value is not None and value != "" for value in item.values())
    
    try:
        # 尝试整体解析 JSON
        json_data = json.loads(json_str)
        if not isinstance(json_data, list):
            print("JSON 数据不是列表格式")
            return []
        
        # 筛选符合要求的项
        valid_items = []
        for item in json_data:
            if is_valid_item(item):
                valid_items.append(item)
            else:
                print(f"跳过不符合格式或值为空的项: {item}")
        
        return valid_items
    
    except json.JSONDecodeError as e:
        print(f"整体 JSON 解析失败: {e}")
        # 逐项解析作为 fallback
        valid_items = []
        items = re.split(r'\s*},\s*', json_str.strip('[]'))
        for item_str in items:
            item_str = item_str.strip()
            if not item_str:
                continue
            if not item_str.startswith('{'):
                item_str = '{' + item_str
            if not item_str.endswith('}'):
                item_str += '}'
            
            # 修复缺少双引号的键名
            item_str = re.sub(r'([{,\s])(\w+)"(?=\s*:)', r'\1"\2"', item_str)
            
            try:
                item = json.loads(item_str)
                if is_valid_item(item):
                    valid_items.append(item)
                else:
                    print(f"跳过不符合格式或值为空的项: {item}")
            except json.JSONDecodeError as e2:
                print(f"逐项解析失败: {e2}, 项内容: {item_str}")
        
        return valid_items



def extract_summary_from_text(text: str, src_lang: str = 'en', tgt_lang: str = 'zh',dic={}):
    """
    从大模型的返回文本中提取 JSON 数据，并筛选出符合严格格式要求的项。

    要求：
        - JSON 是列表形式
        - 每项是字典
        - 必须包含 src_lang 和 tgt_lang 对应的 topic 键
        - 所有值必须是非空字符串

    参数:
        text: 模型返回的文本
        src_lang: 源语言后缀（例如 'en'）
        tgt_lang: 目标语言后缀（例如 'zh'）

    返回:
        符合要求的 JSON 项列表；否则返回空列表
    """
    # 显示语言名映射（可按需扩展）

    # 定义必需的键
    required_keys_list  = [
        {f"{dic.get(src_lang, src_lang)} summary", f"{dic.get(tgt_lang, tgt_lang)} summary", "Theme distribution prediction"},
        {f"Merged {dic.get(src_lang, src_lang)} summary",f"Merged {dic.get(tgt_lang, tgt_lang)} summary",f"Merged {dic.get(tgt_lang, tgt_lang)} theme distribution prediction"},
        {f"{dic.get(src_lang, src_lang)} proper nouns",f"{dic.get(tgt_lang, tgt_lang)} proper nouns"},
        {f"{dic.get(src_lang, src_lang)} hint",f"{dic.get(tgt_lang, tgt_lang)} hint"}
        ]

    def is_valid_item(item: Dict[str, Any]) -> bool:
        """检查项是否合法：键匹配，值为非空字符串"""
        if not isinstance(item, dict):
            print("项不是字典")
            return False
        if set(item.keys()) not in required_keys_list:
            print(f"键不匹配，实际为：{set(item.keys())}，应在这个中：{required_keys_list}")
            return False
        for key, value in item.items():
            if not isinstance(value, str) or value.strip() == "":
                print(f"字段 '{key}' 的值无效：{value}")
                return False
        return True

    # 清除 <think>...</think> 内容
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 提取 JSON（匹配最外层中括号）
    match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if not match:
        print("未找到 JSON 数据")
        return []

    json_str = match.group(0)

    try:
        json_data = json.loads(json_str)
        if not isinstance(json_data, list):
            print("JSON 不是列表格式")
            return []

        for item in json_data:
            if not is_valid_item(item):
                return []

        return json_data

    except json.JSONDecodeError as e:
        print(f"JSON 解码失败: {e}")
        return []




def extract_triplets_from_text(text, src_lang='en', tgt_lang='zh'):
    """
    从大模型的返回文本中提取 JSON 数据，并筛选出符合严格格式要求的项。
    适用于 deepseek-r1 模型。
    
    参数:
        text: 模型返回的文本
        src_lang: 源语言后缀（例如 'en'）
        tgt_lang: 目标语言后缀（例如 'zh'）
    
    返回:
        符合要求的 JSON 项列表
    """
    # 去除 <think>...</think> 之间的内容（如果存在）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # 提取 JSON 部分（匹配最外层的 `[` 和 `]` 之间的内容）
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if not match:
        print("未找到 JSON 数据")
        return []
    
    json_str = match.group(0)
    
    # 定义必需的键
    required_keys = {
        f"node_1_{src_lang}", f"node_1_{tgt_lang}",
        f"node_2_{src_lang}", f"node_2_{tgt_lang}",
        f"edge_{src_lang}", f"edge_{tgt_lang}"
    }
    
    def is_valid_item(item):
        """检查 item 是否符合要求：键集合匹配且值非空"""
        if not isinstance(item, dict) or set(item.keys()) != required_keys:
            return False
        # 检查值是否为空（None 或空字符串）
        return all(value is not None and value != "" for value in item.values())
    
    try:
        # 尝试整体解析 JSON
        json_data = json.loads(json_str)
        if not isinstance(json_data, list):
            print("JSON 数据不是列表格式")
            return []
        
        # 筛选符合要求的项
        valid_items = []
        for item in json_data:
            if is_valid_item(item):
                valid_items.append(item)
            else:
                print(f"跳过不符合格式或值为空的项: {item}")
        
        return valid_items
    
    except json.JSONDecodeError as e:
        print(f"整体 JSON 解析失败: {e}")
        # 逐项解析作为 fallback
        valid_items = []
        items = re.split(r'\s*},\s*', json_str.strip('[]'))
        for item_str in items:
            item_str = item_str.strip()
            if not item_str:
                continue
            if not item_str.startswith('{'):
                item_str = '{' + item_str
            if not item_str.endswith('}'):
                item_str += '}'
            
            # 修复缺少双引号的键名，并确保冒号存在
            item_str = re.sub(r'([{,\s])(\w+)"(?=\s*:)', r'\1"\2"', item_str)
            
            try:
                item = json.loads(item_str)
                if is_valid_item(item):
                    valid_items.append(item)
                else:
                    print(f"跳过不符合格式或值为空的项: {item}")
            except json.JSONDecodeError as e2:
                print(f"逐项解析失败: {e2}, 项内容: {item_str}")
        
        return valid_items

    



def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}):
    if model is None:
        model = "deepseek-r1:32b"

    # 动态生成 SYS_PROMPT，使用 dic.get() 避免 KeyError
    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
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

    USER_PROMPT = f"context: ```\n{input}\n```\n\noutput: "

    print(SYS_PROMPT + "\n" + USER_PROMPT)

    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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



def ProperNounsPrompt(input: str, summary: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}):
    if model is None:
        model = "deepseek-r1:32b"


    # 使用摘要作为全局把控
    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert. "
        f"You are provided with a context chunk (delimited by ```) containing a summary and a {dic.get(src_lang, src_lang)} paragraph. "
        "Thought 0: Use the summary as a global reference to understand the broader context of the entire paragraph. "
        "The summary helps identify key entities (e.g., persons, locations, organizations) or concepts that might be ambiguous or unrecognized in the context.\n"
        f"Thought 1: Extract all the proper nouns from the {dic.get(src_lang, src_lang)} paragraph and translate them to {dic.get(tgt_lang, tgt_lang)}, which don't include Pinyin or any romanization in the translations. "
        f"Ensure the {dic.get(tgt_lang, tgt_lang)} translations are accurate, consistent, and conform to the linguistic and cultural conventions of {dic.get(tgt_lang, tgt_lang)}.\n"
        "Format your output as a list of JSON. Like the following:\n"
        "[\n"
        "   {\n"
        f'       "{dic.get(src_lang, src_lang)} proper nouns": "A {dic.get(src_lang, src_lang)} proper nouns",\n'
        f'       "{dic.get(tgt_lang, tgt_lang)} proper nouns": "A {dic.get(tgt_lang, tgt_lang)} proper nouns",\n'
        "   }\n"
        "]"
    )


    # 用户提示
    USER_PROMPT = f"context: ```\nSummary: {summary}\n\nparagraph:\n{input}\n```\noutput:"

    print(SYS_PROMPT + "\n" + USER_PROMPT)


    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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





def CombineSummaryPrompt(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}):
    if model is None:
        model = "deepseek-r1:32b"

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
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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

    # 对 Topics 进行排序，按重要性从高到低输出，最多输出五个topics
    # SYS_PROMPT = (
    #     "You are a professional article analysis agent. You are provided with a context (delimited by ```).\n"
    #     f"Extract the topics from the following input {dic.get(src_lang, src_lang)} article and use a few words to describe each topic.\n"
    #     "Rank the extracted topics by their importance or relevance to the article, from highest to lowest.\n"
    #     "Limit your output to a maximum of 5 topics.\n"
    #     f"Translate the {dic.get(src_lang, src_lang)} topics into {dic.get(tgt_lang, tgt_lang)}.\n"
    #     "Format your output as a list of JSON, sorted by rank (highest to lowest), like the following:\n"
    #     "[\n"
    #     "   {\n"
    #     f'       "{dic.get(src_lang, src_lang)} topic": "The {dic.get(src_lang, src_lang)} topic",\n'
    #     f'       "{dic.get(tgt_lang, tgt_lang)} topic": "The {dic.get(tgt_lang, tgt_lang)} topic"\n'
    #     "   }, {...}\n"
    #     "]"
    # )


    USER_PROMPT = f"context: ```\n{input}\n```\n\noutput: "
    print(SYS_PROMPT + "\n" + USER_PROMPT)

    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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




"""
input都是 chunk 的长度，对翻译的输出划分为句子，但是会出错，很难得到准确的句子行数
"""
def InferenceFirstChunkPrompt_without_summary(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s theme distribution prediction, "
        f"the current paragraph’s topic, and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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



"""
input都是 chunk 的长度，对翻译的输出划分为句子，但是会出错，很难得到准确的句子行数
"""
def InferenceFirstChunkPrompt_without_theme(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, "
        f"the current paragraph’s topic, and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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




"""
input都是 chunk 的长度，对翻译的输出划分为句子，但是会出错，很难得到准确的句子行数
"""
def InferenceFirstChunkPrompt_without_topic(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, theme distribution prediction, "
        f"and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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




"""
input都是 chunk 的长度，对翻译的输出划分为句子，但是会出错，很难得到准确的句子行数
"""
def InferenceFirstChunkPrompt_without_triplets(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, theme distribution prediction, "
        f"and current paragraph’s topic. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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




def InferencePrompt_without_summary(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    # 系统提示
    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s theme distribution prediction, "
        f"transition hint between the previous and current paragraphs, the current paragraph’s topic, and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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




def InferencePrompt_without_theme(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    # 系统提示
    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, "
        f"transition hint between the previous and current paragraphs, the current paragraph’s topic, and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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





def InferencePrompt_without_hint(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    # 系统提示
    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, theme distribution prediction, "
        f"the current paragraph’s topic, and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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





def InferencePrompt_without_topic(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    # 系统提示
    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, theme distribution prediction, "
        f"transition hint between the previous and current paragraphs, and historical translations of some proper nouns. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, ensuring proper nouns remain consistent with their historical translations and maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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





def InferencePrompt_without_triplets(input: str, model="deepseek-r1:32b", src_lang='en', tgt_lang='zh', dic={}, line_count="20", chunk_lines: List[str] = None):
    if model is None:
        model = "deepseek-r1:32b"

    # 系统提示
    SYS_PROMPT = (
        f"You are a {dic.get(src_lang, src_lang)}-{dic.get(tgt_lang, tgt_lang)} bilingual expert translating a long {dic.get(src_lang, src_lang)} article. "
        f"You are provided with a context (delimited by ```) including the article’s {dic.get(src_lang, src_lang)} and {dic.get(tgt_lang, tgt_lang)} summaries, theme distribution prediction, "
        f"transition hint between the previous and current paragraphs, and current paragraph’s topic. "
        f"Translate the current {dic.get(src_lang, src_lang)} source paragraph into {dic.get(tgt_lang, tgt_lang)} as a cohesive unit, maintaining a consistent style across all sentences. "
        "The paragraph is provided with each sentence on a separate line, separated by '\\n', and wrapped with '<s>' and '</s>' tags to clearly indicate its start and end. "
        f"Your output must be a JSON list containing exactly {line_count} objects, each corresponding to one source sentence. "
        f"Each object must include the original {dic.get(src_lang, src_lang)} sentence (without '<s>' and '</s>' tags) under '{dic.get(src_lang, src_lang)} sentence' and its translation under '{dic.get(tgt_lang, tgt_lang)} translation' without '<s>' and '</s>' tags. "
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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
    
    max_attempts = 20  # 最大尝试次数
    for attempt in range(max_attempts):
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT, options={"num_predict": -1})
        
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

