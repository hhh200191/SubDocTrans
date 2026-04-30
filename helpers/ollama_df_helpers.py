import uuid
import pandas as pd
import numpy as np
from .ollama_prompts import extractConcepts
from .ollama_prompts import graphPrompt, SummaryPrompt, TopicPrompt, HintPrompt, graphPrompt_with_summary, ProperNounsPrompt


def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df


def df2ConceptsList(dataframe: pd.DataFrame) -> list:
    # dataframe.reset_index(inplace=True)
    results = dataframe.apply(
        lambda row: extractConcepts(
            row.text, {"chunk_id": row.chunk_id, "type": "concept"}
        ),
        axis=1,
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe


def df2Graph(dataframe: pd.DataFrame, model=None, src_lang='en', tgt_lang='zh', dic={}) -> list:
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model, src_lang, tgt_lang, dic), axis=1
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


def df2Proper_nouns(dataframe: pd.DataFrame, model=None, src_lang='en', tgt_lang='zh', dic={}) -> list:
    results = dataframe.apply(
        lambda row: ProperNounsPrompt(row.text, row.summary, model, src_lang, tgt_lang, dic), axis=1
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



def df2Inference(dataframe: pd.DataFrame, model=None, src_lang='en', tgt_lang='zh', dic={}) -> list:
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





def graph2Df(nodes_list, src_lang='en', tgt_lang='zh') -> tuple[pd.DataFrame, dict]:
    if not nodes_list:
        print("Warning: nodes_list is empty.")
        return pd.DataFrame(), {}
    
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    
    node1_lang1 = f"node_1_{src_lang}"
    node2_lang1 = f"node_2_{src_lang}"
    edge_lang1 = f"edge_{src_lang}"
    node1_lang2 = f"node_1_{tgt_lang}"
    node2_lang2 = f"node_2_{tgt_lang}"
    edge_lang2 = f"edge_{tgt_lang}"

    # 检查所需列是否存在
    required_columns = [node1_lang1, node2_lang1, edge_lang1]
    missing_columns = [col for col in required_columns if col not in graph_dataframe.columns]
    if missing_columns:
        print(f"Error: Missing columns in graph_dataframe: {missing_columns}")
        return pd.DataFrame(), {}

    # 过滤掉 node_1_{src_lang} 和 node_1_{tgt_lang} 相同的行
    graph_dataframe = graph_dataframe[
        (graph_dataframe[node1_lang1] != graph_dataframe[node1_lang2]) | 
        (graph_dataframe[node1_lang1].isna()) | 
        (graph_dataframe[node1_lang2].isna())
    ]

    lang1_columns = [node1_lang1, node2_lang1, edge_lang1]
    graph_df_lang1 = graph_dataframe[lang1_columns].dropna(subset=[node1_lang1, node2_lang1])
    graph_df_lang1[node1_lang1] = graph_df_lang1[node1_lang1].apply(lambda x: x)
    graph_df_lang1[node2_lang1] = graph_df_lang1[node2_lang1].apply(lambda x: x)
    graph_df_lang1 = graph_df_lang1.rename(columns={
        node1_lang1: "node_1",
        node2_lang1: "node_2",
        edge_lang1: "edge"
    })

    mapping_dict = {}
    for _, row in graph_dataframe.iterrows():
        # 遇到相同的节点和边的话，只保留第一次出现的翻译
        if pd.notna(row[node1_lang1]) and pd.notna(row[node1_lang2]) and row[node1_lang1] not in mapping_dict:
            mapping_dict[row[node1_lang1]] = row[node1_lang2]
        if pd.notna(row[node2_lang1]) and pd.notna(row[node2_lang2]) and row[node2_lang1] not in mapping_dict:
            mapping_dict[row[node2_lang1]] = row[node2_lang2]
        if pd.notna(row[edge_lang1]) and pd.notna(row[edge_lang2]) and row[edge_lang1] not in mapping_dict:
            mapping_dict[row[edge_lang1]] = row[edge_lang2]
    
    return graph_df_lang1, mapping_dict








def inference(dataframe: pd.DataFrame, model=None, src_lang='en', tgt_lang='zh', dic={}) -> list:
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