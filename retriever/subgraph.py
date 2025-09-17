"""
启发式搜索逻辑
- 对于 anchor node 进行一跳扩展
- 对于扩展后的结果进行 连通
"""
import sys
import json
import os
import tqdm
import pandas as pd

from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType
from utils import codegraph_to_nxgraph

################################# 子图重构代码 #################################
def get_path_to_repo(node, pre_node_dict, graph_nx):
    """获取该节点到 repo 的路径
    :param node -> CodeGraph Node 采样出的子图节点
    :param pre_node_dict -> list(Node) 每个节点
    :return
    """
    if node.get_type() == NodeType.REPO:
        return [node]
    
    pre_nodes = list()
    if node.node_id in pre_node_dict:
        pre_nodes = pre_node_dict[node.node_id]
    else:
        for pre_node in graph_nx.predecessors(node):
            # 判断 Edge 类型 - contains
            if graph_nx[pre_node][node][0]['type'] == EdgeType.CONTAINS:
                pre_nodes.append(pre_node)
                if pre_node.get_type() != NodeType.REPO:
                    pre_nodes.extend(get_path_to_repo(pre_node, pre_node_dict, graph_nx))
                pre_node_dict[node.node_id] = pre_nodes
                break

    return pre_nodes

def reconstruct_graph(subgraph_nodes, graph_nx, pre_node_dict):
    """
    根据所给节点重构 连通 的 CodeGraph
    pre_node_dict 全局复用
    """
    
    nodes = subgraph_nodes
    all_nodes = set(nodes)
    for node in nodes:
        pre_nodes = get_path_to_repo(node, pre_node_dict, graph_nx)
        all_nodes |= set(pre_nodes)

    # 根据节点裁剪子图
    subgraph = graph_nx.subgraph(list(all_nodes))
    
    return subgraph

################################# 子图重构代码 #################################

################################# BFS代码 #################################
def bfs_expand(graph_nx, subgraph_nodes, hops=1):
    """
    通过 bfs 扩展
    - 最笼统的版本：不区分方向的1-hop
    :param graph_nx nx格式的原图
    :param subgraph_nodes 需要扩展的节点
    :param hops 需要扩展的跳数
    """

    seed_node = subgraph_nodes
    visited_node = set()
    # 记录所有被 nhop 覆盖的节点
    nhops_neighbors = set([node.node_id for node in seed_node])
    
    for hop_idx in range(hops):
        tmp_seed_node = []
        for node in seed_node:
            if node.node_id in visited_node:
                continue
            visited_node.add(node.node_id)
            suc_nodes = graph_nx.successors(node)
            pre_nodes = graph_nx.predecessors(node)
            for node in suc_nodes:
                tmp_seed_node.append(node)
                nhops_neighbors.add(node.node_id)
            
            for node in pre_nodes:
                tmp_seed_node.append(node)
                nhops_neighbors.add(node.node_id)
        
        seed_node = tmp_seed_node
    return nhops_neighbors

def bfs_expand_file(graph_nx, subgraph_nodes, hops=1):
    """
    通过 bfs 扩展
    - 限制 File 遍历2跳
    :param graph_nx nx格式的原图
    :param subgraph_nodes 需要扩展的节点
    :param hops 需要扩展的跳数
    """

    seed_node = subgraph_nodes
    visited_node = set()
    nhops_neighbors = set([node.node_id for node in seed_node])
    
    for hop_idx in range(hops):
        tmp_seed_node = []
        for node in seed_node:
            if node.node_id in visited_node:
                continue
            visited_node.add(node.node_id)
            suc_nodes = graph_nx.successors(node)
            pre_nodes = graph_nx.predecessors(node)
            for node in suc_nodes:
                if node.get_type() == NodeType.FILE:
                    tmp_seed_node.append(node)
                nhops_neighbors.add(node.node_id)
            
            for node in pre_nodes:
                if node.get_type() == NodeType.FILE:
                    tmp_seed_node.append(node)
                nhops_neighbors.add(node.node_id)
        
        seed_node = tmp_seed_node
    return nhops_neighbors
################################# BFS代码 #################################

################################# 辅助函数 #################################
def get_graph_file_name(item):
    """
    生成 graph_file_name
    """
    repo = item.repo
    repo = repo.replace("/", "#", 1)
    base_commit = item.base_commit
    return repo + "#" + base_commit + ".graph.json"
################################# 辅助函数 #################################

if __name__ == "__main__":

    # 数据变量定义
    test_basic_df = pd.read_json("/test_lite_basic_info.json")
    test_basic_df["graph_file"] = test_basic_df.apply(lambda item: get_graph_file_name(item), axis=1)
    
    graph_data_path = "/swe-bench-lite3/"
    anchor_node_path = "/anchor_nodes.json"

    with open(anchor_node_path, "r", encoding="utf-8") as file:
        anchor_node_dict = json.load(file)

    subgraph_id_dict = {}

    for idx, item in tqdm.tqdm(test_basic_df.iterrows()):

        instance_id = item.instance_id
        graph_file = item.graph_file
        tmp_graph_data_path = graph_data_path + graph_file

        # 解析图数据
        graph = parse(tmp_graph_data_path)
        graph_nx = codegraph_to_nxgraph(graph)

        # 获取 anchor_nodes
        anchor_nodes_raw = anchor_node_dict[instance_id]
        extractor_anchors = anchor_nodes_raw["extractor_anchor_nodes"]
        inferer_anchors = [node for node_list in anchor_nodes_raw["inferer_anchor_nodes"] for node in node_list]
        anchor_nodes = list(set(extractor_anchors + inferer_anchors))

        # 先 bfs 再 reconstruct
        anchor_nodes = [graph.get_node_by_id(node_id) for node_id in anchor_nodes]
        expanded_nodes = bfs_expand_file(graph_nx, anchor_nodes, hops=2)

        expanded_nodes = [graph.get_node_by_id(node_id) for node_id in expanded_nodes]
        
        pre_node_dict = {}
        subgraph = reconstruct_graph(expanded_nodes, graph_nx, pre_node_dict)

        result_nodes = subgraph.nodes()
        
        # 获取子图的节点id
        result_nodes = [node.node_id for node in result_nodes if node.get_type() == NodeType.FILE]
        
        subgraph_id_dict[instance_id] = list(result_nodes)
        
        # save
        with open("subgraph_nodes.json", 'w', encoding='utf-8') as file:
            json.dump(subgraph_id_dict, file)