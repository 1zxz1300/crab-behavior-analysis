#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import numpy as np
import os
import networkx as nx
import random
import matplotlib
import seaborn as sns
matplotlib.use("Agg")
from pathlib import Path
from matplotlib import pyplot as plt
from collections import deque
import csv


# ------------------------------------------------------------------
# 计算树结构的层级坐标（原始函数）
# ------------------------------------------------------------------
def hierarchy_pos(G, root=None, width=.5, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    为树形结构生成层级布局坐标。
    G: nx.Graph 或 nx.DiGraph
    root: 根节点名，比如 'Root'
    """

    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    # 确定根节点
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0,
                       xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)

        # 有向图用 successors，无向图用 neighbors
        if isinstance(G, nx.DiGraph):
            children = list(G.successors(root))
        else:
            children = list(G.neighbors(root))
            if parent is not None and parent in children:
                children.remove(parent)

        if len(children) != 0:
            dx = width / float(len(children))
            nextx = xcenter - width / 2.0 - dx / 2.0
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# ------------------------------------------------------------------
# 新增：从整棵树里计算“每个节点包含哪些 motif”
# ------------------------------------------------------------------
def compute_motif_sets(T, root="Root"):
    """
    改进版：计算每个节点下面包含的 motif（基序）集合。
    支持多种节点命名形式，例如：
      - "0", "1", "15"   → 代表真实基序编号
      - "leaf_right_3"   → 叶子节点，包含一个编号（3）
      - "h_18_13"        → 层级节点
    """
    from collections import deque
    import re

    parent = {root: None}
    children = {n: [] for n in T.nodes()}

    q = deque([root])
    while q:
        u = q.popleft()
        for v in T.neighbors(u):
            if v == parent.get(u):
                continue
            parent[v] = u
            children[u].append(v)
            q.append(v)

    motif_sets = {}

    def extract_motif_id(name):
        """提取节点名中可能的数字 ID"""
        if isinstance(name, int):
            return [name]
        if isinstance(name, str):
            # 纯数字节点
            if name.isdigit():
                return [int(name)]
            # leaf_xx_数字
            m = re.findall(r"\d+", name)
            if len(m) > 0:
                # 保留最后一个数字作为基序 ID
                return [int(m[-1])]
        return []

    def dfs(u):
        s = set()
        s.update(extract_motif_id(u))
        for v in children[u]:
            s |= dfs(v)
        motif_sets[u] = s
        return s

    dfs(root)

    # 深度计算
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            depth[v] = depth[u] + 1
            q.append(v)

    # 调试打印（确认确实提取到了内容）
    nonempty = {k: v for k, v in motif_sets.items() if len(v) > 0}
    print(f"[check] 已识别 {len(nonempty)} 个包含基序的节点样例：")
    for k, v in list(nonempty.items())[:5]:
        print(f"  {k}: {sorted(v)}")

    return motif_sets, parent, children, depth




# ------------------------------------------------------------------
# merge_func / graph_to_tree（基本保持你现在用的版本）
# ------------------------------------------------------------------
def merge_func(transition_matrix, n_cluster, motif_norm, merge_sel):
    """
    Merge node selection function.
    Added robust handling for NaN / zero divisions and initialization of merge_nodes.
    """
    merge_nodes = (np.array([0]), np.array([0]))  # 初始化，防止未定义
    cost_temp = np.inf  # 用 inf 代替 100，更安全

    if merge_sel == 0:
        # Merge nodes with the highest transition probability
        if np.all(np.isnan(transition_matrix)):
            print("Warning: transition_matrix contains only NaN values.")
            return merge_nodes
        cost = np.nanmax(transition_matrix)
        merge_nodes = np.where(cost == transition_matrix)
        return merge_nodes

    if merge_sel == 1:
        for i in range(n_cluster):
            for j in range(n_cluster):
                # 跳过对角线
                if i == j:
                    continue
                # 安全计算分母
                num = (motif_norm[i] + motif_norm[j])
                denom = np.abs(transition_matrix[i, j] + transition_matrix[j, i])
                if denom == 0 or np.isnan(denom):
                    continue
                cost = num / denom
                if np.isnan(cost):
                    continue
                if cost < cost_temp:
                    cost_temp = cost
                    merge_nodes = (np.array([i]), np.array([j]))

        # 如果循环完没找到合并节点，返回默认值
        if merge_nodes is None or cost_temp == np.inf:
            merge_nodes = (np.array([0]), np.array([0]))

    return merge_nodes


def graph_to_tree(motif_usage, transition_matrix, n_cluster, merge_sel=1):

    if merge_sel == 1:
        motif_usage_temp = motif_usage
        motif_usage_temp_colsum = motif_usage_temp.sum(axis=0)
        motif_norm = motif_usage_temp / motif_usage_temp_colsum
        motif_norm_temp = motif_norm.copy()
    else:
        motif_norm_temp = None

    merging_nodes = []
    hierarchy_nodes = []
    trans_mat_temp = transition_matrix.copy()
    is_leaf = np.ones((n_cluster), dtype='int')
    node_label = []
    leaf_idx = []

    transition_matrix = np.nan_to_num(transition_matrix)
    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1) == 0)
        reduction = len(temp) + 1
    else:
        reduction = 1

    for i in range(n_cluster - reduction):

        nodes = merge_func(trans_mat_temp, n_cluster, motif_norm_temp, merge_sel)

        if np.size(nodes) >= 2:
            nodes = np.array([nodes[0][0], nodes[1][0]])

        if is_leaf[nodes[0]] == 1:
            is_leaf[nodes[0]] = 0
            node_label.append('leaf_left_' + str(i))
            leaf_idx.append(1)

        elif is_leaf[nodes[0]] == 0:
            node_label.append('h_' + str(i) + '_' + str(nodes[0]))
            leaf_idx.append(0)

        if is_leaf[nodes[1]] == 1:
            is_leaf[nodes[1]] = 0
            node_label.append('leaf_right_' + str(i))
            hierarchy_nodes.append('h_' + str(i) + '_' + str(nodes[1]))
            leaf_idx.append(1)

        elif is_leaf[nodes[1]] == 0:
            node_label.append('h_' + str(i) + '_' + str(nodes[1]))
            hierarchy_nodes.append('h_' + str(i) + '_' + str(nodes[1]))
            leaf_idx.append(0)

        merging_nodes.append(nodes)

        node1_trans_x = trans_mat_temp[nodes[0], :]
        node2_trans_x = trans_mat_temp[nodes[1], :]

        node1_trans_y = trans_mat_temp[:, nodes[0]]
        node2_trans_y = trans_mat_temp[:, nodes[1]]

        new_node_trans_x = node1_trans_x + node2_trans_x
        new_node_trans_y = node1_trans_y + node2_trans_y

        trans_mat_temp[nodes[1], :] = new_node_trans_x
        trans_mat_temp[:, nodes[1]] = new_node_trans_y

        trans_mat_temp[nodes[0], :] = 0
        trans_mat_temp[:, nodes[0]] = 0

        trans_mat_temp[nodes[1], nodes[1]] = 0

        if merge_sel == 1:
            motif_norm_1 = motif_norm_temp[nodes[0]]
            motif_norm_2 = motif_norm_temp[nodes[1]]

            new_motif = motif_norm_1 + motif_norm_2

            motif_norm_temp[nodes[0]] = 0
            motif_norm_temp[nodes[1]] = 0

            motif_norm_temp[nodes[1]] = new_motif

    merge = np.array(merging_nodes)

    T = nx.Graph()

    T.add_node('Root')
    node_dict = {}

    if leaf_idx[-1] == 0:
        temp_node = 'h_' + str(merge[-1, 1]) + '_' + str(28)
        T.add_edge(temp_node, 'Root')
        node_dict[merge[-1, 1]] = temp_node

    if leaf_idx[-1] == 1:
        T.add_edge(merge[-1, 1], 'Root')

    if leaf_idx[-2] == 0:
        temp_node = 'h_' + str(merge[-1, 0]) + '_' + str(28)
        T.add_edge(temp_node, 'Root')
        node_dict[merge[-1, 0]] = temp_node

    if leaf_idx[-2] == 1:
        T.add_edge(merge[-1, 0], 'Root')

    idx = len(leaf_idx) - 3

    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1) == 0)
        reduction = len(temp) + 2
    else:
        reduction = 2

    for i in range(n_cluster - reduction)[::-1]:

        if leaf_idx[idx - 1] == 1:
            if merge[i, 1] in node_dict:
                T.add_edge(merge[i, 0], node_dict[merge[i, 1]])
            else:
                T.add_edge(merge[i, 0], temp_node)

        if leaf_idx[idx] == 1:
            if merge[i, 1] in node_dict:
                T.add_edge(merge[i, 1], node_dict[merge[i, 1]])
            else:
                T.add_edge(merge[i, 1], temp_node)

        if leaf_idx[idx] == 0:
            try:
                new_node = 'h_' + str(merge[i, 1]) + '_' + str(i)
            except (KeyError, IndexError):
                new_node = 'h_missing_' + str(i)

            if merge[i, 1] in node_dict:
                T.add_edge(node_dict[merge[i, 1]], new_node)
            else:
                T.add_edge(temp_node, new_node)

            if leaf_idx[idx - 1] == 1:
                temp_node = new_node
                node_dict[merge[i, 1]] = new_node
            else:
                try:
                    new_node_2 = 'h_' + str(merge[i, 0]) + '_' + str(i)
                except (KeyError, IndexError):
                    new_node_2 = 'h_missing_' + str(i)

                if merge[i, 1] in node_dict:
                    T.add_edge(node_dict[merge[i, 1]], new_node_2)
                else:
                    T.add_edge(new_node, new_node_2)

                node_dict[merge[i, 1]] = new_node
                node_dict[merge[i, 0]] = new_node_2

        elif leaf_idx[idx - 1] == 0:
            # 保留原逻辑
            pass

    return T


# ------------------------------------------------------------------
# 画树 + 新增导出 node->motif 映射
# ------------------------------------------------------------------
def draw_tree(T, config, n_cluster):
    import yaml, os
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import networkx as nx
    import numpy as np

    # 读 config，确定输出目录（保持和原版一样）
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    project_path = config["project_path"]
    model_name = config.get("Model", config.get("model_name", "VAME"))
    out_dir = os.path.join(
        project_path,
        f"results/{model_name}/hmm-{n_cluster}",
        "figures",
    )
    os.makedirs(out_dir, exist_ok=True)

    # 计算节点坐标
    pos = hierarchy_pos(T, "Root", width=.5, vert_gap=0.1, vert_loc=0, xcenter=50)

    # ========= 只画“基序树 + 基序编号” =========
    fig = plt.figure(figsize=(20, 12))

    # 1. 所有边
    nx.draw_networkx_edges(T, pos, width=1.0)

    # 2. 区分叶子 / 内部节点
    leaf_nodes = [n for n in T.nodes() if T.degree[n] == 1 and n != "Root"]
    internal_nodes = [n for n in T.nodes() if n not in leaf_nodes]

    # 3. 画内部节点（不写字）
    nx.draw_networkx_nodes(
        T, pos,
        nodelist=internal_nodes,
        node_size=200,
        node_color="white",
        edgecolors="black",
        linewidths=0.8,
    )

    # 4. 画叶子节点
    nx.draw_networkx_nodes(
        T, pos,
        nodelist=leaf_nodes,
        node_size=450,
        node_color="white",
        edgecolors="black",
        linewidths=0.8,
    )

    # 5. 找出“真正的 HMM 基序节点”：在 VAME 里原始基序就是整数 0..n_cluster-1
    motif_nodes = [n for n in T.nodes()
                   if isinstance(n, (int, np.integer)) and 0 <= int(n) < n_cluster]

    # 6. 只给这些节点打标签（用它自己的编号）
    labels = {n: str(int(n)) for n in motif_nodes}
    nx.draw_networkx_labels(T, pos, labels=labels, font_size=8)

    plt.title(f"Behavior Motif Hierarchy (n_clusters={n_cluster})")
    save_path = os.path.join(out_dir, "hierarchy_tree_motifs_only.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[tree] Saved motif-only tree to: {save_path}")




# ------------------------------------------------------------------
# 下面 traverse_* 函数维持你原来的版本（可选是否使用）
# ------------------------------------------------------------------
def traverse_tree(T, root_node=None):
    if root_node == None:
        node=['Root']
    else:
        node=['root_node']
    traverse_list = []
    traverse_preorder = '{'
    
    def _traverse_tree(T, node, traverse_preorder):
        traverse_preorder += str(node[0])
        traverse_list.append(node[0])
        children = list(T.neighbors(node[0]))
        
        if len(children) == 3:
            for child in children:
                if child in traverse_list:
                    children.remove(child)
            
        if len(children) > 1:
            traverse_preorder += '{'
            traverse_preorder_temp = _traverse_tree(T, [children[0]], '')
            traverse_preorder += traverse_preorder_temp
             
            traverse_preorder += '}{'
            
            traverse_preorder_temp = _traverse_tree(T, [children[1]], '')
            traverse_preorder += traverse_preorder_temp
            traverse_preorder += '}'
        
        return traverse_preorder
    
    traverse_preorder = _traverse_tree(T, node, traverse_preorder)
    traverse_preorder += '}'
    
    return traverse_preorder


def _traverse_tree(T, node, traverse_preorder,traverse_list):
    traverse_preorder += str(node[0])
    traverse_list.append(node[0])
    children = list(T.neighbors(node[0]))
    
    if len(children) == 3:
        for child in children:
            if child in traverse_list:
                children.remove(child)
        
    if len(children) > 1:
        traverse_preorder += '{'
        traverse_preorder_temp = _traverse_tree(T, [children[0]], '',traverse_list)
        traverse_preorder += traverse_preorder_temp
         
        traverse_preorder += '}{'
        
        traverse_preorder_temp = _traverse_tree(T, [children[1]], '',traverse_list)
        traverse_preorder += traverse_preorder_temp
        traverse_preorder += '}'
    
    return traverse_preorder
    
def traverse_tree(T, root_node=None):
    if root_node == None:
        node=['Root']
    else:
        node=[root_node]
    traverse_list = []
    traverse_preorder = '{'
    traverse_preorder = _traverse_tree(T, node, traverse_preorder,traverse_list)
    traverse_preorder += '}'
    
    return traverse_preorder


def _traverse_tree_cutline(
    T,
    node,
    traverse_list,
    cutline,
    level,
    community_bag,
    community_list=None,
):
    cur = node[0]

    if cur in traverse_list:
        return community_bag

    traverse_list.append(cur)

    if community_list is not None and isinstance(cur, int):
        community_list.append(cur)

    children = [c for c in T.neighbors(cur) if c not in traverse_list]

    if len(children) == 0:
        return community_bag

    if len(children) > 1:
        if nx.shortest_path_length(T, "Root", cur) == cutline:
            left_nodes = []
            right_nodes = []
            community_bag = _traverse_tree_cutline(
                T, [children[0]], traverse_list, cutline, level + 1,
                community_bag, left_nodes
            )
            community_bag = _traverse_tree_cutline(
                T, [children[1]], traverse_list, cutline, level + 1,
                community_bag, right_nodes
            )

            joined = left_nodes + right_nodes
            if joined:
                community_bag.append(joined)

            if isinstance(cur, int):
                community_bag.append([cur])
        else:
            community_bag = _traverse_tree_cutline(
                T, [children[0]], traverse_list, cutline, level + 1,
                community_bag, community_list
            )
            community_bag = _traverse_tree_cutline(
                T, [children[1]], traverse_list, cutline, level + 1,
                community_bag, community_list
            )

    elif len(children) == 1:
        community_bag = _traverse_tree_cutline(
            T, [children[0]], traverse_list, cutline, level + 1,
            community_bag, community_list
        )

    return community_bag


def traverse_tree_cutline(T, root_node=None, cutline=2):
    if root_node is None:
        node = ["Root"]
    else:
        node = [root_node]

    traverse_list = []
    community_bag = []
    level = 0

    community_bag = _traverse_tree_cutline(
        T, node, traverse_list, cutline, level, community_bag, []
    )

    return community_bag


    