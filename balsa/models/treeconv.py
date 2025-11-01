# Copyright 2022 The Balsa Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn

from balsa.util import plans_lib

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TreeConvolution(nn.Module):
    """Balsa 的树卷积神经网络：将 (查询特征, 执行计划树) 映射为一个标量值。

    输出的值可以是执行代价（cost）或延迟（latency），用于查询优化中的计划评估。
    该网络结合了查询的全局特征与计划树的局部结构信息，通过树卷积操作进行联合建模。
    """

    def __init__(self, feature_size, plan_size, label_size, version=None):
        """初始化树卷积网络。

        Args:
            feature_size (int): 查询特征向量的维度（即 query_feats 的最后一维大小）。
            plan_size (int): 每个计划树节点的特征维度（即 trees 的第二维大小）。
            label_size (int): 输出标签的维度（通常为 1，表示预测的代价或延迟）。
            version (any): 保留参数，当前仅支持默认值 None。
        """
        super(TreeConvolution, self).__init__()
        # 当前仅支持默认版本（version=None），其他值会触发断言错误
        assert version is None, f"Unsupported version: {version}"

        # 1. 查询特征编码器（MLP）：将原始查询特征映射为高维嵌入
        # 输入: [batch_size, feature_size]
        # 输出: [batch_size, 32]，后续将广播到每个树节点
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LayerNorm(128),          # 层归一化，提升训练稳定性
            nn.LeakyReLU(),             # 激活函数，避免梯度消失
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )

        # 2. 树卷积主干网络：处理 (查询嵌入 + 计划树节点特征) 的组合
        # 输入通道数 = 32（查询嵌入） + plan_size（计划节点特征）
        self.conv = nn.Sequential(
            TreeConv1d(32 + plan_size, 512),   # 树结构卷积层
            TreeStandardize(),                 # 树结构标准化（类似 BatchNorm，但适配树）
            TreeAct(nn.LeakyReLU()),           # 树结构激活函数包装器
            TreeConv1d(512, 256),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(256, 128),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeMaxPool(),                     # 树结构最大池化，将整棵树压缩为一个向量
        )

        # 3. 输出 MLP：将池化后的全局表示映射为最终预测值
        self.out_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, label_size),  # 通常 label_size=1
        )

        # 初始化网络权重
        self.reset_weights()

    def reset_weights(self):
        """自定义权重初始化策略，遵循类似 Transformer 的初始化方式。

        - 权重矩阵：从 N(0, 0.02²) 正态分布初始化
        - 偏置项（bias）：初始化为 0
        - LayerNorm 的缩放参数（weight）：初始化为 1
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # 多维参数（如 Linear 的 weight、Embedding）→ 正态初始化
                nn.init.normal_(p, std=0.02)
            elif 'bias' in name:
                # 所有偏置项 → 零初始化
                nn.init.zeros_(p)
            else:
                # 假设其余标量参数为 LayerNorm 的 weight → 初始化为 1
                # 注：此处依赖命名约定（如 'norm.weight'）
                nn.init.ones_(p)

    def forward(self, query_feats, trees, indexes):
        """前向传播：联合编码查询与计划树，输出预测值。

        Args:
            query_feats: 查询的特征向量，形状为 [batch_size, feature_size]。
            trees: 计划树的节点特征矩阵，形状为 [batch_size, plan_size, max_tree_nodes]。
                   每棵树被展平为节点序列，缺失节点用 padding 补零。
            indexes: 树结构索引信息，用于 TreeConv1d 等模块定位父子关系。
                     具体格式由 TreeConv 实现决定（通常为节点父指针或邻接信息）。

        Returns:
            out: 预测的代价或延迟，形状为 [batch_size, label_size]（通常为 [B, 1]）。
        """
        # Step 1: 编码查询特征，并扩展为与树节点对齐的形状
        # query_feats: [B, F] → unsqueeze → [B, 1, F]
        # query_mlp → [B, 1, 32] → transpose → [B, 32, 1]
        query_embs = self.query_mlp(query_feats.unsqueeze(1))
        query_embs = query_embs.transpose(1, 2)  # [B, 32, 1]

        # 广播查询嵌入到每个树节点位置
        max_subtrees = trees.shape[-1]  # 最大树节点数（含 padding）
        # 扩展为 [B, 32, max_subtrees]，每个节点都“看到”相同的查询上下文
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                       max_subtrees)

        # Step 2: 拼接查询嵌入与计划树节点特征
        # trees: [B, plan_size, max_subtrees]
        # concat: [B, 32 + plan_size, max_subtrees]
        concat = torch.cat((query_embs, trees), axis=1)

        # Step 3: 通过树卷积主干网络处理
        # TreeConv 模块通常接收 (特征张量, 索引信息) 元组
        out = self.conv((concat, indexes))  # 输出: [B, 128]

        # Step 4: 通过输出 MLP 得到最终预测
        out = self.out_mlp(out)  # [B, label_size]

        return out


# =============================================================================
# 树结构专用神经网络模块（Tree-aware Neural Modules）
# =============================================================================

class TreeConv1d(nn.Module):
    """适配树形数据的 1D 卷积层。

    与标准 Conv1d 不同，TreeConv1d 利用树的结构信息（通过 indexes），
    对每个节点及其两个子节点（共3个节点）进行卷积操作。
    """

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self._in_dims = in_dims   # 输入特征维度（每个节点的特征数）
        self._out_dims = out_dims # 输出特征维度

        # 使用标准 Conv1d，但 kernel_size=3, stride=3，
        # 每次处理连续3个节点（父+左子+右子）
        self.weights = nn.Conv1d(in_dims, out_dims, kernel_size=3, stride=3)

    def forward(self, trees):
        """前向传播。

        Args:
            trees: 元组 (data, indexes)
                - data: [B, in_dims, N]，N 为最大节点数（含 padding）
                - indexes: [B, N, 3]，每个节点对应的 (自身, 左子, 右子) 的索引（从1开始）

        Returns:
            (feats, indexes): 
                - feats: [B, out_dims, N+1]，第0位为填充的零向量（对应无效节点0）
                - indexes: 原样返回，供后续层使用
        """
        data, indexes = trees

        # Step 1: 根据 indexes 从 data 中 gather 出每个节点及其子节点的特征
        # indexes: [B, N, 3] → expand to [B, N, 3*in_dims] → transpose to [B, 3*in_dims, N]
        # torch.gather 按索引从 data 的最后一维（节点维度）取值
        gathered = torch.gather(
            data, 2,
            indexes.expand(-1, -1, self._in_dims).transpose(1, 2)
        )  # shape: [B, in_dims, N*3]

        # Step 2: 应用卷积（kernel=3, stride=3），将每3个连续节点（父+左+右）映射为一个输出
        feats = self.weights(gathered)  # shape: [B, out_dims, N]

        # Step 3: 在输出开头添加一个零向量（对应索引0，表示“空节点”）
        zeros = torch.zeros((data.shape[0], self._out_dims), device=DEVICE).unsqueeze(2)  # [B, out_dims, 1]
        feats = torch.cat((zeros, feats), dim=2)  # [B, out_dims, N+1]

        return feats, indexes


class TreeMaxPool(nn.Module):
    """树结构最大池化层：对每棵树的所有节点特征取最大值，生成全局表示。"""

    def forward(self, trees):
        """前向传播。

        Args:
            trees: 元组 (data, indexes)，此处仅使用 data

        Returns:
            values: [B, C]，每棵树在每个通道上的最大值
        """
        # trees[0]: [B, C, N]
        return trees[0].max(dim=2).values  # 在节点维度（dim=2）上取最大值


class TreeAct(nn.Module):
    """树结构激活函数包装器：对树节点特征应用激活函数，保留结构索引。"""

    def __init__(self, activation):
        super().__init__()
        self.activation = activation  # 例如 nn.LeakyReLU()

    def forward(self, trees):
        """前向传播。

        Args:
            trees: 元组 (data, indexes)

        Returns:
            (activated_data, indexes)
        """
        return self.activation(trees[0]), trees[1]


class TreeStandardize(nn.Module):
    """树结构标准化层：对每棵树的所有节点特征进行全局标准化（减均值，除标准差）。"""

    def forward(self, trees):
        """前向传播。

        Args:
            trees: 元组 (data, indexes)

        Returns:
            (standardized_data, indexes)
        """
        data = trees[0]  # [B, C, N]

        # 计算整棵树（所有节点、所有通道）的均值和标准差
        mu = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)  # [B, 1, 1]
        s = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)    # [B, 1, 1]

        # 标准化（加 epsilon 防除零）
        standardized = (data - mu) / (s + 1e-5)

        return standardized, trees[1]


# =============================================================================
# 模型工具函数
# =============================================================================

def ReportModel(model, blacklist=None):
    """打印模型参数量和内存占用，并输出模型结构。

    Args:
        model: PyTorch 模型
        blacklist: 可选字符串，若参数名包含此字符串则不计入统计（如 'embedding'）

    Returns:
        mb: 模型参数占用的内存（MB，假设 float32）
    """
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))  # 参数数量 = 各维度乘积
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024  # float32 每参数占4字节
    print('number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


# =============================================================================
# 树数据预处理工具函数，其调用关系见 https://icnwjfenk3wd.feishu.cn/wiki/OD6ewznXQiqil6kybhec1Jdbndc
# =============================================================================

def _batch(data):
    """将变长的树节点特征列表批量化为统一张量（padding 到最大长度）。

    Args:
        data: List[np.ndarray]，每个元素形状为 [num_nodes, feature_dim]

    Returns:
        np.ndarray: 形状 [batch_size, max_num_nodes, feature_dim]
    """
    lens = [vec.shape[0] for vec in data]
    if len(set(lens)) == 1:
        # 所有树节点数相同，直接堆叠
        return np.asarray(data)
    # 否则创建零填充张量
    xs = np.zeros((len(data), np.max(lens), data[0].shape[1]), dtype=np.float32)
    for i, vec in enumerate(data):
        xs[i, :vec.shape[0], :] = vec
    return xs


# @profile
def _make_preorder_ids_tree(curr, root_index=1):
    """为树生成前序遍历的节点 ID 结构。

    Args:
        curr: 当前树节点（具有 .children 属性）
        root_index: 当前子树根节点的 ID（从1开始）

    Returns:
        tuple: (tree_structure, max_id)
            - tree_structure: 递归三元组 (my_id, left_subtree, right_subtree)
                              叶子节点表示为 (my_id, 0, 0)
            - max_id: 该子树中最大的节点 ID
    """
    if not curr.children:
        # 叶子节点
        return (root_index, 0, 0), root_index
    # 递归处理左右子树
    lhs, lhs_max_id = _make_preorder_ids_tree(curr.children[0], root_index=root_index + 1)
    rhs, rhs_max_id = _make_preorder_ids_tree(curr.children[1], root_index=lhs_max_id + 1)
    return (root_index, lhs, rhs), rhs_max_id


# @profile
def _walk(curr, vecs):
    """前序遍历树结构，将每个节点的 (自身ID, 左子ID, 右子ID) 展平存入列表。

    Args:
        curr: 当前节点的三元组结构 (id, left, right)
        vecs: 输出列表（引用传递）
    """
    if curr[1] == 0:
        # 叶子节点：只记录自身ID（左右子为0）
        vecs.append(curr)
    else:
        # 内部节点：记录 (自身ID, 左子根ID, 右子根ID)
        vecs.append((curr[0], curr[1][0], curr[2][0]))
        _walk(curr[1], vecs)  # 递归左子树
        _walk(curr[2], vecs)  # 递归右子树


# @profile
def _make_indexes(root):
    """为单棵树生成 TreeConv1d 所需的索引张量。

    输出形状为 [N, 1] 的列向量，按前序遍历顺序存储每个节点的 (自身, 左子, 右子) ID，
    展平后供 torch.gather 使用。

    Example:
        Join(A, B) → indexes = [[1], [2], [3], [2], [0], [0], [3], [0], [0]]
        表示：
          节点1（根）的子节点是2和3；
          节点2（叶）无子节点（0,0）；
          节点3（叶）无子节点（0,0）。
    """
    preorder_ids, _ = _make_preorder_ids_tree(root)
    vecs = []
    _walk(preorder_ids, vecs)
    vecs = np.asarray(vecs).reshape(-1, 1)  # 展平为列向量
    return vecs


# @profile
def _featurize_tree(curr_node, node_featurizer):
    """为单棵树的每个节点生成特征向量（自底向上）。

    使用缓存避免重复计算，最终返回形状为 [num_nodes + 1, feature_dim] 的矩阵，
    其中索引0为零向量（对应无效节点），索引1~N对应实际节点（按前序遍历顺序）。

    Args:
        curr_node: 树根节点
        node_featurizer: 特征化器，需实现 FeaturizeLeaf 和 Merge 方法

    Returns:
        np.ndarray: [num_nodes + 1, feature_dim]
    """
    def _bottom_up(curr):
        """后序遍历，为每个节点计算特征（缓存结果）。"""
        if hasattr(curr, '__node_feature_vec'):
            return curr.__node_feature_vec
        if not curr.children:
            # 叶子节点：直接特征化
            vec = node_featurizer.FeaturizeLeaf(curr)
        else:
            # 内部节点：合并左右子节点特征
            left_vec = _bottom_up(curr.children[0])
            right_vec = _bottom_up(curr.children[1])
            vec = node_featurizer.Merge(curr, left_vec, right_vec)
        curr.__node_feature_vec = vec
        return vec

    _bottom_up(curr_node)

    # 按前序遍历顺序收集所有节点特征
    vecs = []
    plans_lib.MapNode(curr_node, lambda node: vecs.append(node.__node_feature_vec))

    # 创建结果矩阵（索引0为零向量）
    ret = np.zeros((len(vecs) + 1, vecs[0].shape[0]), dtype=np.float32)
    ret[1:] = vecs
    return ret


# @profile
def make_and_featurize_trees(trees, node_featurizer):
    """批量处理多棵树：生成特征张量和索引张量。

    Args:
        trees: List[TreeNode]，树节点列表
        node_featurizer: 节点特征化器

    Returns:
        tuple:
            - trees_tensor: [B, feature_dim, max_nodes]，节点特征（转置后）
            - indexes_tensor: [B, max_nodes, 3]，节点索引（long 类型）
    """
    # 1. 为每棵树生成索引（形状 [N_i, 1]），然后批量化
    indexes = torch.from_numpy(_batch([_make_indexes(x) for x in trees])).long()
    # reshape: [B, max_N*3, 1] → [B, max_N, 3]
    indexes = indexes.reshape(indexes.shape[0], -1, 3)

    # 2. 为每棵树生成节点特征（形状 [N_i+1, F]），批量化并转置为 [B, F, N]
    trees = torch.from_numpy(
        _batch([_featurize_tree(x, node_featurizer) for x in trees])
    ).transpose(1, 2)  # [B, N, F] → [B, F, N]

    return trees, indexes
