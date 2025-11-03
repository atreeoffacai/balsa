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
"""Workload definitions."""
import glob
import os

import numpy as np

import balsa
from balsa import hyperparams
from balsa.util import plans_lib
from balsa.util import postgres

_EPSILON = 1e-6


def ParseSqlToNode(path):
    """
    将指定路径的 SQL 文件解析为一个表示查询计划的 Node 对象（balsa.Node），
    并附加原始 SQL、文件路径、查询名称及 PostgreSQL 的 EXPLAIN JSON 等元信息。

    Args:
        path (str): SQL 文件的完整路径（例如：'queries/job/q1.sql'）

    Returns:
        balsa.Node: 表示该 SQL 查询逻辑/物理计划的节点对象，包含丰富的元数据。
    """
    # 1. 从完整路径中提取文件名（不含目录），例如 'q1.sql'
    base = os.path.basename(path)
    
    # 2. 去掉文件扩展名，得到查询名称（例如 'q1'），用于标识该查询
    query_name = os.path.splitext(base)[0]
    
    # 3. 读取 SQL 文件内容为字符串
    with open(path, 'r') as f:
        sql_string = f.read()
    
    # 4. 调用 PostgreSQL 后端接口，将 SQL 字符串转换为计划树节点（Node）
    #    同时返回对应的 EXPLAIN (FORMAT JSON) 结果（字典形式）
    #    注意：SqlToPlanNode 可能会执行 EXPLAIN 并解析其输出
    node, json_dict = postgres.SqlToPlanNode(sql_string)
    
    # 5. 将原始文件路径存入 node 的 info 字典，便于后续追踪来源
    node.info['path'] = path
    
    # 6. 保存原始 SQL 字符串，方便调试或重放
    node.info['sql_str'] = sql_string
    
    # 7. 保存查询名称（如 'q1'），用于日志、可视化或结果命名
    node.info['query_name'] = query_name
    
    # 8. 保存 PostgreSQL 返回的 EXPLAIN JSON 原始数据，
    #    可用于分析计划结构、代价估算、实际执行时间等
    node.info['explain_json'] = json_dict
    
    # 9. 确保 Node 内部已解析或缓存 SQL 相关信息（如表名、谓词等），
    #    GetOrParseSql() 通常会触发一次 SQL 解析（如果尚未完成）
    node.GetOrParseSql()
    
    # 10. 返回填充了元数据的查询计划节点
    return node


class Workload(object):

    @classmethod
    def Params(cls):
        p = hyperparams.InstantiableParams(cls)
        p.Define('query_dir', None, 'Directory to workload queries.')
        p.Define(
            'query_glob', '*.sql',
            'If supplied, glob for this pattern.  Otherwise, use all queries.'\
            '  Example: 29*.sql.'
        )
        p.Define(
            'loop_through_queries', False,
            'Loop through a random permutation of queries? '
            'Desirable for evaluation.')
        p.Define(
            'test_query_glob', None,
            'Similar usage as query_glob. If None, treating all queries'\
            ' as training nodes.'
        )
        p.Define('search_space_join_ops',
                 ['Hash Join', 'Merge Join', 'Nested Loop'],
                 'Join operators to learn.')
        p.Define('search_space_scan_ops',
                 ['Index Scan', 'Index Only Scan', 'Seq Scan'],
                 'Scan operators to learn.')
        return p

    def __init__(self, params):
        self.params = params.Copy()
        p = self.params
        # Subclasses should populate these fields.
        self.query_nodes = None
        self.workload_info = None
        self.train_nodes = None
        self.test_nodes = None

        if p.loop_through_queries:
            self.queries_permuted = False
            self.queries_ptr = 0

    def _ensure_queries_permuted(self, rng):
        """Permutes queries once."""
        if not self.queries_permuted:
            self.query_nodes = rng.permutation(self.query_nodes)
            self.queries_permuted = True

    def _get_sql_set(self, query_dir, query_glob):
        if query_glob is None:
            return set()
        else:
            globs = query_glob
            if type(query_glob) is str:
                globs = [query_glob]
            sql_files = np.concatenate([
                glob.glob('{}/{}'.format(query_dir, pattern))
                for pattern in globs
            ]).ravel()
        sql_files = set(sql_files)
        return sql_files

    def Queries(self, split='all'):
        """Returns all queries as balsa.Node objects."""
        assert split in ['all', 'train', 'test'], split
        if split == 'all':
            return self.query_nodes
        elif split == 'train':
            return self.train_nodes
        elif split == 'test':
            return self.test_nodes

    def WithQueries(self, query_nodes):
        """Replaces this Workload's queries with 'query_nodes'."""
        self.query_nodes = query_nodes
        self.workload_info = plans_lib.WorkloadInfo(query_nodes)

    def FilterQueries(self, query_dir, query_glob, test_query_glob):
        all_sql_set_new = self._get_sql_set(query_dir, query_glob)
        test_sql_set_new = self._get_sql_set(query_dir, test_query_glob)
        assert test_sql_set_new.issubset(all_sql_set_new), (test_sql_set_new,
                                                            all_sql_set_new)

        all_sql_set = set([n.info['path'] for n in self.query_nodes])
        assert all_sql_set_new.issubset(all_sql_set), (
            'Missing nodes in init_experience; '
            'To fix: remove data/initial_policy_data.pkl, or see README.')

        query_nodes_new = [
            n for n in self.query_nodes if n.info['path'] in all_sql_set_new
        ]
        train_nodes_new = [
            n for n in query_nodes_new
            if test_query_glob is None or n.info['path'] not in test_sql_set_new
        ]
        test_nodes_new = [
            n for n in query_nodes_new if n.info['path'] in test_sql_set_new
        ]
        assert len(train_nodes_new) > 0

        self.query_nodes = query_nodes_new
        self.train_nodes = train_nodes_new
        self.test_nodes = test_nodes_new

    def UseDialectSql(self, p):
        dialect_sql_dir = p.engine_dialect_query_dir
        for node in self.query_nodes:
            assert 'sql_str' in node.info and 'query_name' in node.info
            path = os.path.join(dialect_sql_dir,
                                node.info['query_name'] + '.sql')
            assert os.path.isfile(path), '{} does not exist'.format(path)
            with open(path, 'r') as f:
                dialect_sql_string = f.read()
            node.info['sql_str'] = dialect_sql_string


class JoinOrderBenchmark(Workload):
    """
    表示 Join Order Benchmark (JOB) 工作负载的类。
    该类负责加载 JOB 查询集，并将其划分为训练集和测试集。
    """

    @classmethod
    def Params(cls):
        """
        类方法：定义并返回该工作负载所需的参数配置。
        这些参数将用于控制查询路径、搜索空间等行为。
        """
        # 调用父类（Workload）的 Params 方法，获取基础参数
        p = super().Params()
        
        # 获取 balsa 模块所在目录的绝对路径，并向上回退一级（即项目根目录）
        # 使用绝对路径是为了兼容 RLlib（Ray 的强化学习库），它要求路径是绝对的
        module_dir = os.path.abspath(os.path.dirname(balsa.__file__) + '/../')
        
        # 设置 JOB 查询文件所在的目录（相对于项目根目录）
        p.query_dir = os.path.join(module_dir, 'queries/join-order-benchmark')
        
        return p

    def __init__(self, params):
        """
        初始化 JoinOrderBenchmark 实例。
        
        Args:
            params: 包含配置参数的对象（通常由 Params() 生成）
        """
        # 调用父类的初始化方法
        super().__init__(params)
        p = params
        
        # 加载所有查询，并划分为：全部查询、训练查询、测试查询
        self.query_nodes, self.train_nodes, self.test_nodes = self._LoadQueries()
        
        # 创建工作负载信息对象，用于后续计划生成和优化
        self.workload_info = plans_lib.WorkloadInfo(self.query_nodes)
        
        # 设置该工作负载允许使用的物理操作符（如 join 类型、scan 类型）
        # 这些操作符定义了搜索空间的边界
        self.workload_info.SetPhysicalOps(
            p.search_space_join_ops,   # 允许的 join 操作符列表（如 HashJoin, NestedLoop 等）
            p.search_space_scan_ops    # 允许的 scan 操作符列表（如 SeqScan, IndexScan 等）
        )

    def _LoadQueries(self):
        """
        从文件系统加载所有 SQL 查询文件，并转换为 balsa.Node 对象。
        同时根据配置将查询划分为训练集和测试集。
        
        Returns:
            tuple: (all_nodes, train_nodes, test_nodes)
                - all_nodes: 所有查询对应的 Node 列表
                - train_nodes: 用于训练的 Node 列表
                - test_nodes: 用于测试的 Node 列表
        """
        p = self.params
        
        # 1. 加载所有匹配 p.query_glob 模式的 SQL 文件路径（集合形式，去重）
        all_sql_set = self._get_sql_set(p.query_dir, p.query_glob)
        
        # 2. 加载测试集查询（匹配 p.test_query_glob 模式的 SQL 文件路径）
        test_sql_set = self._get_sql_set(p.query_dir, p.test_query_glob)
        
        # 3. 确保测试集是全集的子集（逻辑校验）
        assert test_sql_set.issubset(all_sql_set), "测试查询必须是全部查询的子集"
        
        # 4. 将所有 SQL 文件路径按名称排序（便于调试时顺序一致）
        all_sql_list = sorted(all_sql_set)
        
        # 5. 将每个 SQL 文件解析为 balsa.Node 对象（内部包含查询计划树等信息）
        all_nodes = [ParseSqlToNode(sqlfile) for sqlfile in all_sql_list]
        
        # 6. 划分训练集：
        #    如果未指定测试查询模式（p.test_query_glob 为 None），则全部作为训练集；
        #    否则，排除掉属于测试集的查询。
        train_nodes = [
            n for n in all_nodes
            if p.test_query_glob is None or n.info['path'] not in test_sql_set
        ]
        
        # 7. 划分测试集：仅包含在 test_sql_set 中的查询
        test_nodes = [n for n in all_nodes if n.info['path'] in test_sql_set]
        
        # 8. 确保训练集非空（防止配置错误导致无法训练）
        assert len(train_nodes) > 0, "训练集不能为空"
        
        return all_nodes, train_nodes, test_nodes


class RunningStats(object):
    """Computes running mean and standard deviation.

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs.Record(np.random.randn())
        print(rs.Mean(), rs.Std())
    """

    def __init__(self, n=0., m=None, s=None):
        self.n = n
        self.m = m
        self.s = s

    def Record(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def Mean(self):
        return self.m if self.n else 0.0

    def Variance(self):
        return self.s / (self.n) if self.n else 0.0

    def Std(self, epsilon_guard=True):
        eps = 1e-6
        std = np.sqrt(self.Variance())
        if epsilon_guard:
            return np.maximum(eps, std)
        return std
