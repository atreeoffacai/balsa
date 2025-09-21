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

import collections
import time
import multiprocessing

import numpy as np
import torch
import random

from balsa import search
from balsa.models import treeconv
from balsa.util import dataset as ds
from balsa.util import plans_lib



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class PlannerConfig(
        collections.namedtuple(
            'PlannerConfig',
            [
                'search_space', 'enable_nestloop', 'enable_hashjoin',
                'enable_mergejoin'
            ],
        )):
    """Experimental: a simple tuple recording what ops can be planned."""

    @classmethod
    def Get(cls, name):
        return {
            'NestLoopHashJoin': PlannerConfig.NestLoopHashJoin(),
            'LeftDeepNestLoop': PlannerConfig.LeftDeepNestLoop(),
            'LeftDeepNestLoopHashJoin':
                PlannerConfig.LeftDeepNestLoopHashJoin(),
            'LeftDeep': PlannerConfig.LeftDeep(),
            'Dbmsx': PlannerConfig.Dbmsx(),
        }[name]

    @classmethod
    def Default(cls):
        return cls(search_space='bushy',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=True)

    @classmethod
    def NestLoopHashJoin(cls):
        return cls(search_space='bushy',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=False)

    @classmethod
    def LeftDeepNestLoop(cls):
        return cls(search_space='leftdeep',
                   enable_nestloop=True,
                   enable_hashjoin=False,
                   enable_mergejoin=False)

    @classmethod
    def LeftDeepNestLoopHashJoin(cls):
        return cls(search_space='leftdeep',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=False)

    @classmethod
    def LeftDeep(cls):
        return cls(search_space='leftdeep',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=True)

    @classmethod
    def Dbmsx(cls):
        return cls(search_space='dbmsx',
                   enable_nestloop=True,
                   enable_hashjoin=True,
                   enable_mergejoin=True)

    def KeepEnabledJoinOps(self, join_ops):
        ops = []
        for op in join_ops:
            if op == 'Nested Loop' and self.enable_nestloop:
                ops.append(op)
            elif op == 'Hash Join' and self.enable_hashjoin:
                ops.append(op)
            elif op == 'Merge Join' and self.enable_mergejoin:
                ops.append(op)
        assert len(ops) > 0, (self, join_ops)
        return ops

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def __lt__(self, other):
        if self.visits != 0 and other.visits != 0:
            return self.value / self.visits < other.value / other.visits
        elif self.visits == 0 and other.visits != 0:
            return False
        elif self.visits != 0 and other.visits == 0:
            return True
        else:
            return True

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return len(self.state) == 1
    
    def is_oldLeaf(self):
        return (len(self.children) == 0 and self.visits != 0)

class Optimizer(object):
    """Creates query execution plans using learned model."""

    def __init__(
        self,
        workload_info,
        plan_featurizer,
        parent_pos_featurizer,
        query_featurizer,
        inverse_label_transform_fn,
        model,
        tree_conv=False,
        beam_size=10,
        search_until_n_complete_plans=1,
        plan_physical=False,
        use_label_cache=True,
        use_plan_restrictions=True,
    ):
        self.workload_info = workload_info
        self.plan_featurizer = plan_featurizer
        self.parent_pos_featurizer = parent_pos_featurizer
        self.query_featurizer = query_featurizer
        self.inverse_label_transform_fn = inverse_label_transform_fn
        self.use_label_cache = use_label_cache
        self.use_plan_restrictions = use_plan_restrictions

        # Plan search params
        if not plan_physical:
            jts = workload_info.join_types
            assert np.array_equal(jts, ['Join']), jts
            sts = workload_info.scan_types
            assert np.array_equal(sts, ['Scan']), sts
        self.plan_physical = plan_physical
        self.beam_size = beam_size
        self.search_until_n_complete_plans = search_until_n_complete_plans
        self.tree_conv = tree_conv
        self.SetModel(model)

        # Debugging.
        self.total_joins = 0
        self.total_random_triggers = 0
        self.num_queries_with_random = 0

    def SetModel(self, model):
        self.value_network = model.to(DEVICE)
        # Set the model in eval mode.  Only affects modules (e.g., Dropout)
        # that have different behaviors in train vs. in test.  Do it once here
        # so that Optimizer.plan() below doesn't have to do it on every query.
        self.value_network.eval()
        # Reset the cache due to model being changed.
        self.label_cache = {}

    # @profile
    # 利用深度学习模型预测数据库查询计划的成本，以辅助查询优化器做出更高效的选择。
    def infer(self, query_node, plan_nodes, set_model_eval=False):
        """Forward pass.

        Args:
            query_node: a plans_lib.Node object. Represents the query context.
            plan_nodes: a list of plans_lib.Node objects. The set of plans to
              score.

        Returns:
            costs, a float. Higher costs indicate more expensive plans.
        """
        labels = [None] * len(plan_nodes)
        plans, idx = [], []
        if self.use_label_cache:
            # Gather cached labels
            lookup_keys = [(query_node.info['query_name'],
                            plan.to_str(with_cost=False))
                           for plan in plan_nodes]
            for i, lookup_key in enumerate(lookup_keys):
                label = self.label_cache.get(lookup_key)
                if label is not None:
                    labels[i] = label
                else:
                    plans.append(plan_nodes[i])
                    idx.append(i)
            # No plans to score.
            if len(plans) == 0:
                return labels
        else:
            plans = plan_nodes

        # Perform inference on new plans.
        if set_model_eval:
            # Expensive.  Caller should try to call only once.
            self.value_network.eval()
        with torch.no_grad():
            query_enc = self.query_featurizer(query_node)
            all_query_vecs = [query_enc] * len(plans)
            all_plans = []
            all_indexes = []
            if self.tree_conv:
                all_plans, all_indexes = treeconv.make_and_featurize_trees(
                    plans, self.plan_featurizer)
            else:
                for plan_node in plans:
                    all_plans.append(self.plan_featurizer(plan_node))

                if self.parent_pos_featurizer is not None:
                    for plan_node in plans:
                        all_indexes.append(
                            self.parent_pos_featurizer(plan_node))

            if self.tree_conv or hasattr(self.plan_featurizer, 'pad'):
                query_feat = torch.from_numpy(np.asarray(all_query_vecs)).to(
                    DEVICE, non_blocking=True)
                plan_feat = torch.from_numpy(np.asarray(all_plans)).to(
                    DEVICE, non_blocking=True)
                pos_feat = torch.from_numpy(np.asarray(all_indexes)).to(
                    DEVICE, non_blocking=True)
                cost = self.value_network(query_feat, plan_feat,
                                          pos_feat).cpu().numpy()
            else:
                all_costs = [1] * len(all_plans)
                batch = ds.PlansDataset(
                    all_query_vecs,
                    all_plans,
                    all_indexes,
                    all_costs,
                    transform_cost=False,
                    return_indexes=False,
                )
                loader = torch.utils.data.DataLoader(batch,
                                                     batch_size=len(all_plans),
                                                     shuffle=False)
                processed_batch = list(loader)[0]
                query_feat, plan_feat = processed_batch[0].to(
                    DEVICE), processed_batch[1].to(DEVICE)
                cost = self.value_network(query_feat, plan_feat).cpu().numpy()

            cost = self.inverse_label_transform_fn(cost)
            plan_labels = cost.reshape(-1,).tolist()

            if self.use_label_cache:
                # Update the cache with the labels.
                for i in range(len(plan_labels)):
                    labels[idx[i]] = plan_labels[i]
                    self.label_cache[lookup_keys[idx[i]]] = plan_labels[i]
            else:
                labels = plan_labels
            return labels

    def plan(self, query_node, search_method, **kwargs):
        """Generate a query execution plan using the specified search method."""
        if search_method == 'mcts':
            return self._mcts_search(query_node, **kwargs)
        elif search_method == 'beam_bk':
            return self._beam_search_bk(query_node, **kwargs)
        raise ValueError(f'Unsupported search_method: {search_method}')

    def _oneSimulation(self,tup):
        root, exploration_weight, query_node, join_graph, bushy, planner_config, avoid_eq_filters, joinHash2CostMap = tup
        sim_time_start = time.time()
        # print('num_simulations: {:.1f}', ns)
        # selection,结果要么是终止态，要么是未simulation过的当前叶子节点
        node = self._select(root, exploration_weight)
        assert isinstance(node, MCTSNode), f"node should be MCTSNode, got {type(node)}"
        # node expansion
        if (not node.is_terminal()) and node.is_oldLeaf():
            self._expand(node, query_node, join_graph, bushy, planner_config, avoid_eq_filters)
            node = node.children[0]
        # 正常 simulation / rollout
        # value = self._simulate(node, query_node, join_graph, bushy, planner_config, avoid_eq_filters)

        #直接使用 model 进行 infer
        # values = []
        # for i in range(len(node.state)):
        #     cost = self.infer(query_node, [node.state[i]])[0]
        #     values.append(cost)
        # value = max(cost)
        
        # simulation 替换成 beam_search 试试,不可以，因为每次调用beamsearch结果都一样，而不是从当前状态出发
        # tup_bs = self._beam_search_bk(query_node,beam_size=beam_size,bushy=bushy,return_all_found=return_all_found,
        #                               planner_config=planner_config, verbose=verbose, avoid_eq_filters=avoid_eq_filters,epsilon_greedy=epsilon_greedy)
        # planning_time, found_plan, predicted_latency, found_plans = tup_bs
        # value = predicted_latency
        # print('xiaoma: bs in mcts predicted time: {:.1f}', predicted_latency)

        # simulation 替换成 beam_search ，从当前状态出发版
        # now node must be a new leaf, may be terminal
        if not node.is_terminal():
            # h = self._mctsStateHash(node.state)
            # ret = self._mctsGetFromExplored(h, exploredState=exploredState)
            # if ret is None:
            tup_bs = self._simulation_beam_search_bk_from_a_state(joinHash2CostMap,node.state,query_node,beam_size=beam_size_int,bushy=bushy,return_all_found=return_all_found,
                                    planner_config=planner_config, verbose=False, avoid_eq_filters=avoid_eq_filters,epsilon_greedy=epsilon_greedy)    
            # else:
            #     simulation_number -= 1
            #     tup_bs = exploredState[h]
            planning_time, found_plan, predicted_latency, found_plans = tup_bs
            # 不能直接cotinue，因为得propagation
            # if planning_time == 0 and found_plan == 0 and predicted_latency == 0 and found_plans == 0:
            #     mcts_planning_time = (time.time() - planning_start_t) * 1e3
            #     return [firBSPlanninTime, 
            #             mcts_planning_time, 
            #             bestNode.state[0], # plan
            #             bestNode.value, 
            #             firstBSBestResult, 
            #             simulation_number]
                
            value = predicted_latency
            # 记录第一次最好的结果，作为beamsize结果
            if firstBSBestResult == 0.0:
                firstBSBestResult = predicted_latency
                firBSPlanninTime = planning_time
                beam_size_int = 1
            # else:
            #     if (firstBSBestResult - predicted_latency) / firstBSBestResult > 1:
            #         jianhaojiushou = True
            #     # if simulation_number >= 30:
            #     #     jianhaojiushou = True
        else:
            # terminal
            # h = self._mctsStateHash(node.state)
            # v = joinHash2CostMap.get(h)
            # if value is None:
            found_plan = node.state[0]
            value = self.infer(query_node=query_node,plan_nodes=[found_plan])[0]
            # else:
                # found_plan = node.state[0]
                # value = v

        # print('xiaoma: bs in mcts predicted time: {:.1f}', predicted_latency)
            
        def updateBestNode(oriNode, newNode):
            if oriNode.value > newNode.value:
                return newNode
            else:
                return oriNode
            
        if bestNode.state == None:
            bestNode.state = [found_plan]
            bestNode.value = value
        else:
            nowNode = MCTSNode(state=[found_plan])
            nowNode.value = value
            bestNode = updateBestNode(bestNode, nowNode)

        # back propagation
        self._backpropagate(node, value)
        sim_time = (time.time() - sim_time_start) * 1e3
        print("time: ", sim_time)
        return (root, predicted_latency, planning_time)

    # query_node, beam_size=10, bushy=False, return_all_found=False, planner_config=None,
    # verbose=False, avoid_eq_filters=False, epsilon_greedy=0

    # num_simulations=20 渐变初始值
    def _mcts_search(self, query_node, beam_size=10, bushy=False, return_all_found=False, planner_config=None,
                      verbose=False, avoid_eq_filters=False, epsilon_greedy=0, num_simulations=20, exploration_weight=1.414, **kwargs):
        """Produce a plan via MCTS.
        Args:
            query_node: a Node, a parsed version of the query to optimize.
            num_simulations: number of MCTS simulations to run.
            exploration_weight: the exploration weight for UCB1.
        """
        def getAlpha(x0, xn, n):
            return 1-((xn-1)/(x0-1))**(1.0/n)

        firstBSBestResult = 0.0
        firBSPlanninTime = 0.0
        planning_start_t = time.time()
        join_graph, _ = query_node.GetOrParseSql()
        root_state = query_node.GetLeaves()
        root = MCTSNode(root_state)
        # print('一开始的状态：')
        # for ele_in_state in root_state:
        #         print(ele_in_state)
        bestNode = MCTSNode(None)
        needStep = len(root.state) - 1
        # print('nead steps: ', needStep)
        alpha = getAlpha(num_simulations, 1.1, needStep)
        step_count = 0 
        simulation_number = 0
        planningTimeout = False
        joinHash2CostMap = {}
        while (not root.is_terminal()) and (not planningTimeout):
            # print('当前不是最终状态, 状态为: ')
            # for ele_in_state in root.state:
            #     print(ele_in_state)
            step_count += 1
            if step_count != 1:
                num_simulations = (1-alpha) * num_simulations + alpha
            simulation_int = int(num_simulations)
            if simulation_int == 1:
                simulation_int += 1
            beam_size_int = int(beam_size)
            # print('第',step_count ,'步的num_simulations: ', int(num_simulations))
            for ns in range(simulation_int):
                simulation_number += 1
                print("simulation_number: ", simulation_number)
                sim_time_start = time.time()
                # print('num_simulations: {:.1f}', ns)
                # selection,结果要么是终止态，要么是未simulation过的当前叶子节点
                node = self._select(root, exploration_weight)
                assert isinstance(node, MCTSNode), f"node should be MCTSNode, got {type(node)}"
                # node expansion
                if (not node.is_terminal()) and node.is_oldLeaf():
                    self._expand(node, query_node, join_graph, bushy, planner_config, avoid_eq_filters)
                    node = node.children[0]
                # 正常 simulation / rollout
                # value = self._simulate(node, query_node, join_graph, bushy, planner_config, avoid_eq_filters)

                #直接使用 model 进行 infer
                # values = []
                # for i in range(len(node.state)):
                #     cost = self.infer(query_node, [node.state[i]])[0]
                #     values.append(cost)
                # value = max(cost)
                
                # simulation 替换成 beam_search 试试,不可以，因为每次调用beamsearch结果都一样，而不是从当前状态出发
                # tup_bs = self._beam_search_bk(query_node,beam_size=beam_size,bushy=bushy,return_all_found=return_all_found,
                #                               planner_config=planner_config, verbose=verbose, avoid_eq_filters=avoid_eq_filters,epsilon_greedy=epsilon_greedy)
                # planning_time, found_plan, predicted_latency, found_plans = tup_bs
                # value = predicted_latency
                # print('xiaoma: bs in mcts predicted time: {:.1f}', predicted_latency)

                # simulation 替换成 beam_search ，从当前状态出发版
                # now node must be a new leaf, may be terminal
                if not node.is_terminal():
                    # h = self._mctsStateHash(node.state)
                    # ret = self._mctsGetFromExplored(h, exploredState=exploredState)
                    # if ret is None:
                    tup_bs = self._simulation_beam_search_bk_from_a_state(joinHash2CostMap,node.state,query_node,beam_size=beam_size_int,bushy=bushy,return_all_found=return_all_found,
                                            planner_config=planner_config, verbose=False, avoid_eq_filters=avoid_eq_filters,epsilon_greedy=epsilon_greedy)    
                    # else:
                    #     simulation_number -= 1
                    #     tup_bs = exploredState[h]
                    planning_time, found_plan, predicted_latency, found_plans = tup_bs
                    # 不能直接cotinue，因为得propagation
                    # if planning_time == 0 and found_plan == 0 and predicted_latency == 0 and found_plans == 0:
                    #     mcts_planning_time = (time.time() - planning_start_t) * 1e3
                    #     return [firBSPlanninTime, 
                    #             mcts_planning_time, 
                    #             bestNode.state[0], # plan
                    #             bestNode.value, 
                    #             firstBSBestResult, 
                    #             simulation_number]
                        
                    value = predicted_latency
                    # 记录第一次最好的结果，作为beamsize结果
                    if firstBSBestResult == 0.0:
                        firstBSBestResult = predicted_latency
                        firBSPlanninTime = planning_time
                        beam_size_int = 1
                    # else:
                    #     if (firstBSBestResult - predicted_latency) / firstBSBestResult > 1:
                    #         jianhaojiushou = True
                    #     # if simulation_number >= 30:
                    #     #     jianhaojiushou = True
                else:
                    # terminal
                    # h = self._mctsStateHash(node.state)
                    # v = joinHash2CostMap.get(h)
                    # if value is None:
                    found_plan = node.state[0]
                    value = self.infer(query_node=query_node,plan_nodes=[found_plan])[0]
                    # else:
                        # found_plan = node.state[0]
                        # value = v

                # print('xiaoma: bs in mcts predicted time: {:.1f}', predicted_latency)
                    
                def updateBestNode(oriNode, newNode):
                    if oriNode.value > newNode.value:
                        return newNode
                    else:
                        return oriNode
                    
                if bestNode.state == None:
                    bestNode.state = [found_plan]
                    bestNode.value = value
                else:
                    nowNode = MCTSNode(state=[found_plan])
                    nowNode.value = value
                    bestNode = updateBestNode(bestNode, nowNode)

                # back propagation
                self._backpropagate(node, value)
                planning_time = (time.time() - planning_start_t) * 1e3
                if planning_time > 10 * firBSPlanninTime:
                    planningTimeout = True
                sim_time = (time.time() - sim_time_start) * 1e3
                print("time: ", sim_time)
            # best_child = max(root.children, key=lambda c: c.visits)
            best_child = min(root.children)
            best_state = best_child.state
            # for ele_in_state in best_state:
            #     print('xiaoma: state element',ele_in_state)
            root = MCTSNode(best_state)
        mcts_planning_time = (time.time() - planning_start_t) * 1e3
        # print('xiaoma: mcts Planning took {:.1f}ms'.format(planning_time),' step_count :', step_count)
        # return [planning_time, best_state[0], best_child.value / best_child.visits]
        # lentency = self.infer(query_node, [root.state[0]])[0]
        return [firBSPlanninTime, 
                mcts_planning_time, 
                bestNode.state[0], # plan
                bestNode.value, 
                firstBSBestResult, 
                simulation_number]
    
    # selection
    def _select(self, node, exploration_weight):
        """Select the most promising node using UCB1."""
        while not node.is_leaf() and not node.is_terminal():
            node = self._best_child(node, exploration_weight)
        return node

    def _best_child(self, node, exploration_weight):
        """Choose the best child node based on UCB1 score."""
        
        def getMinMaxValueFormNodeChildren(node):
            minAveValueNoInf = float('inf')
            maxAveValueNoInf = -float('inf')
            for child in node.children:
                if child.visits == 0:
                    return (minAveValueNoInf, maxAveValueNoInf, child)
                else:
                    value = child.value / child.visits
                    minAveValueNoInf = min(value, minAveValueNoInf)
                    maxAveValueNoInf = max(value, maxAveValueNoInf)
            return (minAveValueNoInf, maxAveValueNoInf, None)
                

        def minMaxNormalize(value, maxValue, minValue):
            return (value - minValue) / (maxValue - minValue)
        
        minAveValueNoInf,  maxAveValueNoInf, oneChild = getMinMaxValueFormNodeChildren(node)
        if oneChild != None: # getMinMaxValueFormNodeChildren 过程中发现了 无穷小节点，直接选中返回
            return oneChild

        best_score = float('inf')
        best_child = None
        for child in node.children:
            if child.visits == 0: # -inf end adcancely
                score = -float('inf')
                best_child = child
                return best_child
            else:
                exploit = minMaxNormalize(child.value / child.visits, minAveValueNoInf, maxAveValueNoInf)
                explore = -exploration_weight * np.sqrt(np.log(node.visits) / child.visits)
                score = exploit + explore
            if score < best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node, query_node, join_graph, bushy, planner_config, avoid_eq_filters):
        """Expand the node by generating all possible child states."""
        possible_plans = self._get_possible_plans(query_node, 
                                                  node.state, 
                                                  join_graph, 
                                                  bushy=bushy,
                                                  planner_config=planner_config,
                                                  avoid_eq_filters=avoid_eq_filters)
        # costs = self.infer(query_node, [join for join, _, _ in possible_plans])
        # valid_costs, valid_new_states = self._make_new_states(node.state, costs, possible_plans)

        for plan, left_idx, right_idx in possible_plans:
            planCost = self.infer(query_node=query_node,plan_nodes=[plan])[0]
            plan.cost = planCost
            new_state = node.state[:]
            new_state[left_idx] = plan
            del new_state[right_idx]
            child_node = MCTSNode(new_state, parent=node, action=(left_idx, right_idx))
            node.children.append(child_node)
    
    def _simulate(self, node, query_node, join_graph, bushy, planner_config, avoid_eq_filters):
        """Simulate a random plan from the current state and return its cost."""
        state = node.state[:]
        while len(state) > 1:
            possible_plans = self._get_possible_plans(query_node, state, join_graph, bushy=bushy,planner_config=planner_config,avoid_eq_filters=avoid_eq_filters)
            # costs = self.infer(query_node, [join for join, _, _ in possible_plans])
            rand_idx = random.randint(0, len(possible_plans) - 1)
            plan, left_idx, right_idx = possible_plans[rand_idx]
            state[left_idx] = plan
            del state[right_idx]
        # new_state_cost = -1e30
        # for rel in state:
        #     if rel.IsJoin():
        #         # Goodness(state) = max V_theta(subplan), for all subplan
        #         # in state.
        #         new_state_cost = max(new_state_cost, rel.cost)

        # 这里是没有运用模型前景的能力，直接simulation到了终止态，然后得到其代价
        cost = self.infer(query_node, [state[0]])[0]
        return cost

    def _backpropagate(self, node, value):
        """Update the node statistics up to the root."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _get_possible_plans(self,
                            query_node,
                            state,
                            join_graph,
                            bushy=False,
                            planner_config=None,
                            avoid_eq_filters=False):
        """Expands a state.  Returns a list of successor states."""
        if not bushy:
            if planner_config.search_space == 'leftdeep':
                func = self._get_possible_plans_left_deep
            else:
                func = self._get_possible_plans_dbmsx
        else:
            func = self._get_possible_plans_bushy
        return func(query_node,
                    state,
                    join_graph,
                    planner_config=planner_config,
                    avoid_eq_filters=avoid_eq_filters)

    # @profile
    def _get_possible_plans_bushy(self,
                                  query_node,
                                  state,
                                  join_graph,
                                  planner_config=None,
                                  avoid_eq_filters=False):
        possible_joins = []
        num_rels = len(state)
        for i in range(num_rels):
            for j in range(num_rels):
                if i == j:
                    continue
                l = state[i]
                r = state[j]
                # Hinting a join between non-neighbors may fail (PG may
                # disregard the hint).
                if not plans_lib.ExistsJoinEdgeInGraph(l, r, join_graph):
                    continue
                for plan in self._enumerate_plan_operators(
                        l,
                        r,
                        planner_config=planner_config,
                        avoid_eq_filters=avoid_eq_filters):
                    possible_joins.append((plan, i, j))
        return possible_joins

    def _get_possible_plans_dbmsx(self,
                                  query_node,
                                  state,
                                  join_graph,
                                  planner_config=None,
                                  avoid_eq_filters=False):
        raise NotImplementedError

    # 最终返回的是一个列表，每一项是 (join_plan, left_index, right_index)，它们表示的是当前 要进行 Join 操作的两个子计划（或关系）在 state 列表中的索引位置。
    # left_index: 左边操作数（左子计划）在 state 中的索引；
    # right_index: 右边操作数（右子计划）在 state 中的索引；
    # 而 state 是一个表示当前查询状态的列表，其中每个元素通常是：
    # 单个表（Scan）
    # 或者一个已经生成的 Join 子计划
    # 在当前状态的基础上生成所有可能的下一步 Join 动作，而不是一次性生成完整的查询计划
    def _get_possible_plans_left_deep(self,
                                      query_node,
                                      state,
                                      join_graph,
                                      planner_config=None,
                                      avoid_eq_filters=False):
        possible_joins = []
        num_rels = len(state)
        # 这部分代码在检查当前状态中是否存在已经执行的 Join 操作
        join_index = None
        for i, s in enumerate(state):
            if s.IsJoin():
                assert join_index is None, 'two joins found'
                join_index = i
        # 如果当前还没有 Join，则枚举所有两两关系间的合法 Join；
        if join_index is None:
            # Base state: all unspecified scans.
            scored = set()
            for i in range(num_rels):
                for j in range(num_rels):
                    if i == j:
                        continue
                    if (i, j) in scored:
                        continue
                    scored.add((i, j))
                    l = state[i]
                    r = state[j]
                    # Hinting a join between non-neighbors may fail (PG may
                    # disregard the hint).
                    if not plans_lib.ExistsJoinEdgeInGraph(l, r, join_graph):
                        continue
                    for plan in self._enumerate_plan_operators(
                            l,
                            r,
                            planner_config=planner_config,
                            avoid_eq_filters=avoid_eq_filters):
                        possible_joins.append((plan, i, j))
        # 如果已经有 Join，则只允许在这个 Join 基础上添加新的关系（形成左深树）；
        else:
            i, l = join_index, state[join_index]
            for j in range(0, len(state)):
                if j == i:
                    continue
                r = state[j]
                # Hinting a join between non-neighbors may fail (PG may
                # disregard the hint).
                if not plans_lib.ExistsJoinEdgeInGraph(l, r, join_graph):
                    continue
                for plan in self._enumerate_plan_operators(
                        l,
                        r,
                        planner_config=planner_config,
                        avoid_eq_filters=avoid_eq_filters):
                    possible_joins.append((plan, i, j))
        return possible_joins

    def _enumerate_plan_operators(self,
                                  left,
                                  right,
                                  planner_config=None,
                                  avoid_eq_filters=False):
        join_ops = self.workload_info.join_types
        scan_ops = self.workload_info.scan_types
        if planner_config:
            join_ops = planner_config.KeepEnabledJoinOps(join_ops)
        # Hack.
        if planner_config and planner_config.search_space == 'dbmsx':
            engine = 'dbmsx'
        else:
            engine = 'postgres'
        return search.EnumerateJoinWithOps(
            left,
            right,
            join_ops=join_ops,
            scan_ops=scan_ops,
            avoid_eq_filters=avoid_eq_filters,
            engine=engine,
            use_plan_restrictions=self.use_plan_restrictions)

    # @profile
    # 根据当前状态（state）和可能的连接操作（possible_joins），生成新的部分查询计划状态（new states），
    # 并为每个新状态计算一个“成本”值。 返回 返回来两个列表，状态的costs，新状态s
    def _make_new_states(self, 
                         query_node,
                         state, # 当前的部分查询计划状态（一组未合并完的表或中间连接结果）
                         costs, # 每个 possible_join 的预测代价（延迟、资源消耗等）
                         possible_joins): # 可行的连接操作列表，其中每个 possible_join 是一个三元组：
                                          # join: 一个 JoinNode，表示两个子计划的连接；
                                          # left_idx: 左侧子计划在 state 中的索引；
                                          # right_idx: 右侧子计划在 state 中的索引；
        num_rels = len(state)
        valid_costs = [None] * len(possible_joins)
        valid_new_states = [None] * len(possible_joins)
        for i in range(len(possible_joins)):
            join, left_idx, right_idx = possible_joins[i]
            join.cost = costs[i]
            new_state = state[:]  # Shallow copy.浅拷贝原状态
            new_state[left_idx] = join # 将 left_idx 处的节点替换为 join；
            del new_state[right_idx] # 删除 right_idx 处的节点（因为这两个表已经被合并了）；
            # 计算新状态的“成本” ： 该状态中所有 JoinNode 的最大 cost；
            # 即：如果这个状态包含多个中间连接结果，那么它的成本取这些中最差的一个（最悲观估计）；
            new_state_cost = -1e30
            for rel in new_state:
                if rel.IsJoin():
                    # Goodness(state) = max V_theta(subplan), for all subplan
                    # in state.
                    if rel.cost == None:
                        new_state_cost = max(new_state_cost, self.infer(query_node,[rel])[0])
                    else:
                        new_state_cost = max(new_state_cost, rel.cost)
            valid_costs[i] = new_state_cost
            valid_new_states[i] = new_state
        return valid_costs, valid_new_states

    # @profile
    def _beam_search_bk(self,
                        query_node,
                        beam_size=10, # 束搜索宽度，即每轮保留的最有希望的状态数
                        bushy=False,
                        return_all_found=False, # 是否返回所有找到的完整计划
                        planner_config=None, 
                        verbose=False, # 是否打印详细日志
                        avoid_eq_filters=False, # 是否避免等值过滤条件
                        epsilon_greedy=0): # 在搜索过程中随机选择的概率
        """Produce a plan via beam search.

        Args:
          query_node: a Node, a parsed version of the query to optimize.  In
            principle we should take the raw SQL string, but this is a
            convenient proxy.
          beam_size: size of the fixed set of most promising Nodes to be
            explored.
        """
        # 确保传入的 planner_config 和 bushy 参数一致。
        # 如果是 bushy=True，则必须要求配置允许 bushy 搜索空间；
        # 否则不允许。
        if planner_config:
            if bushy:
                assert planner_config.search_space == 'bushy', planner_config
            else:
                assert planner_config.search_space != 'bushy', planner_config
    
        # 记录开始时间，用于后续统计规划耗时。    
        planning_start_t = time.time()
        # Join graph.
        join_graph, _ = query_node.GetOrParseSql()
        # Base tables to join.
        query_leaves = query_node.GetLeaves() # 所有参与连接的基本表节点列表。
        # A "state" is a list of Nodes, each representing a partial plan. If a
        # state has only one element, then it is a complete plan.
        # 初始状态是一个包含所有叶子节点的列表。
        init_state = query_leaves
        # A fringe is a priority queue of (cost of a state, a state).
        # 优先队列，保存 (cost, state) 对，表示当前待探索的状态。
        # 一开始cost为空
        fringe = [(0, init_state)]

        # Bookkeeping of open (unexpanded) and closed (expanded) states.
        # Reference: page 2 of
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.435.447&rep=rep1&type=pdf
        # TODO: can factor out these logic into a Fringe / a FringeState class.
        # TODO: Unify 'states_open' and 'fringe'.
        # 这两个字典用于避免重复探索相同状态。
        states_open = {}  # StateHash(state) -> cost. # 未扩展过的状态
        states_expanded = {}  # StateHash(state) -> cost. # 已扩展过的状态

        # 将状态转换为字符串集合，再用 frozenset 包裹以保证无序性，最后取哈希。
        # 目的是为了判断两个状态是否相同。
        def StateHash(state):
            """Orderless hashing."""
            return hash(
                frozenset([
                    # to_str() is faster than hint_str(); this can be further
                    # optimized by using shorter strings.
                    subplan.to_str(with_cost=False) for subplan in state
                ]))

        # 下面这些函数封装了对 states_open 和 states_expanded 的操作。

        # 把一个状态加入 open 表；
        def MarkInOpen(state_cost, state, state_hash):
            states_open[state_hash] = state_cost

        # 
        def RemoveFromOpen(state_cost, state):
            h = StateHash(state)
            prev_cost = states_open.pop(h)
            assert prev_cost == state_cost, (prev_cost, state_cost, state,
                                             states_open)

        # 从 open 移动到 expanded；
        def MoveFromOpenToExpanded(state_cost, state):
            h = StateHash(state)
            prev_cost = states_open.pop(h)
            assert prev_cost == state_cost, (prev_cost, state_cost, state,
                                             states_open)
            states_expanded[h] = state_cost

        # 判断某个状态是否已存在 open 或 expanded 中。
        def GetFromOpenOrExpanded(state):
            h = StateHash(state)
            ret = states_open.get(h)
            if ret is not None:
                return ret, h
            return states_expanded.get(h), h

        # 将初始状态加入 open 表中。这个时候cost = 0
        MarkInOpen(0, init_state, StateHash(init_state))

        is_eps_greedy_triggered = False

        # 主循环：束搜索过程
        terminal_states = []
        # 继续搜索直到找到足够多的完整计划（通常是 1 个）或 fringe 为空。
        while len(terminal_states) < self.search_until_n_complete_plans and fringe:
            # 从 fringe 中取出当前成本最低的状态；将其从 open 移动到 expanded。
            # 和广度优先遍历不同，广度优先遍历按照加入顺序弹出
            state_cost, state = fringe.pop(0)
            # 状态从队列中弹出来的时候，从 Open 到 expand，但是对于初始状态，仍然state_cost = 0
            MoveFromOpenToExpanded(state_cost, state)
            # 检查是否是终止状态：如果只剩一个子计划，则已经完成一个完整计划。
            if len(state) == 1:
                # A terminal.
                terminal_states.append((state_cost, state))
                continue

            # 获取当前状态下可以进行的所有合法连接操作；
            possible_plans = self._get_possible_plans(
                query_node,
                state,
                join_graph,
                bushy=bushy,
                planner_config=planner_config,
                avoid_eq_filters=avoid_eq_filters)
            # 模型预测每个连接的成本；
            costs = self.infer(query_node,
                               [join for join, _, _ in possible_plans])
            # 根据连接生成新的部分计划状态。
            valid_costs, valid_new_states = self._make_new_states(
                query_node, state, costs, possible_plans)
            # 更新 fringe（加入新状态）：
            for i, (valid_cost,
                    new_state) in enumerate(zip(valid_costs, valid_new_states)):
                # Add to open if it is not in open or expanded.
                # 如果新状态未被探索过，则加入 fringe 并标记为 open；
                ret, state_hash = GetFromOpenOrExpanded(new_state)
                if ret is None:
                    fringe.append((valid_cost, new_state))
                    MarkInOpen(valid_cost, new_state, state_hash)
                # 否则跳过（因为成本应该是一样的）。
                else:
                    prev_cost = ret
                    assert valid_cost == prev_cost, (valid_cost, prev_cost,
                                                     new_state, states_open,
                                                     states_expanded)
                    
            # ε-greedy 探索策略
            # 以一定概率随机保留fringe中一个状态，丢弃其他
            r = np.random.rand()
            if r < epsilon_greedy:
                # Randomly pick one state in the fringe and discard the rest.
                # Note that 'fringe' at this step can have larger than
                # 'beam_size' elements.
                rand_idx = np.random.randint(len(fringe))
                new_fringe = [fringe[rand_idx]]
                # Remove the discarded states from 'open' so that they may be
                # able to be explored down the line.
                for i, fringe_elem in enumerate(fringe):
                    if i == rand_idx:
                        continue
                    _state_cost, _state = fringe_elem
                    # 从Open移出，接着又从从fringe移出
                    RemoveFromOpen(_state_cost, _state)
                # Swap.
                fringe = new_fringe

                # Debugging.
                self.total_random_triggers += 1
                is_eps_greedy_triggered = True
            
            # fringe中，按照成本排序；只保留前 beam_size 个状态。
            fringe = sorted(fringe, key=lambda x: x[0])
            fringe = fringe[:beam_size]

        # beam search 主过程结束，计算并输出总耗时。
        planning_time = (time.time() - planning_start_t) * 1e3
        print('xiaoma: bs Planning took {:.1f}ms'.format(planning_time))

        # Print terminal_states.
        if verbose:
            print('terminal_states:')

        # 收集所有完整的计划及其成本；找出成本最小的那个作为最终结果。
        all_found = []
        min_cost = np.min([c for c, s in terminal_states])
        min_cost_idx = np.argmin([c for c, s in terminal_states])
        for i, (cost, state) in enumerate(terminal_states):
            all_found.append((cost, state[0]))
            if verbose:
                if cost == min_cost:
                    print('  {:.1f} {}  <-- cheapest'.format(
                        cost,
                        str([s.hint_str(self.plan_physical) for s in state])))
                else:
                    print('  {:.1f} {}'.format(
                        cost,
                        str([s.hint_str(self.plan_physical) for s in state])))
                    
        # 默认返回三项：耗时（毫秒），最优计划节点plan，最优成本lentency
        ret = [
            planning_time, terminal_states[min_cost_idx][1][0],
            terminal_states[min_cost_idx][0]
        ]
        # 如果 return_all_found=True，则追加第四个元素：所有找到的完整计划列表。
        if return_all_found:
            ret.append(all_found)

        self.total_joins += len(query_leaves) - 1
        self.num_queries_with_random += int(is_eps_greedy_triggered)

        return ret
    
    def _mctsStateHash(self,state):
            """Orderless hashing."""
            return hash(
                frozenset([
                    # to_str() is faster than hint_str(); this can be further
                    # optimized by using shorter strings.
                    subplan.to_str(with_cost=False) for subplan in state
                ]))
    def _mctsGetFromExplored(self, state_hash, exploredState):
            assert isinstance(exploredState,dict)
            bsFuncRet = exploredState.get(state_hash)
            return bsFuncRet
    def _mctsMarkInExploredState(selef,state_hash,state_cost,exploredState):
            assert isinstance(exploredState,dict)
            exploredState[state_hash] = state_cost

    def _simulation_beam_search_bk_from_a_state(self,
                        joinHash2CostMap,
                        init_state,
                        query_node,
                        beam_size=10, # 束搜索宽度，即每轮保留的最有希望的状态数
                        bushy=False,
                        return_all_found=False, # 是否返回所有找到的完整计划
                        planner_config=None, 
                        verbose=False, # 是否打印详细日志
                        avoid_eq_filters=False, # 是否避免等值过滤条件
                        epsilon_greedy=0): # 在搜索过程中随机选择的概率
        """Produce a plan via beam search.

        Args:
          query_node: a Node, a parsed version of the query to optimize.  In
            principle we should take the raw SQL string, but this is a
            convenient proxy.
          beam_size: size of the fixed set of most promising Nodes to be
            explored.
        """
        # 确保传入的 planner_config 和 bushy 参数一致。
        # 如果是 bushy=True，则必须要求配置允许 bushy 搜索空间；
        # 否则不允许。
        if planner_config:
            if bushy:
                assert planner_config.search_space == 'bushy', planner_config
            else:
                assert planner_config.search_space != 'bushy', planner_config
    
        # 记录开始时间，用于后续统计规划耗时。    
        planning_start_t = time.time()
        # Join graph.
        join_graph, _ = query_node.GetOrParseSql()
        # Base tables to join.
        query_leaves = query_node.GetLeaves() # 所有参与连接的基本表节点列表。
        # A "state" is a list of Nodes, each representing a partial plan. If a
        # state has only one element, then it is a complete plan.
        # 初始状态是一个包含所有叶子节点的列表。
        init_state = init_state
        # A fringe is a priority queue of (cost of a state, a state).
        # 优先队列，保存 (cost, state) 对，表示当前待探索的状态。
        fringe = [(0, init_state)]

        # Bookkeeping of open (unexpanded) and closed (expanded) states.
        # Reference: page 2 of
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.435.447&rep=rep1&type=pdf
        # TODO: can factor out these logic into a Fringe / a FringeState class.
        # TODO: Unify 'states_open' and 'fringe'.
        # 这两个字典用于避免重复探索相同状态。
        states_open = {}  # StateHash(state) -> cost. # 未扩展过的状态
        states_expanded = {}  # StateHash(state) -> cost. # 已扩展过的状态
        # 将状态转换为字符串集合，再用 frozenset 包裹以保证无序性，最后取哈希。
        # 目的是为了判断两个状态是否相同。
        def StateHash(state):
            """Orderless hashing."""
            return hash(
                frozenset([
                    # to_str() is faster than hint_str(); this can be further
                    # optimized by using shorter strings.
                    subplan.to_str(with_cost=False) for subplan in state
                ]))

        # 下面这些函数封装了对 states_open 和 states_expanded 的操作。

        # 把一个状态加入 open 表；
        def MarkInOpen(state_cost, state, state_hash):
            states_open[state_hash] = state_cost

        # 
        def RemoveFromOpen(state_cost, state):
            h = StateHash(state)
            if h in states_open:
                prev_cost = states_open.pop(h)
                assert prev_cost == state_cost, (prev_cost, state_cost, state,
                                                states_open)

        # 从 open 移动到 expanded；
        def MoveFromOpenToExpanded(state_cost, state):
            h = StateHash(state)
            if h in states_open:
                prev_cost = states_open.pop(h)
                # prev_cost = states_open.pop(h)
                assert prev_cost == state_cost, (prev_cost, state_cost, state,
                                                states_open)
                states_expanded[h] = state_cost
                return h
            else:
                return None

        # 判断某个状态是否已存在 open 或 expanded 中。
        def GetFromOpenOrExpanded(stateHash):
            value_cost = states_open.get(stateHash)
            if value_cost is not None:
                return value_cost, stateHash
            return states_expanded.get(stateHash), stateHash

        # 将初始状态加入 open 表中。
        h = StateHash(init_state)
        MarkInOpen(0, init_state, h)
        # self._mctsMarkInExploredState(state_hash=h,state_cost=0,exploredState=bsExploredState)

        is_eps_greedy_triggered = False

        # 主循环：束搜索过程
        terminal_states = []
        # 继续搜索直到找到足够多的完整计划（通常是 1 个）或 fringe 为空。
        while len(terminal_states) < self.search_until_n_complete_plans and fringe:
            # 从 fringe 中取出当前成本最低的状态；将其从 open 移动到 expanded。
            state_cost, state = fringe.pop(0)
            h = MoveFromOpenToExpanded(state_cost, state)
            # 检查是否是终止状态：如果只剩一个子计划，则已经完成一个完整计划。
            if len(state) == 1:
                # A terminal.
                terminal_states.append((state_cost, state))
                continue

            # 获取当前状态下可以进行的所有合法连接操作；possible_plans : state[i], left_id, right_id
            possible_plans = self._get_possible_plans(
                query_node,
                state,
                join_graph,
                bushy=bushy,
                planner_config=planner_config,
                avoid_eq_filters=avoid_eq_filters)
            # 模型预测每个连接的成本；
            # new_join_list = []
            # newjointmpH_list = []
            # new_possible_plans = []
            # old_possible_plans = []
            # costs = []
            # for idx, (join ,_,_) in enumerate(possible_plans):
            #     th = self._mctsStateHash([join])
            #     cost = joinHash2CostMap.get(th)
            #     if cost is not None:
            #         costs.append(cost)
            #         old_possible_plans.append(possible_plans[idx])
            #     else:
            #         newjointmpH_list.append(th)
            #         new_join_list.append(join)
            #         new_possible_plans.append(possible_plans[idx])
            # newcosts = self.infer(query_node, new_join_list)
            # costs = costs + newcosts
            # tempMap = dict(zip(newjointmpH_list, newcosts))
            # assert isinstance(joinHash2CostMap,dict)
            # joinHash2CostMap.update(tempMap)
            # possible_plans = []
            # possible_plans = old_possible_plans + new_possible_plans
            costs = self.infer(query_node, [join for join, _, _ in possible_plans])
            # 根据连接生成新的部分计划状态。
            valid_costs, valid_new_states = self._make_new_states(
                query_node, state, costs, possible_plans)
            # 更新 fringe（加入新状态）：
            for i, (valid_cost,new_state) in enumerate(zip(valid_costs, valid_new_states)):
                # Add to open if it is not in open or expanded.
                # 如果新状态未被探索过，则加入 fringe 并标记为 open；
                h = StateHash(new_state)
                ret, state_hash = GetFromOpenOrExpanded(h)
                # 如果有关这个状态的结果是已经知道的，那直接根据经验给出结果就行
                # mctsRet = self._mctsGetFromExplored(h, exploredState=exploredState)
                # if mctsRet is not None and mctsRet != 0:
                #     # _, found_plan, predicted_latency, _ = mctsRet
                #     # terminal_states.append((predicted_latency, [found_plan]))
                #     print("mingzhong++")
                #     continue
                if ret is None:
                    # 全新
                    fringe.append((valid_cost, new_state))
                    MarkInOpen(valid_cost, new_state, state_hash)
                    # self._mctsMarkInExploredState(state_hash=state_hash,state_cost=0,exploredState=bsExploredState)
                # 否则跳过（因为成本应该是一样的）。
                else:
                    prev_cost = ret
                    assert valid_cost == prev_cost, (valid_cost, prev_cost,
                                                     new_state, states_open,
                                                     states_expanded)
            # ε-greedy 探索策略
            # 以一定概率随机保留fringe中一个状态，丢弃其他
            r = np.random.rand()
            if r < epsilon_greedy:
                # Randomly pick one state in the fringe and discard the rest.
                # Note that 'fringe' at this step can have larger than
                # 'beam_size' elements.
                rand_idx = np.random.randint(len(fringe))
                new_fringe = [fringe[rand_idx]]
                # Remove the discarded states from 'open' so that they may be
                # able to be explored down the line.
                for i, fringe_elem in enumerate(fringe):
                    if i == rand_idx:
                        continue
                    _state_cost, _state = fringe_elem
                    RemoveFromOpen(_state_cost, _state)
                # Swap.
                fringe = new_fringe

                # Debugging.
                self.total_random_triggers += 1
                is_eps_greedy_triggered = True
            
            # fringe中，按照成本排序；只保留前 beam_size 个状态。
            fringe = sorted(fringe, key=lambda x: x[0])
            fringe = fringe[:beam_size]

        # beam search 主过程结束，计算并输出总耗时。
        planning_time = (time.time() - planning_start_t) * 1e3
        # print('xiaoma: bs Planning took {:.1f}ms'.format(planning_time))

        # Print terminal_states.
        if verbose:
            print('terminal_states:')

        # 收集所有完整的计划及其成本；找出成本最小的那个作为最终结果。
        all_found = []
        min_cost = np.min([c for c, s in terminal_states])
        min_cost_idx = np.argmin([c for c, s in terminal_states])
        for i, (cost, state) in enumerate(terminal_states):
            all_found.append((cost, state[0]))
            if verbose:
                if cost == min_cost:
                    print('  {:.1f} {}  <-- cheapest'.format(
                        cost,
                        str([s.hint_str(self.plan_physical) for s in state])))
                else:
                    print('  {:.1f} {}'.format(
                        cost,
                        str([s.hint_str(self.plan_physical) for s in state])))
                    
        # 默认返回三项：耗时（毫秒），最优计划节点plan，最优成本lentency
        ret = [
            planning_time, terminal_states[min_cost_idx][1][0],
            terminal_states[min_cost_idx][0]
        ]
        # 如果 return_all_found=True，则追加第四个元素：所有找到的完整计划列表。
        if return_all_found:
            ret.append(all_found)

        self.total_joins += len(query_leaves) - 1
        self.num_queries_with_random += int(is_eps_greedy_triggered)
        # for key in bsExploredState:
        #     bsExploredState[key] = ret
        # assert isinstance(exploredState, dict)
        # exploredState.update(bsExploredState)
        return ret

    def SampleRandomPlan(self, query_node, bushy=True):
        """Samples a random, valid plan."""
        planning_start_t = time.time()
        join_graph, _ = query_node.GetOrParseSql()
        query_leaves = query_node.CopyLeaves()
        num_rels = len(query_leaves)
        num_random_plans = 100
        num_random_plans = 1000

        def _SampleOne(state):
            while len(state) > 1:
                possible_plans = self._get_possible_plans(query_node,
                                                          state,
                                                          join_graph,
                                                          bushy=bushy)
                _, valid_new_states = self._make_new_states(
                    query_node, state, [0.0] * len(possible_plans), possible_plans)
                rand_idx = np.random.randint(len(valid_new_states))
                state = valid_new_states[rand_idx]
            predicted = self.infer(query_node, [state[0]])
            return predicted, state

        best_predicted = [np.inf]
        best_state = None
        for _ in range(num_random_plans):
            state = query_leaves
            assert len(state) == num_rels, len(state)
            predicted, state = _SampleOne(state)
            if predicted[0] < best_predicted[0]:
                best_predicted = predicted
                best_state = state

        planning_time = (time.time() - planning_start_t) * 1e3
        predicted = best_predicted
        state = best_state
        print('Found best random plan out of {}:'.format(num_random_plans))
        print('  {:.1f} {}'.format(
            predicted[0], str([s.hint_str(self.plan_physical) for s in state])))
        print('Planning took {:.1f}ms'.format(planning_time))
        return predicted[0], state[0]
