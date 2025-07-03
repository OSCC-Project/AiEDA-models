from enum import EnumMeta
import logging
import random
import gym
from gym.envs.user.reader import read_data
import gym.envs.user.dp_reward
import math
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import math
from .abacus import Abacus, Cell, ClustersOnRow
import copy

logger = logging.getLogger(__name__)

'''
最小单位都是 grid
state: [10, 10 , 5]  10行，每行保存前10个 Abacus中的cluster 的属性, e,w,q,x,y
reward: 每一个action 返回abacus reward
action: 每个单元的(i,y)把第i个元素放到第y行中去

'''

index = {'x': 0, 'y': 1, 'w': 2, 'h': 3, 'id': 4}


class PlaceEnv1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, cell_box, row_num, site_num, row_height, site_width):

        # meta data
        self.ROW_MAX = 10
        # self.CELL_NUM_MAX = 3950
        self.WINDOW = 10
        self.POST_CELL_NUM = 30

        # meta data
        # stable state
        # scala
        self.row_num = row_num
        self.site_num = site_num
        self.row_height = row_height
        self.site_width = site_width
        self.cell_num = len(cell_box)
        # vector
        self.origin_cell_box = []
        for (i, tuple) in enumerate(cell_box):
            self.origin_cell_box.append(
                Cell(i, i, tuple[0], tuple[1], tuple[2], tuple[3]))
        self.origin_cell_box.sort(key=lambda t: (t.x + t.w / 2))
        for i in range(len(self.origin_cell_box)):
            self.origin_cell_box[i].id = i
            self.origin_cell_box[i].x = self.origin_cell_box[i].x // site_width
            self.origin_cell_box[i].w = self.origin_cell_box[i].w // site_width
            self.origin_cell_box[i].y = self.origin_cell_box[i].y / row_height
            self.origin_cell_box[i].h = 1

        # self.abacus_legal = Abacus(
        #     self.row_num, self.site_num, self.origin_cell_box)
        self.current_cell_id = 0
        self.observation_space_dimension = 5 * self.WINDOW * self.ROW_MAX
        # self.action_space = spaces.discrete(self.ROW_MAX)
        self.abscus_final = None
        self.failed_action = 0
        self.gamma = 1  # 折扣因子
        self.viewer = None
        self.state = None
        self.reset()
        # self._max_episode_steps = self.cell_num

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def getGamma(self):
        return self.gamma

    def step(self, action):
        # 系统当前状态
        origin_y = self.origin_cell_box[self.current_cell_id].y
        origin_row = math.floor(origin_y + 0.5)
        row_id = origin_row + (action - self.ROW_MAX // 2)
        ok = True
        reward = None

        if row_id < 0 or row_id >= self.row_num:
            ok = False
        else:
            reward = self.abacus_legal.place_row(row_id, self.current_cell_id)
        if reward == None:
            ok = False
        if ok == False:
            is_terminal = False
            # reward = -1e5 * (self.failed_action + 1)
            self.failed_action += 1
            if self.failed_action == 10:
                is_terminal = False
            next_state = self.get_state()
            return next_state, reward, is_terminal, {}
        self.failed_action = 0
        reward = reward * self.site_width * self.site_width
        reward += self.origin_cell_box[self.current_cell_id].w * \
            ((abs(row_id - origin_y) * self.row_height) ** 2)
        self.current_cell_id += 1
        if self.current_cell_id == self.cell_num:
            # print(self.current_cell_id)
            is_terminal = True
            next_state = ([], [], [])
            # ok = self.abacus_legal.check()
            # res = self.abacus_legal.get_quadratic_movement()
            # print(ok)
            return next_state, - reward, is_terminal, {}
        else:
            is_terminal = False
        next_state = self.get_state()
        return next_state, - reward, is_terminal, {}

    def step_all(self, action):
        # 系统当前状态
        for (id, row_add) in enumerate(action):
            origin_y = self.abacus_legal.origin_cells[id].y
            origin_row = math.floor(origin_y + 0.5)
            row_id = origin_row + row_add
            if row_id < 0:
                row_id = 0
            if row_id >= self.row_num:
                row_id = self.row_num - 1
            self.abacus_legal.final_cells[id].y = row_id

        return self.abacus_legal.legalization()

    def abacus_step(self):
        origin_y = self.origin_cell_box[self.current_cell_id].y
        origin_row = math.floor(origin_y + 0.5)
        window_size = 10
        row_list = []
        for l in range(5):
            if (origin_row + l < self.row_num):
                row_list.append(origin_row + l)
            if (origin_row - l - 1 >= 0):
                row_list.append(origin_row - l - 1)

        min_cost = 1e12
        best_row = origin_row
        w = self.origin_cell_box[self.current_cell_id].w
        for row_id in row_list:
            if self.abacus_legal.row_clusters[row_id].sum_width + w > self.site_num:
                continue
            y_cost = w*((abs(row_id - origin_y) * self.row_height)
                        ** 2)
            if y_cost > min_cost:
                break
            cost = y_cost + self.site_width**2 * self.abacus_legal.place_row_trial(
                row_id, self.current_cell_id)
            if cost < min_cost:
                min_cost = cost
                best_row = row_id
        y_cost = w*((abs(best_row - origin_y) * self.row_height)
                    ** 2)
        cost = y_cost + self.site_width**2 * self.abacus_legal.place_row(
            best_row, self.current_cell_id)
        reward = cost
        self.current_cell_id += 1
        if self.current_cell_id == self.cell_num:
            # print(self.current_cell_id)
            is_terminal = True
            next_state = ([], [], [])
            return best_row - origin_row + self.WINDOW//2, next_state, - reward, is_terminal, {}
        else:
            is_terminal = False
        next_state = self.get_state()
        return best_row - origin_row + self.WINDOW//2, next_state, - reward, is_terminal, {}

    def get_post_cell_state(self):
        id = self.current_cell_id
        post_cell_state = []
        for cell_id in range(id, id + self.POST_CELL_NUM):
            if cell_id < self.cell_num:
                c = self.origin_cell_box[cell_id]
                post_cell_state.append([c.x, c.y, c.w, c.h, c.id + 1])
            else:
                post_cell_state.append([0, 0, 0, 0, 0])
        return post_cell_state

    def get_row_density_state(self):
        id = self.current_cell_id
        origin_y = self.origin_cell_box[self.current_cell_id].y
        ori_row = math.floor(origin_y + 0.5)
        row_density_state = []
        for row_id in range(ori_row - self.ROW_MAX // 2, ori_row + self.ROW_MAX // 2, 1):
            if row_id >= 0 and row_id < self.row_num:
                if self.abacus_legal.row_clusters[row_id].sum_width + self.origin_cell_box[id].w <= self.site_num:
                    row_density_state.append(
                        1 - self.abacus_legal.row_clusters[row_id].sum_width / self.site_num)
                else:
                    row_density_state.append(0)
            else:
                row_density_state.append(0)
        return row_density_state

    def get_one_row_cluster_state(self, row_id):
        cluster_on_row = self.abacus_legal.row_clusters[
            row_id] if row_id >= 0 and row_id < self.row_num else ClustersOnRow()
        res_list = []
        num = len(cluster_on_row.clusters)
        for i in range(min(self.WINDOW, num)):
            c = cluster_on_row.clusters[num - i - 1]
            res_list.append([c.e, c.w, c.q, c.x, row_id])
        if len(res_list) < self.WINDOW:
            res_list = res_list + [[0, 0, 0, 0, 0]
                                   for i in range(self.WINDOW - len(res_list))]
        return res_list

    def get_row_cluster_state(self):
        id = self.current_cell_id
        origin_y = self.origin_cell_box[self.current_cell_id].y
        ori_row = math.floor(origin_y + 0.5)
        row_cluster_state = []
        for row_id in range(ori_row - self.ROW_MAX // 2, ori_row + self.ROW_MAX // 2, 1):
            row_cluster_state.append(self.get_one_row_cluster_state(row_id))
        return row_cluster_state

    def get_state(self):
        self.state = []
        row_cluster_state = np.asarray(
            self.get_row_cluster_state())
        row_cluster_state = row_cluster_state.transpose(2, 0, 1)
        row_density_state = np.asarray(self.get_row_density_state())

        post_cell_state = np.asarray(self.get_post_cell_state())
        post_cell_state = post_cell_state.transpose(1, 0)
        return row_cluster_state, row_density_state, post_cell_state

    def reset(self):
        self.failed_action = 0
        self.state = []
        self.current_cell_id = 0
        self.abacus_legal = Abacus(
            self.row_num, self.site_num, self.origin_cell_box.copy())
        if self.abscus_final is None:
            for i in range(self.cell_num):
                self.abacus_step()
            self.abscus_final = copy.deepcopy(self.abacus_legal.origin_cells)
            for cell in self.abacus_legal.final_cells:
                self.abscus_final[cell.id].y = cell.y
            self.abacus_legal = Abacus(
                self.row_num, self.site_num, self.origin_cell_box.copy())
        else:
            self.abacus_legal.final_cells = copy.deepcopy(self.abscus_final)

        return self.abacus_legal.get_quadratic_movement()

    def render(self, mode='human'):
        # from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 600

    def close(self):
        if self.viewer:
            self.viewer.close()
