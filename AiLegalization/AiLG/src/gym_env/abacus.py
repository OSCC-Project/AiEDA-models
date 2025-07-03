'''
Author:
date:
'''

import os
import math
import copy
from tabnanny import check

from typing import List
import multiprocessing as mp
index = {'x': 0, 'y': 1, 'w': 2, 'h': 3, 'id': 4}


class Cell:
    def __init__(self, id=0, origin_id=0, x=0, y=0, w=0, h=0):
        self.id = id
        self.origin_id = origin_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class Cluster:
    def __init__(self):
        self.x = 0
        self.e = self.q = self.w = 0
        self.cells = []

    def add_cell(self, c: Cell):
        self.e += c.w  # weight
        self.q += c.w * (c.x - self.w)
        self.w += c.w
        self.cells.append(c.id)

    def add_cluster(self, b):
        self.e += b.e
        self.q += b.q - 1. * b.e * self.w
        self.w += b.w
        self.cells = self.cells + b.cells


class ClustersOnRow:
    def __init__(self):
        self.clusters = []
        self.sum_width = 0

    def collapse(self, cluster_id: int, x_min: float, x_max: float):
        c = self.clusters[cluster_id]
        c.x = c.q / c.e
        if c.x < x_min:
            c.x = x_min
        elif c.x > x_max - c.w:
            c.x = x_max - c.w
        if (cluster_id > 0
                and self.clusters[cluster_id - 1].x + self.clusters[cluster_id - 1].w + 0.5
                >= c.x):
            self.clusters[cluster_id - 1].add_cluster(c)
            self.clusters.pop()
            self.collapse(cluster_id - 1, x_min, x_max)

    def push_back_cell(self, t, x_min, x_max):
        self.sum_width += t.w
        lastCluster = self.clusters[-1] if len(self.clusters) else None
        if t.x > x_max - t.w:
            t.x = x_max - t.w
        if t.x < x_min:
            t.x = x_min
        if lastCluster is None or lastCluster.x + lastCluster.w + 0.5 < t.x:
            new_cluster = Cluster()
            new_cluster.x = t.x
            new_cluster.add_cell(t)
            self.clusters.append(new_cluster)
        else:
            lastCluster.add_cell(t)
            self.collapse(len(self.clusters) - 1, x_min, x_max)

    def deepcopy(self):
        res = ClustersOnRow()
        res.clusters = copy.deepcopy(self.clusters)
        res.sum_width = self.sum_width
        return res


class Abacus:
    def __init__(self, row_num, site_num, cell_lists):
        self.origin_cells = copy.deepcopy(cell_lists)
        self.final_cells = copy.deepcopy(cell_lists)
        self.row_clusters = [ClustersOnRow() for i in range(row_num)]
        self.site_num = site_num
        self.row_num = row_num

    def place_row(self, row_id: int, cell_id: int):
        cell = self.final_cells[cell_id]
        # self.row_clusters[row_id]
        row_cluster = self.row_clusters[row_id]
        if row_cluster.sum_width + cell.w > self.site_num:
            return None
        self.row_clusters[row_id].push_back_cell(cell, 0, self.site_num)
        last_cluster = self.row_clusters[row_id].clusters[-1]
        x = math.floor(last_cluster.x + 0.5)
        incCost = 0
        for id in last_cluster.cells:
            # if incCost > 0:
            #     print(1213)
            orx = self.origin_cells[id].x
            prex = self.final_cells[id].x
            self.final_cells[id].x = x
            self.final_cells[id].y = row_id
            incCost += self.final_cells[id].w * (1. * (x - orx) * (
                x - orx) - 1. * (prex - orx) * (prex - orx))
            x += self.final_cells[id].w
            x = math.floor(x)
        return incCost

    def place_row_trial(self, row_id: int, cell_id: int):
        cell = self.final_cells[cell_id]
        # self.row_clusters[row_id]
        row_cluster = self.row_clusters[row_id].deepcopy()
        if row_cluster.sum_width + cell.w > self.site_num:
            return None
        row_cluster.push_back_cell(cell, 0, self.site_num)
        last_cluster = row_cluster.clusters[-1]
        x = math.floor(last_cluster.x + 0.5)
        incCost = 0
        for id in last_cluster.cells:
            orx = self.origin_cells[id].x
            prex = self.final_cells[id].x
            incCost += self.final_cells[id].w * \
                ((x - orx) ** 2 - (prex - orx) ** 2)
            x += self.final_cells[id].w
            x = math.floor(x + 0.5)
        return incCost

    def get_quadratic_movement(self):
        res = 0
        for i in range(len(self.final_cells)):
            f_cell = self.final_cells[i]
            o_cell = self.origin_cells[i]
            res += f_cell.w * ((f_cell.x - o_cell.x)**2 +
                               ((f_cell.y - o_cell.y)*8)**2)
        return res

    def check(self):
        ok = True
        cell_num = 0
        for row_id in range(self.row_num):
            row_cluster = self.row_clusters[row_id]
            pre_x = 0
            for cluster in row_cluster.clusters:
                x = self.final_cells[cluster.cells[0]].x
                if pre_x > x or x < 0:
                    ok = False
                for id in cluster.cells:
                    if x > self.final_cells[id].x:
                        ok = False
                    x += self.final_cells[id].w
                    cell_num += 1
                pre_x = x
                if x > self.site_num:
                    ok = False
        if cell_num != len(self.final_cells):
            ok = False
        return ok

    def place_one_row(self, row_id, cell_list):
        cell_list.sort(
            key=lambda cell: cell.x + cell.w/2)
        # self.row_clusters[row_id]
        for c in cell_list:
            self.place_row(row_id, c.id)

    def work_process(self, worker_args):
        row_id, cells = worker_args
        self.place_one_row(row_id, cells)

    def legalization(self):
        pool = None
        # pool = mp.Pool(1)
        #  = ((self.get_reward, self._get_weights_try(
        #     self.weights, p)) for p in population)
        row_cells = [[] for i in range(self.row_num)]
        for cell in self.final_cells:
            row_cells[math.floor(cell.y)].append(cell)

        worker_args = [(row_id, cells)
                       for (row_id, cells) in enumerate(row_cells)]
        if pool is None:
            for tu in worker_args:
                self.work_process(tu)
            # pass
        else:
            pool.map(self.work_process, worker_args)
            pool.close()
            # 运行进程池中的进程
            pool.join()

        if not self.check():
            return self.final_cells, -1e10
        return self.final_cells, -self.get_quadratic_movement()
