from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import re
import os
import csv
import subprocess
from tqdm import tqdm
from model.gnntr_train import CrossLO
from dataset.utils import *
from dataset.syngen import gen_random_opt_seq_by_gaussian, gen_scripts_csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import torch
from torch_geometric.loader import DataLoader

from dataset.dataset import convert_aig_to_graphml, nsga_Data, DSEDataset

aig_in = '' # aig file path
liberty = '' # liberty file path asap7.lib
abs_tool_abc = '' # abc path
graphml_tool = '' # aig2graphml tool path
result_dir = './nsga'
checkpoint_dir = './checkpoint'
aig_name = os.path.basename(aig_in).split('.')[0]
des_res_dir = os.path.join(result_dir,aig_name)
data_dir = '' # data dir
des_class = 'EPFL'
root_dir = os.path.join(data_dir, des_class)
des_dir = os.path.join(root_dir, aig_name)
if os.path.exists(des_res_dir) is False:
    os.makedirs(des_res_dir)
csv_dir = os.path.join(des_dir,'scripts.csv')
graphml_dir = os.path.join(des_res_dir,f'{aig_name}.graphml')
batch_size = 20
rand_num = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_all_dir = '' # csv path containing abc run results
def apply_abc_optimization(aig_in, liberty, opt_script, abs_tool_abc):
    """ Apply the logic optimization of one given AIG and the optimization sequence
    
    Args:
        aig_in (str): path of the source AIG file
        opt_script (str): the optimization scripts
    Returns:
        area, delay (float, float): the area and delay of the optimized circuit
    """
    
    script = "read_aiger {0}; read_lib {1}; strash; {2} map; print_stats".format(aig_in, liberty, opt_script)
    command = "{0} -c \"{1}\"".format(abs_tool_abc, script)
    result = subprocess.run(command, shell=True, capture_output=True, text= True)
    output = result.stdout

    area_match = re.search(r'area =([\d\.]+)', output)
    delay_match = re.search(r'delay =([\d\.]+)', output)

    area = float(area_match.group(1)) if area_match else None
    delay = float(delay_match.group(1)) if delay_match else None
    global num_abc 
    tmp = int(num_abc/101)
    csv_dir = os.path.join(csv_all_dir,f'{aig_name}{tmp}.csv')
    if os.path.exists(csv_dir):
        with open(csv_dir, 'a') as f:
            f.write(f"{opt_script},{area},{delay}\n")
    else:
        with open(csv_dir, 'w') as f:
            f.write("seq,area,delay\n")
            f.write(f"{opt_script},{area},{delay}\n")
    
    num_abc = num_abc + 1

    return area, delay

class Individual(object):
    def __init__(self):
        self.solution = None  
        self.objective = defaultdict()

        self.n = 0  
        self.rank = 0  
        self.S = [] 
        self.distance = 0  

    def bound_process(self, bound_min, bound_max):
        
        for i, item in enumerate(self.solution):
            if item > bound_max:
                self.solution[i] = bound_max
            elif item < bound_min:
                self.solution[i] = bound_min

    def calculate_objective(self, objective_fun):
        self.objective = objective_fun(self.solution)

    def __lt__(self, other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        for i in range(len(v1)):
            if v1[i] > v2[i]:
                return 0  
        return 1

def fast_non_dominated_sort(P):
    
    F = defaultdict(list)

    for p in P:
        p.S = []
        p.n = 0
        for q in P:
            if p < q:  # if p dominate q
                p.S.append(q)  # Add q to the set of solutions dominated by p
            elif q < p:
                p.n += 1  # Increment the domination counter of p
        if p.n == 0:
            p.rank = 1
            F[1].append(p)

    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q

    return F


def crowding_distance_assignment(L):
    l = len(L)  # number of solution in F

    for i in range(l):
        L[i].distance = 0  # initialize distance

    for m in L[0].objective.keys():
        L.sort(key=lambda x: x.objective[m])  # sort using each objective value
        L[0].distance = float('inf')
        L[l - 1].distance = float('inf')  # so that boundary points are always selected

        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]

        try:
            for i in range(1, l - 1):  # for all other points
                L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
        except Exception:
            print(str(m) + "目标方向上，最大值为" + str(f_max) + "最小值为" + str(f_min))


def binary_tournament(ind1, ind2):

    if ind1.rank != ind2.rank:  
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:  
        return ind1 if ind1.distance > ind2.distance else ind2
    else:  
        return ind1

def AI_make_pop(num=10):
    gen_scripts_csv(des_dir, rand_num, 10)
    if os.path.exists(graphml_dir) is False:
        convert_aig_to_graphml(aig_in, graphml_dir, graphml_tool)

    nsga_data = DSEDataset(des_dir, aig_name, target = 'dse', force_reload=True)
    # print(nsga_data[1].opt_seq)
    model_area = CrossLO(batch_size=batch_size).to(device)
    area_ckp = '' # the path of AI model area checkpoint 
    delay_ckp = '' # the path of AI model delay checkpoint 
    model_area.load_state_dict(torch.load(area_ckp))
    model_delay = CrossLO(batch_size=batch_size).to(device)
    model_delay.load_state_dict(torch.load(delay_ckp))
    loader = DataLoader(nsga_data, batch_size=batch_size, shuffle=False)
    area_preds = []
    delay_preds = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            area_pred = model_area(data)
            delay_pred = model_delay(data)
        area_preds.append(area_pred)
        delay_preds.append(delay_pred)
    area_preds = torch.cat(area_preds)
    delay_preds = torch.cat(delay_preds)
    area_preds_np = area_preds.cpu().numpy()
    delay_preds_np = delay_preds.cpu().numpy()
    df = pd.read_csv(csv_dir)
    df['area_pred'] = area_preds_np
    df['delay_pred'] = delay_preds_np
    df['QoR_pred'] = df['area_pred'] + df['delay_pred']
    top_rows = df.nsmallest(num, 'QoR_pred')
    return top_rows

def make_new_pop(P, eta, bound_min, bound_max, objective_fun):
    popnum = len(P)
    Q = []
    # binary tournament selection
    for i in range(int(popnum / 2)):
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tournament(P[i], P[j])
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent2 = binary_tournament(P[i], P[j])

        while (parent1.solution == parent2.solution).all(): 
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = binary_tournament(P[i], P[j])

        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun)

        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q

def init_and_evaluate(individual, objective_fun, opt_seq_str):
    individual.solution = np.array(list(map(int, opt_seq_str.split(','))), dtype=int)
    individual.calculate_objective(objective_fun)
    return individual

def make_new_pop_with_AI(P, eta, popnum, bound_min, bound_max, objective_fun):
    popnum = popnum
    popnum_p = len(P)
    Q = []
    # binary tournament selection
    for i in range(int(popnum_p / 2)):
        i = random.randint(0, popnum_p - 1)
        j = random.randint(0, popnum_p - 1)
        parent1 = binary_tournament(P[i], P[j])

        i = random.randint(0, popnum_p - 1)
        j = random.randint(0, popnum_p - 1)
        parent2 = binary_tournament(P[i], P[j])

        while (parent1.solution == parent2.solution).all():  
            i = random.randint(0, popnum_p - 1)
            j = random.randint(0, popnum_p - 1)
            parent2 = binary_tournament(P[i], P[j])

        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun)

        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    top_rows = AI_make_pop(popnum-int(popnum_p / 2)*2)

    with ThreadPoolExecutor() as executor:
        tasks = [(Individual(), objective_fun, top_rows.iloc[i]['opt_seq'].strip(',')) for i in range(popnum-int(popnum_p / 2)*2)]
        futures = [executor.submit(init_and_evaluate, *task) for task in tasks]

        for future in as_completed(futures):
            individual = future.result()
            Q.append(individual)

    return Q

def calculate_objective_for_individual(individual, objective_fun):
    individual.calculate_objective(objective_fun)
    return individual

def crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun):

    poplength = len(parent1.solution)

    offspring1 = Individual()
    offspring2 = Individual()
    offspring1.solution = np.empty(poplength)
    offspring2.solution = np.empty(poplength)

    for i in range(poplength):
        rand = random.random()
        beta = (rand * 2) ** (1 / (eta + 1)) if rand < 0.5 else (1 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
        offspring1.solution[i] = int(0.5 * ((1 + beta) * parent1.solution[i] + (1 - beta) * parent2.solution[i]))
        offspring2.solution[i] = int(0.5 * ((1 - beta) * parent1.solution[i] + (1 + beta) * parent2.solution[i]))

    for i in range(poplength):
        mu = random.random()
        delta = int(2 * mu ** (1 / (eta + 1)) if mu < 0.5 else (1 - (2 * (1 - mu)) ** (1 / (eta + 1))))
        offspring1.solution[i] = offspring1.solution[i] + delta

    offspring1.bound_process(bound_min, bound_max)
    offspring2.bound_process(bound_min, bound_max)

    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(calculate_objective_for_individual, offspring1, objective_fun)
        future2 = executor.submit(calculate_objective_for_individual, offspring2, objective_fun)
        
        offspring1 = future1.result()
        offspring2 = future2.result()

    return [offspring1, offspring2]

def QoR(x):
    f = defaultdict(float)
    f[1] = 0
    f[2] = 0
    script = ''
    for num in x:
        if num == 14 or num == 15 or num == 16:
            num = 13
        script += f'{OptDict[num]};'
    f[1], f[2]= apply_abc_optimization(aig_in,liberty,script,abs_tool_abc)
    return f

def plot_P(P):

    X = []
    Y = []
    for ind in P:
        X.append(ind.objective[1])
        Y.append(ind.objective[2])

    plt.xlabel('area')
    plt.ylabel('delay')
    plt.scatter(X, Y)

def main():
    QoRs_=[]
    areas=[]
    delays=[]
    t1 = time.time() 
    for seed in range(5):
        
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        resyn2 = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
        resyn2_area, resyn2_delay = apply_abc_optimization(aig_in, liberty, resyn2, abs_tool_abc)
        print(f'Resyn2: area: {resyn2_area} delay: {resyn2_delay}')

        aigname = os.path.basename(aig_in).split('.')[0]
        aig_dir = os.path.join(result_dir, aigname)
        if not os.path.exists(aig_dir):
            os.makedirs(aig_dir)

        generations = 9  # 迭代次数
        popnum = 10  # 种群大小
        eta = 1  # 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。

        poplength = 10  # 单个个体解向量的维数
        bound_min = 1  # 定义域
        bound_max = 16
        objective_fun = QoR

        P = []
        top_rows = AI_make_pop(popnum)

        with ThreadPoolExecutor() as executor:
            tasks = [(Individual(), objective_fun, top_rows.iloc[i]['opt_seq'].strip(',')) for i in range(popnum)]
            futures = [executor.submit(init_and_evaluate, *task) for task in tasks]

            for future in as_completed(futures):
                individual = future.result()
                P.append(individual)

        fast_non_dominated_sort(P)
        Q = []

        P_t = P  
        Q_t = Q  
        QoRs = []
        # flag = False
        for gen_cur in tqdm(range(generations), desc=f'{aigname} NSGA with AI({seed})'):
            R_t = P_t + Q_t  
            F = fast_non_dominated_sort(R_t)

            P_n = [] 
            i = 1
            while len(P_n) + len(F[i]) < popnum:  # until the parent population is filled
                crowding_distance_assignment(F[i])  # calculate crowding-distance in F_i
                P_n = P_n + F[i]  # include ith non dominated front in the parent pop
                i = i + 1  # check the next front for inclusion
            F[i].sort(key=lambda x: x.distance)  # sort in descending order using <n，因为本身就在同一层，所以相当于直接比拥挤距离

            Q_n = make_new_pop_with_AI(P_n, eta, popnum, bound_min, bound_max,
                            objective_fun)  # use selection,crossover and mutation to create a new population Q_n

            P_t = P_n
            Q_t = Q_n

            best_solutions = F[1] +F[2] 
            qors = []
            for solution in best_solutions:
                script = ''
                for num in solution.solution:
                    if num == 14 or num == 15 or num == 16:
                        num = 13
                    script += f'{OptDict[num]};'
                qor = 2 - (solution.objective[1]/resyn2_area + solution.objective[2]/resyn2_delay)
                qors.append([script, qor, solution.objective[1], solution.objective[2]])
            max_qor_row = max(qors, key=lambda x: x[1])

            QoRs.append(max_qor_row)
            print(f" QoR: {max_qor_row[1]}, area: {max_qor_row[2]}, delay: {max_qor_row[3]}, Solution: {max_qor_row[0]},")

            plt.clf()
            plt.title('current generation:' + str(gen_cur + 1))
            plot_P(P_t)
            save_dir = os.path.join(aig_dir,f'result/{gen_cur}.png')
            if not os.path.exists(os.path.dirname(save_dir)):
                os.makedirs(os.path.dirname(save_dir))
            plt.savefig(save_dir)
            plt.close()
            if gen_cur == generations - 1:
                QoRs_.append(max_qor_row[1])
                areas.append(max_qor_row[2])
                delays.append(max_qor_row[3])
            
    t2 = time.time()
    print('time:',(t2-t1)/5)
    print('mean:',sum(QoRs_)/len(QoRs_),'std:',np.std(QoRs_), 'area:',sum(areas)/len(areas), 'delay',sum(delays)/len(delays))

if __name__ == '__main__':
    main()
