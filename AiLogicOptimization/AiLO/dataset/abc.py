import os
import re
from tqdm import tqdm
import subprocess
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataset import load_mean_std

from utils import *

scripts_dir = '' # the path of the optimization scripts
data_dir =''   #the path to store the data
graphml_tool = '' # the path of the aig2graphml tool
tool_abc = '' # the path of the abc tool
liberty = '' # the path of the liberty file, such as asap7.lib

def line2arr(line):
    operations = line.split(';')
    opt_numbers = []
    for operation in operations:
        operation = operation.strip()
        for key, value in OptDict.items():
            if operation == value:
                opt_numbers.append(key)
                break
    return opt_numbers

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
    # print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text= True)
    output = result.stdout

    area_match = re.search(r'area =([\d\.]+)', output)
    delay_match = re.search(r'delay =([\d\.]+)', output)

    area = float(area_match.group(1)) if area_match else None
    delay = float(delay_match.group(1)) if delay_match else None
    return area, delay


def apply_abc_to_label(aig_dir, scripts_dir, liberty, abs_tool_abc, csv_dir):
    """ Apply the logic optimization of all given AIG and the optimization sequence """
    with open(scripts_dir, 'r') as file:
            lines = file.readlines()
    basename = os.path.basename(aig_dir)
    aig_name = os.path.splitext(basename)[0]
    aig_names = []
    opt_scripts = []
    areas = []
    delays = []
    with ThreadPoolExecutor(max_workers = 32) as executor:
        futures = []
        for line in lines:
            future = executor.submit(apply_abc_optimization, aig_dir, liberty, line, abs_tool_abc)
            futures.append(future)
        
        pbar = tqdm(total=len(lines), desc=f"{aig_name}")
        for i, future in enumerate(as_completed(futures)):
            area, delay, opt_script = future.result()
            aig_names.append(aig_name)
            opt_scripts.append(opt_script)
            areas.append(area)
            delays.append(delay)
            pbar.set_description(f"{aig_name}")
            pbar.update(1)
        pbar.close()
    data = {
        'file' : aig_name,
        'opt_script' : opt_scripts,
        'areas' : areas,
        'delays' : delays
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_dir)

def apply_abc(data_dir, des_class, designs, scripts_dir):
    
    des_dir = os.path.join(data_dir, des_class, design)
    design_area_dir = os.path.join(des_dir, 'des_area.csv')
    design_delay_dir = os.path.join(des_dir, 'des_delay.csv')
    character_dir = os.path.join(des_dir, 'character.csv')
    aig_dir = os.path.join(data_dir, f'benchmark/{des_class}/{design}.aig')
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    scripts_txt_dir = os.path.join(scripts_dir, 'scripts.txt')
    areas, delays = [], []
    opt_scripts = []
    opt_seqs = []

    with open(scripts_txt_dir, 'r') as file:
        lines = file.readlines()

    def process_line(line):
        area, delay = apply_abc_optimization(aig_dir, liberty, line, tool_abc)
        return line.strip(), area, delay

    with ThreadPoolExecutor() as executor:
        future_to_line = {executor.submit(process_line, line): line for line in lines}
        for future in tqdm(as_completed(future_to_line), total=len(lines), desc=f"Processing {design}"):
            line, area, delay = future.result()
            areas.append(area)
            delays.append(delay)
            opt_scripts.append(line)
            opt_seqs.append(line2arr(line))

    pd.DataFrame({'opt_script': opt_scripts, 'opt_seq': opt_seqs, 'area': areas, 'delay': delays}).to_csv(character_dir, index=False)

    max_area = max(areas)
    min_area = min(areas)
    max_delay = max(delays)
    min_delay = min(delays)
    mean_area = np.mean(areas)
    mean_delay = np.mean(delays)
    std_area = np.std(areas)
    std_delay = np.std(delays)

    pd.DataFrame({'file': [design], 'max': [max_area], 'min': [min_area], 'mean': [mean_area], 'std': [std_area]}).to_csv(design_area_dir, index=False)
    pd.DataFrame({'file': [design], 'max': [max_delay], 'min': [min_delay], 'mean': [mean_delay], 'std': [std_delay]}).to_csv(design_delay_dir, index=False)

def eval_abc(data_dir, des_class, design, scripts_dir):
    scripts_txt_dir = os.path.join(scripts_dir, 'scripts_eval.txt')
    des_dir = os.path.join(data_dir, des_class, design)
    design_area_dir = os.path.join(des_dir, 'des_area.csv')
    design_delay_dir = os.path.join(des_dir, 'des_delay.csv')
    eval_dir = os.path.join(des_dir, 'eval.csv')
    aig_dir = os.path.join(data_dir, f'benchmark/{des_class}/{design}.aig')
    areas, delays = [], []
    opt_scripts = []
    opt_seqs = []

    mean_area, std_area = load_mean_std(design_area_dir)
    mean_delay, std_delay = load_mean_std(design_delay_dir)

    with open(scripts_txt_dir, 'r') as file:
        lines = file.readlines()

    def process_line(line):
        area, delay = apply_abc_optimization(aig_dir, liberty, line, tool_abc)
        # print(area)
        return line.strip(), area, delay

    with ThreadPoolExecutor() as executor:
        future_to_line = {executor.submit(process_line, line): line for line in lines}
        for future in tqdm(as_completed(future_to_line), total=len(lines), desc=f"Processing {design}"):
            line, area, delay = future.result()
            # areas.append((area-mean_area)/std_area)
            # delays.append((delay-mean_delay)/std_delay) 
            areas.append(area)
            delays.append(delay)
            opt_scripts.append(line)
            opt_seqs.append(line2arr(line))

    pd.DataFrame({'opt_script': opt_scripts, 'opt_seq': opt_seqs, 'area': areas, 'delay': delays}).to_csv(eval_dir, index=False)
    

if __name__ == "__main__":
    des_class = 'EPFL'
    for design in designs:
        apply_abc(data_dir, des_class, design, scripts_dir)
        eval_abc(data_dir, des_class, design, scripts_dir)
            


