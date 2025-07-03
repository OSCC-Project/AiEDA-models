import os
import random
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde, truncnorm

scripts_dir = '' # the path to store the scripts

OptDict = {
    1: "refactor",
    2: "refactor -z",
    3: "refactor -l",
    4: "refactor -l -z",
    5: "rewrite",
    6: "rewrite -z",
    7: "rewrite -l",
    8: "rewrite -l -z" ,
    9: "resub",
    10: "resub -z",
    11: "resub -l",
    12: "resub -l -z",
    13: "balance",
    14: "balance",
    15: "balance",
    16: "balance"
}

def gen_random_opt_seq_by_fixed_size(len_recipe_one):
    """Generate the random optimization sequence, and the length is the same with the given size
    
    Returns:
        scripts: optimization sequence (str)
    """
    
    scripts = ""
    scripts_num = ""
    len_opt_dict = len(OptDict) - 1
    for i in range(len_recipe_one):
        rnumber = random.randint(1, len_opt_dict)
        scripts += OptDict[rnumber] + ';'
        if rnumber == 14 or rnumber == 15 or rnumber == 16:
            rnumber = 13
        scripts_num += str(rnumber) + ','
    return scripts, scripts_num

def gen_random_opt_seq_by_gaussian(len_recipe_one):
        """Generate the random optimization sequence, and the length is constrained by the Gaussian distribution

        Returns:
            scripts: optimization sequence (str)
        """
        
        scripts = ""
        scripts_num = ""
        len_opt_dict = len(OptDict)
        
        # generate the len_opt_seq_real size follow the Gaussian distribution
        len_opt_seq_real = 0
        mean = len_recipe_one / 2
        std_dev = max(len_recipe_one / 4, 1) 
        lower_bound = 1
        upper_bound = len_recipe_one
        a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
        while True:
            try:
                len_opt_seq_real = truncnorm.rvs(a, b, loc=mean, scale=std_dev)
                len_opt_seq_real = int(round(len_opt_seq_real))
            except Exception as e:
                print("generate normal data wrong!")
                len_opt_seq_real = mean
            if 0 < len_opt_seq_real <= len_recipe_one:
                break
        
        # generage the random optimization sequence
        for i in range(len_opt_seq_real):
            rnumber = rnumber = random.randint(1, len_opt_dict)
            scripts += OptDict[rnumber] + ';'
            scripts_num += str(rnumber) + ','
        return scripts, scripts_num, len_opt_seq_real

def gen_scripts(scripts_dir, script_num, script_len):
    scripts = []
    scripts_num = []
    scripts_txt_dir = os.path.join(scripts_dir, 'scripts.txt')
    scripts_num_dir = os.path.join(scripts_dir, 'scripts_num.txt')
    for i in tqdm(range(script_num)):
        while True:
            script, script_num,_ = gen_random_opt_seq_by_gaussian(script_len)
            if script not in scripts:
                scripts.append(script)
                scripts_num.append(script_num)
                break
    with open(scripts_txt_dir, 'w') as f:
        for script in scripts:
            f.write(script + '\n')

    with open(scripts_num_dir, 'w') as f:
        for script_num in scripts_num:
            f.write(script_num + '\n')

def gen_scripts_csv(scripts_dir, script_num, script_len):
    scripts = []
    scripts_num = []
    scripts_csv_dir = os.path.join(scripts_dir, 'scripts.csv')
    # scripts_num_dir = os.path.join(scripts_dir, 'scripts_num.txt')
    for i in range(script_num):
        while True:
            script, script_num = gen_random_opt_seq_by_fixed_size(script_len)
            if script not in scripts:
                scripts.append(script)
                scripts_num.append(script_num)
                break
    pd.DataFrame({'opt_script': scripts, 'opt_seq': scripts_num}).to_csv(scripts_csv_dir, index=False)

if __name__ == "__main__":
    gen_scripts(scripts_dir, 1500, 20)