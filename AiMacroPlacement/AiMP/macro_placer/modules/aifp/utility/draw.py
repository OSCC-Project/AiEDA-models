import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp
from aimp.aifp.database.data_structure.instance import PyInstance
from aimp.aifp.database.data_structure.core import PyCore
from typing import List


def draw_layout(save_path:str, inst_list:List[PyInstance], core:PyCore):
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for inst in inst_list:
        if inst.get_width() == 0:
            continue
        cluster_type = inst.get_type()
        if cluster_type == "macro":
            color = 'r'
        elif cluster_type == "stdcell_cluster":
            color = 'y'
        elif cluster_type == "io_cluster":
            color = 'b'
        else:
            color = 'g'
        
        # cluster_low_x = (inst.get_low_x() - core.get_low_x()) /core.get_width()
        # cluster_low_y = (inst.get_low_y() - core.get_low_y()) / core.get_height()
        # cluster_width = inst.get_width()/core.get_width()
        # cluster_height = inst.get_height()/core.get_height()
        low_x = inst.get_low_x()
        low_y = inst.get_low_y()
        width = inst.get_width()
        height = inst.get_height()
        
        rect = patches.Rectangle((low_x, low_y), width, height, linewidth=0.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.xlim([core.get_low_x(), core.get_low_x() + core.get_width()])
    plt.ylim([core.get_low_y(), core.get_low_y() + core.get_height()])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()