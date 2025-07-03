import os
import re
import numpy as np
import logging
from aimp.aifp.database.data_structure.fp_solution import FPSolution
from aimp.macroPlaceDB import MacroPlaceDB


def save_fp_solution(fp_solution:FPSolution, save_file_path:str):
    f = open(save_file_path, 'w')
    f.write(fp_solution.__str__())


def rl_placer_report_and_write_mp_db(log_dir, report_file_path:str=None, mp_db:MacroPlaceDB=None):
    "read log_files in rl_placer_log_dir and generates report file"
    def get_macro_info(lines, start_line_idx):
        macro_info = []
        while lines[start_line_idx].startswith('macro name'):
            line = lines[start_line_idx].split(',')
            # assert len(line) == 3
            macro = {
                'name': line[0].split(' ')[-1],
                'low_x': float(re.findall(r"-?\d+\.?\d*", line[1])[0]),
                'low_y': float(re.findall(r"-?\d+\.?\d*", line[2])[0]),
                }
            if (len(line) > 3): # some old version does not log macro orientation
                macro['orient'] = line[3].replace(' ', '').split(':')[-1]
            else:
                macro['orient'] = 'N'  # give a default orientation

            macro_info.append(macro)
            start_line_idx += 1
        next_line_idx = start_line_idx
        return macro_info, next_line_idx
    
    print('generating report_files from log_files...')
    log_files = os.listdir(log_dir)

    report_score_dict = {
        'wirelength': [],
        'overflow': [],
        'reward': [],
        'min_wirelength':1e20,
        'min_overflow':1e20,
        'min_wirelength_macro_info': {},
        'min_overflow_macro_info': {}
    }

    for env_log in filter(lambda f: f.startswith('env_'), log_files):
        log_read = open(log_dir + '/' + env_log, 'r')
        lines = log_read.readlines()
        # line_idx = 0
        line_idx = int(len(lines) * 0.8)
        while (line_idx < len(lines)):
            # fine reward record
            # if not lines[line_idx].startswith('reward'):
            if not lines[line_idx].startswith('overlap'):
                line_idx += 1
                continue
            overlap_flag = lines[line_idx].split(':')[-1].replace(' ', '')
            reward = float(re.findall(r"-?\d+\.?\d*", lines[line_idx+1])[0])
            wirelength = float(re.findall(r"-?\d+\.?\d*", lines[line_idx+2])[0])
            overflow = float(re.findall(r"-?\d+\.?\d*", lines[line_idx+3])[0])

            if overlap_flag == 'True':  # ignore overlapped result
                line_idx = line_idx + 4
                continue

            report_score_dict['reward'].append(reward)
            report_score_dict['wirelength'].append(wirelength)
            report_score_dict['overflow'].append(overflow)

            flag = False
            if wirelength < report_score_dict['min_wirelength']:
                flag = True
                report_score_dict['min_wirelength'] = wirelength
                report_score_dict['min_wirelength_macro_info'], next_line_idx = get_macro_info(lines, line_idx + 4)
            if overflow < report_score_dict['min_overflow']:
                flag = True
                report_score_dict['min_overflow'] = overflow
                report_score_dict['min_overflow_macro_info'], next_line_idx = get_macro_info(lines, line_idx + 4)
            line_idx = line_idx + 4 if not flag else next_line_idx

            # line_idx = line_idx + 3 if not flag else next_line_idx
        log_read.close()
    
    # write to report file
    logging.info('============ final report =============')
    if report_file_path == None:
        report_file_path = '{}/RLPlacer.rpt'.format(log_dir)
    report_file_write = open(report_file_path, 'w')

    report_file_write.write('min_wirelength:{}\n'.format(report_score_dict['min_wirelength']))
    report_file_write.write('average_wirelength:{}\n'.format(np.mean(report_score_dict['wirelength'])))
    logging.info('min_wirelength:{}\n'.format(report_score_dict['min_wirelength']))
    logging.info('average_wirelength:{}\n'.format(np.mean(report_score_dict['wirelength'])))


    report_file_write.write('min_overflow:{}\n'.format(report_score_dict['min_overflow']))
    report_file_write.write('average_overflow:{}\n'.format(np.mean(report_score_dict['overflow'])))
    logging.info('min_overflow:{}\n'.format(report_score_dict['min_overflow']))
    logging.info('average_overflow:{}\n'.format(np.mean(report_score_dict['overflow'])))

    report_file_write.write('max_reward:{}\n'.format(np.max(report_score_dict['reward'])))
    report_file_write.write('average_reward:{}\n'.format(np.mean(report_score_dict['reward'])))
    logging.info('max_reward:{}\n'.format(np.max(report_score_dict['reward'])))
    logging.info('average_reward:{}\n'.format(np.mean(report_score_dict['reward'])))

    report_file_write.write('\nmin_wirelength_macro_info:\n')
    report_file_write.write('=============================\n')
    logging.info('\nmin_wirelength_macro_info:\n')
    logging.info('=============================\n')
    for macro in report_score_dict['min_wirelength_macro_info']:
        report_file_write.write('name:{} low_x:{} low_y:{}\n'.format(macro['name'], macro['low_x'], macro['low_y']))
        logging.info('name:{} low_x:{} low_y:{}\n'.format(macro['name'], macro['low_x'], macro['low_y']))


    # report_file_write.write('\nmin_overflow_macro_info:\n')
    # report_file_write.write('=============================\n')
    # logging.info('\nmin_overflow_macro_info:\n')
    # logging.info('=============================\n')
    # for macro in report_score_dict['min_overflow_macro_info']:
    #     report_file_write.write('name:{} low_x:{} low_y:{}\n'.format(macro['name'], macro['low_x'], macro['low_y']))
    #     logging.info('name:{} low_x:{} low_y:{}\n'.format(macro['name'], macro['low_x'], macro['low_y']))

    print('report_files write to {}'.format(report_file_path))

    # generate place_macro_tcl for innovus to evaluate fp result
    
    generate_innovus_place_macro_tcl(report_score_dict, '{}/aifp_place_macro.tcl'.format(log_dir))
    report_file_write.close()

    # update to macroPlaceDB
    if mp_db is not None:
        print('writing back to mp_db...')
        for macro in report_score_dict['min_wirelength_macro_info']:
            node_idx = mp_db.node_name2id_map[macro['name']]
            print('write macro {}:{}'.format(mp_db.node_name2id_map[macro['name']], node_idx))
            mp_db.node_x[node_idx] = macro['low_x']
            mp_db.node_y[node_idx] = macro['low_y']
    else:
        print('not writing mp_db')


def generate_innovus_place_macro_tcl(report_score_dict, file_path='aifp_place_macro.tcl'):
    f = open(file_path, 'w')
    # macro is N or FN in manual_def
    for macro in report_score_dict['min_overflow_macro_info']:
        place_instance_cmd = 'placeInstance {} {} {} {}\n'.format(macro['name'], macro['low_x'] / 2000, macro['low_y'] / 2000, macro['orient'])
        set_status_cmd = 'setInstancePlacementStatus -status fixed -name {}\n'.format(macro['name'])
        f.write(place_instance_cmd)
        f.write(set_status_cmd)


    """
    placeInstance u0_rcg/u0_pll 280.408 3040.0 MX
    setInstancePlacementStatus -status fixed -name u0_rcg/u0_pll
    placeInstance u0_rcg/u1_pll 280.408 2200.0 MX
    setInstancePlacementStatus -status fixed -name u0_rcg/u1_pll
    placeInstance u0_soc_top/sram0 3105.717529296875 714.1587524414062 R180
    setInstancePlacementStatus -status fixed -name u0_soc_top/sram0
    placeInstance u0_soc_top/sram1 1275.8677978515625 823.8112182617188 R180
    setInstancePlacementStatus -status fixed -name u0_soc_top/sram1
    placeInstance u0_soc_top/sram2 2190.792724609375 3345.818359375 R180
    setInstancePlacementStatus -status fixed -name u0_soc_top/sram2
    placeInstance u0_soc_top/sram3 1847.6959228515625 1152.768798828125 R180
    setInstancePlacementStatus -status fixed -name u0_soc_top/sram3
    placeInstance u0_soc_top/sram4 1047.136474609375 3345.818359375 R180
    setInstancePlacementStatus -status fixed -name u0_soc_top/sram4
    placeInstance u0_soc_top/sram5 1847.6959228515625 714.1587524414062 R180
    setInstancePlacementStatus -status fixed -name u0_soc_top/sram5
    placeInstance u0_soc_top/sram6 246.57717895507812 1372.07373046875 R180
    setInstancePlacementStatus -status fixed -name u0_soc_top/sram6
    placeInstance u0_soc_top/sram7 246.57717895507812 1591.378662109375 R180
    setInstancePlacementStatus -status fixed -name u0_soc_top/sram7
    placeInstance u0_soc_top/u0_vga_ctrl/vga/buffer11 1871.5703125 1465.981201171875 R0
    setInstancePlacementStatus -status fixed -name u0_soc_top/u0_vga_ctrl/vga/buffer11
    placeInstance u0_soc_top/u0_vga_ctrl/vga/buffer12 3358.3232421875 2672.15869140625 R0
    setInstancePlacementStatus -status fixed -name u0_soc_top/u0_vga_ctrl/vga/buffer12
    placeInstance u0_soc_top/u0_vga_ctrl/vga/buffer21 1757.20458984375 2562.506103515625 R0
    setInstancePlacementStatus -status fixed -name u0_soc_top/u0_vga_ctrl/vga/buffer21
    placeInstance u0_soc_top/u0_vga_ctrl/vga/buffer22 2557.763916015625 2562.506103515625 R0
    setInstancePlacementStatus -status fixed -name u0_soc_top/u0_vga_ctrl/vga/buffer22
    """





# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     log_dirs = [
#         '/home/liuyuezuo/reconstruct-aifp/log/ariane133/run2',
#         # '/home/liuyuezuo/reconstruct-aifp/log/ariane133/run0',
#         # '/home/liuyuezuo/reconstruct-aifp/log/ispd15/mgc_des_perf_a/run0',
#         # '/home/liuyuezuo/reconstruct-aifp/log/ispd15/mgc_matrix_mult_a/run2', # heuristic-reward-bug-solving, no overlap punishment
#         # '/home/liuyuezuo/reconstruct-aifp/log/ispd15/mgc_matrix_mult_a/run1',
#         # '/home/liuyuezuo/reconstruct-aifp/log/ispd15/mgc_matrix_mult_a/run0',
#         # '/home/liuyuezuo/reconstruct-aifp/log/ispd15/mgc_edit_dist_a/run1',
#         #         '/home/liuyuezuo/reconstruct-aifp/log/ispd15/mgc_pci_bridge32_a/run0'
#     ]

#     for log_dir in log_dirs:
#         rl_placer_report(log_dir, report_file_path=None)