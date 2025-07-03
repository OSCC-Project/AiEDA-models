import logging
from aimp.aifp.database.data_structure import fp_solution

def extract_fp_solution_from_report(report_file):
    solution = fp_solution.FPSolution()
    f = open(report_file, 'r')
    lines = f.readlines()
    for line in lines:
        if (line.startswith('name')):
            line = line.split(' ')
            name = line[0].split(':')[-1]
            low_x = float(line[1].split(':')[-1])
            low_y = float(line[2].split(':')[-1])
            orient = ''
            if (len(line) > 3):
                orient = float(line[3].split(':')[-1])
                print('orient: ', orient)
            solution.add_macro_info(
                fp_solution.MacroInfo(
                    name,
                    low_x,
                    low_y,
                    orient
                )
            )
    return solution


def extract_macro_info_from_report(report_file):
    # try:
    #     f = open(report_file, 'r')
    # except:
    #     logging.info('file {} not exist !'.format(report_file))

    f = open(report_file, 'r')
    macro_info = []
    lines = f.readlines()
    for line in lines:
        if (line.startswith('name')):
            line = line.split(' ')
            name = line[0].split(':')[-1]
            low_x = float(line[1].split(':')[-1])
            low_y = float(line[2].split(':')[-1])
            orient = ''
            if (len(line) > 3):
                orient = float(line[3].split(':')[-1])
            macro_info.append(
                {'name': name,
                 'low_x': low_x,
                 'low_y': low_y,
                 'orient': orient})

    return macro_info