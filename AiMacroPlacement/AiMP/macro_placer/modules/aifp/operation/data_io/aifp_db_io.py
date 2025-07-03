from aimp.aifp.database.data_structure.core import PyCore
from aimp.aifp.database.data_structure.instance import PyInstance
from aimp.aifp.database.data_structure.blockage import PyBlockage
from aimp.aifp.database.data_structure.pin import PyPin
from aimp.macroPlaceDB import MacroPlaceDB
from aimp.aifp.database.rl_env_db.rl_env_db import RLEnvDB

# def get_graph_setting():
#     graph_setting = aifp_cpp.GraphSetting()
#     if (setting.case_select.startswith('ispd15') or setting.case_select=='ariane133'):
#         has_io_cell = False
#     else:
#         has_io_cell = True
#     graph_setting.set_has_io_instance(has_io_cell)
#     graph_setting.set_graph_type(setting.graph['graph_type'])
#     graph_setting.set_use_net_weight(setting.graph['use_net_weight'])
#     graph_setting.set_max_fanout(setting.graph['max_fanout'])
#     graph_setting.set_io_slice_num(setting.graph['io_slice_num'])
#     graph_setting.set_cluster_inst_density(setting.graph['cluster_inst_density'])
#     # graph_setting.set_num_rows(setting.evaluator['clustered_dreamplace']['num_rows'])
#     # graph_setting.set_num_rows(setting.evaluator['clustered_dreamplace']['num_columns'])
#     graph_setting.set_partition_tool(setting.graph['partition_tool'])
    
#     if setting.graph['partition_tool'] == 'metis':
#         graph_setting.set_metis_parts(setting.metis['nparts'])
#         graph_setting.set_metis_ufactor(setting.metis['ufactor'])
#         graph_setting.set_metis_ncon(setting.metis['ncon'])
#     elif setting.graph['partition_tool'] == 'hmetis':
#         graph_setting.set_hmetis_exe(setting.hmetis['hmetis_exe'])
#         graph_setting.set_hmetis_nparts(setting.hmetis['nparts'])
#         graph_setting.set_hmetis_ufactor(setting.hmetis['ufactor'])
#         graph_setting.set_hmetis_nruns(setting.hmetis['nruns'])
#         graph_setting.set_hmetis_dbglvl(setting.hmetis['dbglvl'])
#         graph_setting.set_hmetis_seed(setting.hmetis['seed'])
#         graph_setting.set_hmetis_reconst(setting.hmetis['reconst'])
#         graph_setting.set_hmetis_ptype(setting.hmetis['ptype'])
#         graph_setting.set_hmetis_ctype(setting.hmetis['ctype'])
#         graph_setting.set_hmetis_rtype(setting.hmetis['rtype'])
#         graph_setting.set_hmetis_otype(setting.hmetis['otype'])
#     print('set graph_setting ok')
#     return graph_setting

# def init_irefactor_idb(aifp_db:aifp_cpp.AifpDB, irefactor_db_config_path:str, graph_setting:aifp_cpp.GraphSetting):
#     config_f = open(irefactor_db_config_path, 'r')
#     data = json.load(config_f)
#     config_f.close()
#     input = data['INPUT']
#     output = data['OUTPUT']
#     aifp_db.init_techlef(input['tech_lef_path'])
#     aifp_db.init_lef(input['lef_paths'])
#     aifp_db.init_def(input['def_path'])
#     aifp_db.init_idb()
#     aifp_db.buildGraph(graph_setting)
#     print('irefactor aifp_db init ok')

# def read_from_aifp_db(aifp_db:aifp_cpp.AifpDB):
#     core = aifp_db.get_core()
#     blockage_list = aifp_db.get_blockage_list() if setting.env_train['consider_blockage'] == True else []
#     inst_list = aifp_db.get_node_list()
#     edge_list = aifp_db.get_edge_list()
#     net_list = aifp_db.get_net_list()
#     net_weight = aifp_db.get_net_weight()

#     return {
#             'core': core,
#             'inst_list': inst_list,
#             'edge_list': edge_list,
#             'net_list': net_list,
#             'net_weight': net_weight,
#             'blockage_list': blockage_list}

# def read_from_aifp_db_and_destroy(idb_config_path:str):
#     """create new process to dealing with data-io, then destroy it to save memory, return a data-dict"""
#     q = multiprocessing.Manager().Queue()
#     io_process = multiprocessing.Process(target=_read_from_aifp_db_to_queue, daemon=True, args=(idb_config_path, q))
#     io_process.start()
#     io_process.join()
#     design_data_dict = q.get()
#     return design_data_dict

# def _read_from_aifp_db_to_queue(idb_config_path:str, q):
#     aifp_db = aifp_cpp.AifpDB()
#     graph_setting = get_graph_setting()
#     init_irefactor_idb(aifp_db, idb_config_path, graph_setting)
#     data_dict = read_from_aifp_db(aifp_db)
#     data_dict['core'] = PyCore(data_dict['core'])
#     data_dict['blockage_list'] = [PyBlockage(pyc_blockage) for pyc_blockage in data_dict['blockage_list']]
#     data_dict['inst_list'] = [PyInstance(pyc_inst) for pyc_inst in data_dict['inst_list']]
#     data_dict['net_list'] = [[PyPin(pyc_pin) for pyc_pin in net] for net in data_dict['net_list']]
#     q.put(data_dict)


def read_from_numpy_data(
    xl,
    yl,
    width,
    height,
    num_nodes,
    num_movable,
    node_x,
    node_y,
    node_size_x,
    node_size_y,
    pin2node_map,
    net2pin_map,
    pin_offset_x,
    pin_offset_y,
    node_names,
):
    core = PyCore()
    core.set_low_x(xl)
    core.set_low_y(yl)
    core.set_width(width)
    core.set_height(height)

    num_nodes = num_nodes
    # num_movable = num_movable_nodes
    # num_macros = num_movable_nodes
    print('ok1')
    inst_list = []
    for i in range(num_nodes):
        node = PyInstance()
        node.set_low_x(node_x[i])
        node.set_low_y(node_y[i])
        node.set_width(node_size_x[i])
        node.set_height(node_size_y[i])
        node.set_orient('N')
        node.set_origin_orient('N')
        node.set_index(i)
        node.set_name(node_names[i])

        if (i < num_movable): # assume first 100 nodes are macros
            if i < 100:
                node.set_type("macro")
                node.set_status("fixed")
            else:
                node.set_type("stdcell")
                node.set_status('unfixed')
        else:
            node.set_type("io_cluster")
            node.set_status("fixed")
        inst_list.append(node)
    net_list = []
    pin2node_map = pin2node_map
    for net in net2pin_map:
        pins = []
        for pin_id in net:
            pin = PyPin()
            node_id = pin2node_map[pin_id]
            pin.set_node_index(node_id)
            pin.set_offset_x(pin_offset_x[pin_id])
            pin.set_offset_y(pin_offset_y[pin_id])
            pin.set_name("pin-{}".format(pin_id))
            pins.append(pin)
        net_list.append(pins)
    edge_list = [[0, 1, 1]]
    data_dict = {}
    data_dict['core'] = core
    data_dict['blockage_list'] = []
    data_dict['inst_list'] = inst_list
    data_dict['net_list'] = net_list
    data_dict['net_weight'] = [1 for i in range(len(net_list))]
    data_dict['edge_list'] = edge_list
    return data_dict
    

def read_from_imp_db(mp_db: MacroPlaceDB):
    return read_from_numpy_data(
        mp_db.xl,
        mp_db.yl,
        mp_db.width,
        mp_db.height,
        mp_db.node_x.shape[0],
        mp_db.num_movable_nodes,
        mp_db.node_x,
        mp_db.node_y,
        mp_db.node_size_x,
        mp_db.node_size_y,
        mp_db.pin2node_map,
        mp_db.net2pin_map,
        mp_db.pin_offset_x,
        mp_db.pin_offset_y,
        mp_db.node_names
    )


# def read_edge_list(edge_1_path, edge_2_path):
#     f_edge_i = open(edge_1_path, 'r')
#     f_edge_j = open(edge_2_path, 'r')
#     edge_i_str = re.findall(r"-?\d+\.?\d*", f_edge_i.readlines()[0])
#     edge_j_str = re.findall(r"-?\d+\.?\d*", f_edge_j.readlines()[0])
#     assert len(edge_i_str) == len(edge_j_str)
#     edge_list = []
#     for i in range(len(edge_i_str)):
#         edge_list.append([int(edge_i_str[i]), int(edge_j_str[i]), 1]) # edge weight 1
#     return edge_list


# def read_adj(case_select):
#     adj_i_j_path = os.environ["AIFP_PATH"] + 'input/' + case_select + '/edges_1.dat'
#     adj_j_i_path = os.environ["AIFP_PATH"] + 'input/' + case_select + '/edges_2.dat'

#     adj_i_j_f = open(adj_i_j_path, 'r')
#     adj_i_j = adj_i_j_f.readlines()[0]
#     adj_i_j = re.findall(r"-?\d+\.?\d*", adj_i_j)
#     adj_i_j_f.close()

#     adj_j_i_f = open(adj_j_i_path, 'r')
#     adj_j_i = adj_j_i_f.readlines()[0]
#     adj_j_i = re.findall(r"-?\d+\.?\d*", adj_j_i)
#     adj_j_i_f.close()

#     assert len(adj_i_j) == len(adj_j_i)
#     for i in range(len(adj_i_j)):
#         adj_i_j[i] = int(adj_i_j[i])
#         adj_j_i[i] = int(adj_j_i[i])
    
#     return np.array(adj_i_j, dtype=np.int64), np.array(adj_j_i, dtype=np.int64), np.ones((len(adj_i_j),), dtype=np.float32)


# def read_adj(dense_adj_path:str, sparse=False):
#     if not dense_adj_path.endswith('.csv'):
#         raise ValueError('file not ends with .csv')
#     # pd_frames = pd.read_csv(dense_adj_path)
#     dense_adj = pd.read_csv(dense_adj_path).values

#     adj = dense_adj.astype(np.int32).copy() if not sparse else []
#     # print('adj sum: ', dense_adj.sum())
#     # print(dense_adj.shape)
#     for i in range(dense_adj.shape[0]):
#         for j in range(dense_adj.shape[1]):
#             if dense_adj[i][j] == 0:
#                 if not sparse:
#                     adj[i][j] = 0
#                     adj[j][i] = 0
#             else:
#                 if not sparse:
#                     adj[i][j] = 1
#                     adj[j][i] = 1
#                 else:
#                     adj.append([i, j, dense_adj[i][j]])
#     if not sparse:
#         return adj
#     else:
#         return np.array(adj, dtype=np.int32)