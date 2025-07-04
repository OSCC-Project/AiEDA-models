# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# AutoDMP base config
AUTODMP_BASE_CONFIG = {
    "aux_input": "",
    "lef_input": "",
    "def_input": "",
    "verilog_input": "",
    "gpu": 1,
    "gpu_id": 0,
    "num_bins_x": 512,
    "num_bins_y": 512,
    "global_place_stages": [
        {
            "num_bins_x": 512,
            "num_bins_y": 512,
            "iteration": 3000,
            "learning_rate": 0.01,
            "wirelength": "weighted_average",
            "optimizer": "adam",
            "Llambda_density_weight_iteration": 1,
            "Lsub_iteration": 1,
            "learning_rate_decay": 1.0
        }
    ],
    "target_density": 0.7,
    "density_weight": 8e-05,
    "random_seed": 1000,
    "result_dir": "",
    "scale_factor": 1.0,
    "ignore_net_weight": 10,
    "shift_factor": [0, 0],
    "ignore_net_degree": 100,
    "gp_noise_ratio": 0.025,
    "enable_fillers": 1,
    "global_place_flag": 1,
    "legalize_flag": 1,
    "detailed_place_flag": 0,
    "stop_overflow": 0.07,
    "dtype": "float32",
    "detailed_place_engine": "",
    "detailed_place_command": "",
    "plot_flag": 0,
    "RePlAce_ref_hpwl": 350000,
    "RePlAce_LOWER_PCOF": 0.94,
    "RePlAce_UPPER_PCOF": 1.05,
    "gamma": 0.1318231577,
    "RePlAce_skip_energy_flag": 0,
    "random_center_init_flag": 1,
    "init_loc_perc_x": 0.5,
    "init_loc_perc_y": 0.5,
    "sort_nets_by_degree": 0,
    "num_threads": 8,
    "dump_global_place_solution_flag": 0,
    "dump_legalize_solution_flag": 0,
    "routability_opt_flag": 0,
    "route_num_bins_x": 512,
    "route_num_bins_y": 512,
    "node_area_adjust_overflow": 0.15,
    "max_num_area_adjust": 3,
    "adjust_nctugr_area_flag": 0,
    "adjust_rudy_area_flag": 0,
    "adjust_pin_area_flag": 0,
    "area_adjust_stop_ratio": 0.01,
    "route_area_adjust_stop_ratio": 0.01,
    "pin_area_adjust_stop_ratio": 0.05,
    "unit_horizontal_capacity": 1.5625,
    "unit_vertical_capacity": 1.45,
    "unit_pin_capacity": 0.058,
    "max_route_opt_adjust_rate": 2.0,
    "route_opt_adjust_exponent": 2.0,
    "pin_stretch_ratio": 1.414213562,
    "max_pin_opt_adjust_rate": 1.5,
    "timing_opt_flag": 0,
    "macro_place_flag": 1,
    "use_bb": 0,
    "two_stage_density_scaler": 1000,
    "deterministic_flag": 1,
    "get_congestion_map": 1,
    "macro_halo_x": 0,
    "macro_halo_y": 0,
    "macro_overlap_flag": 0,
    "macro_overlap_weight": 8e-06,
    "macro_overlap_mult_weight": 1,
    "macro_padding_x": 0,
    "macro_padding_y": 0,
    "bndry_padding_x": 0,
    "bndry_padding_y": 0,
    "pin_density": -1,
    "route_info_input": "default",
    "evaluate_pl": 0,
    "risa_weights": 0,
    "base_design_name": "default",
    "macro_pin_halo_x": 20000,
    "macro_pin_halo_y": 20000,
    "with_sta": 0,
}



# Cost ratio for unfinished AutoDMP runs
AUTODMP_BAD_RATIO = 10


# Base PPA: HPWL, RSMT, Congestion, Density
AUTODMP_BASE_PPA = {
    "nvdla_asap7": {
        "hpwl": 1.37e9,
        "rsmt": 1.62e9,
        "congestion": 0.60,
        "density": 0.70,
    },
    "ariane_asap7": {
        "hpwl": 6.37e8,
        "rsmt": 7.55e8,
        "congestion": 0.44,
        "density": 0.70,
    },
    "nvdla_nangate45": {
        "hpwl": 1.31e10,
        "rsmt": 1.50e10,
        "congestion": 0.65,
        "density": 0.51,
    },
    "ariane_nangate45_51": {
        "hpwl": 5.37e9,
        "rsmt": 6.21e9,
        "congestion": 0.53,
        "density": 0.51,
    },
    "ariane_nangate45_68": {
        "hpwl": 5.77e9,
        "rsmt": 6.60e9,
        "congestion": 0.56,
        "density": 0.68,
    },
    "bp_quad_nangate45": {
        "hpwl": 3.44e10,
        "rsmt": 3.65e10,
        "congestion": 0.65,
        "density": 0.68,
    },
    "mempool_nangate45": {
        "hpwl": 1.80e11,
        "rsmt": 1.65e11,
        "congestion": 1.0,
        "density": 0.68,
    },
}


# Best found parameters
AUTODMP_BEST_CFG = {
    "nvdla_asap7": {
        "GP_num_bins_x": 2048,
        "GP_num_bins_y": 512,
        "GP_learning_rate": 0.0002593105404757665,
        "GP_wirelength": "logsumexp",
        "GP_optimizer": "adam",
        "GP_learning_rate_decay": 0.9988138905205957,
        "target_density": 0.7037918984939342,
        "density_weight": 1.0814963049463983e-06,
        "stop_overflow": 0.08204745228090016,
        "RePlAce_ref_hpwl": 364310,
        "RePlAce_LOWER_PCOF": 0.9036439681809721,
        "RePlAce_UPPER_PCOF": 1.0509971089013768,
        "gamma": 0.3745325302390664,
        "init_loc_perc_x": 0.4422497958531522,
        "init_loc_perc_y": 0.4898308592489072,
        "macro_halo_x": 530,
        "macro_halo_y": 782,
    },
    "ariane_asap7": {
        "GP_num_bins_x": 256,
        "GP_num_bins_y": 1024,
        "GP_learning_rate": 0.0006821746956516698,
        "GP_wirelength": "weighted_average",
        "GP_optimizer": "adam",
        "GP_learning_rate_decay": 0.9982062232878134,
        "target_density": 0.6671062288330134,
        "density_weight": 0.0001247338132066084,
        "stop_overflow": 0.07556185605214383,
        "RePlAce_ref_hpwl": 255243,
        "RePlAce_LOWER_PCOF": 0.9153849012327219,
        "RePlAce_UPPER_PCOF": 1.0297406601004784,
        "gamma": 0.4216322781678866,
        "init_loc_perc_x": 0.2356424321906761,
        "init_loc_perc_y": 0.5448366238030582,
        "macro_halo_x": 607,
        "macro_halo_y": 908,
    },
    "ariane_ng45": {
        "GP_num_bins_x": 512,
        "GP_num_bins_y": 256,
        "GP_learning_rate": 0.0009996179107878333,
        "GP_wirelength": "logsumexp",
        "GP_optimizer": "adam",
        "GP_learning_rate_decay": 0.9960132764492116,
        "target_density": 0.6223729538332924,
        "density_weight": 0.0026377670040185477,
        "stop_overflow": 0.0823271136490576,
        "RePlAce_ref_hpwl": 380241,
        "RePlAce_LOWER_PCOF": 0.9514407197563405,
        "RePlAce_UPPER_PCOF": 1.0634994799782473,
        "gamma": 0.4283953467808522,
        "init_loc_perc_x": 0.40594836892635444,
        "init_loc_perc_y": 0.3373454907950247,
        "macro_halo_x": 4900,
        "macro_halo_y": 5720,
    },
    "bp_quad_nangate45": {
        "GP_num_bins_x": 512,
        "GP_num_bins_y": 512,
        "GP_learning_rate": 0.00116654776941842,
        "GP_wirelength": "logsumexp",
        "GP_optimizer": "nesterov",
        "GP_learning_rate_decay": 0.9999340928988243,
        "target_density": 0.7928375667839145,
        "density_weight": 2.951615016177828e-05,
        "stop_overflow": 0.093771218764875,
        "RePlAce_ref_hpwl": 185749,
        "RePlAce_LOWER_PCOF": 0.9860513931331294,
        "RePlAce_UPPER_PCOF": 1.026930781106926,
        "gamma": 0.32571346109405486,
        "init_loc_perc_x": 0.4231496156400661,
        "init_loc_perc_y": 0.3036304257521259,
        "macro_halo_x": 4884,
        "macro_halo_y": 4143,
    },
}
