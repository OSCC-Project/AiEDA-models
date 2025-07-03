from tkinter import Y
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import time
import logging
import os
from collections import defaultdict

from aifp.report.report import rl_placer_report
from aifp.utility.singleton import singleton
from aifp.operation.macro_placer.rl_placer.environment.local_env import LocalEnv
# from aifp.operation.macro_placer.rl_placer.environment.parallel_env import ParallelEnv
from aifp.operation.macro_placer.rl_placer.environment.parallel_env_ray import ParallelEnv
from aifp.operation.macro_placer.rl_placer.agent.policy_based_agent import PolicyBasedAgent
from aifp.network.google_model import GoogleModel
from aifp.network.simple_model import SimpleModel
from aifp.database.data_structure.data_spec import DataSpec
from aifp.database.replay_buffer.rollout_storage import RolloutStorage
from aifp.solver.rl.ppo import PPO
from aifp import setting


@singleton
class RLPlacer:
    def __init__(self, placedb, rl_config):
        self._set_log_dir()
        self._start_time = time.time()
        self._num_threads = min(os.cpu_count() - 2, 30)
        torch.set_num_threads(self._num_threads)

    def run(self):
        # train & evaluate envs
        device = 'cpu'
        device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        train_env = ParallelEnv()
        if setting.rl_config['evaluate'] == True:
            evaluate_env = LocalEnv(is_evaluate=True)
            evaluate_env.set_log_dir('evaluate')

        print('parallel environment nums: {}'.format(train_env.env_nums))
        env_nums = train_env.env_nums
        episode_nums = setting.rl_config['episode_nums']

        # ppo config
        update_epochs = setting.solver['ppo']['update_epochs']
        learning_rate = setting.solver['ppo']['learning_rate']
        epsilon = setting.solver['ppo']['epsilon']
        value_pred_loss_coef = setting.solver['ppo']['value_pred_loss_coef']
        entropy_regularization = setting.solver['ppo']['entropy_regularization']
        importance_ratio_clipping = setting.solver['ppo']['importance_ratio_clipping']
        discount_factor = setting.solver['ppo']['discount_factor']
        gradient_clipping = setting.solver['ppo']['gradient_clipping']

        # RL agent
        # model = GoogleModel(
        #     grid_nums=setting.env_train['max_grid_nums'],
        #     node_nums=train_env.node_nums,
        #     episode_len = train_env.episode_length,
        #     origin_node_dim=len(setting.node_feature_config))
    
        model = SimpleModel(
            grid_nums=setting.env_train['max_grid_nums'],
            episode_len = train_env.episode_length,
            origin_node_dim=len(setting.node_feature_config) + train_env.episode_length,
            gcn_node_dim = 32
            )

        agent = PolicyBasedAgent(model)
        last_iteration = self._load_checkpoint(agent) if setting.rl_config['load_checkpoint'] else 0
        print('last iteration:  ', last_iteration)
        ppo = PPO(
                agent=agent,
                device=device,
                clip_param=importance_ratio_clipping,
                initial_lr=learning_rate,
                entropy_coef=entropy_regularization,
                value_loss_coef=value_pred_loss_coef,
                max_grad_norm=gradient_clipping,
                eps = epsilon)

        print('training_on {}'.format(ppo._agent._model._device))
        # rollout-storage data-specification
        dataspec = DataSpec()
        obs_space = train_env.observation_space
        dataspec.add_data_item('node_obs', obs_space.node_features.dtype, obs_space.node_features.shape)
        dataspec.add_data_item('macro_idx_to_place', obs_space.macro_idx_to_place.dtype, obs_space.macro_idx_to_place.shape)
        dataspec.add_data_item('sparse_adj_i', obs_space.sparse_adj_i.dtype, obs_space.sparse_adj_i.shape)
        dataspec.add_data_item('sparse_adj_j', obs_space.sparse_adj_j.dtype, obs_space.sparse_adj_j.shape)
        dataspec.add_data_item('sparse_adj_weight', obs_space.sparse_adj_weight.dtype, obs_space.sparse_adj_weight.shape)
        dataspec.add_data_item('action_mask', obs_space.action_mask.dtype, obs_space.action_mask.shape)

        dataspec.add_data_item('reward', obs_space.reward.dtype, obs_space.reward.shape)
        dataspec.add_data_item('done', obs_space.done.dtype, obs_space.done.shape)
        dataspec.add_data_item('action', np.float32, shape=(1, ))
        dataspec.add_data_item('logprob', np.float32, shape=(1, ))
        dataspec.add_data_item('value', np.float32, shape=(1, ))

        num_episodes_per_iteration = setting.solver['ppo']['num_episodes_per_iteration']
        assert num_episodes_per_iteration % env_nums == 0
        collect_nums = num_episodes_per_iteration // env_nums
        rollout = RolloutStorage(dataspec, env_nums, collect_nums * train_env.episode_length)

        print('num_epiosdes_per_iteration: {}, collect_nums: {}'.format(num_episodes_per_iteration,  collect_nums))

        # run episodes
        for iteration in range(last_iteration + 1, episode_nums):
            try:
                assert rollout._step == 0 # for debugging
            except:
                print(rollout._step)
                exit()
            start_time = time.time()
            
            total_heuristic_return = 0
            total_placement_return = 0
            iteration_info = []
            # total_episode_return = 0
            action_entropy_list = []

            # collect until num_episodes_per_update
            for collect_num in range(collect_nums):
                torch.set_num_threads(self._num_threads)
                obs_list = train_env.reset(iteration)
                step = 0
                while True:
                    tensor_obs_list = [torch.from_numpy(obs).to(device) for obs in obs_list]
                    value, action, logprob, action_entropy = ppo.get_agent().sample(*tensor_obs_list)
                    next_obs_list, reward, done, info = train_env.step(action)
                    # total_episode_return += np.mean(reward)
                    
                    action_entropy_list.append(np.mean(action_entropy))
                    
                    data_dict = {}
                    # observations to forward model
                    data_dict['node_obs'] = obs_list[0]
                    data_dict['macro_idx_to_place'] = obs_list[1]
                    data_dict['sparse_adj_i'] = obs_list[2]
                    data_dict['sparse_adj_j'] = obs_list[3]
                    data_dict['sparse_adj_weight'] = obs_list[4]
                    data_dict['action_mask'] = obs_list[5]
                    data_dict['reward'] = reward
                    data_dict['done'] = done

                    data_dict['action'] = np.expand_dims(action.astype(np.float32), axis=-1)
                    data_dict['logprob'] = np.expand_dims(logprob.astype(np.float32), axis=-1)
                    data_dict['value'] = np.expand_dims(value.astype(np.float32), axis=-1)
                    rollout.append(data_dict)

                    if done[0]:
                        total_placement_return += np.mean(reward)
                        iteration_info += info
                        print('collect_num {}, placement return: {}'.format(collect_num, np.mean(reward)))
                        break
                    else:
                        total_heuristic_return += np.mean(np.mean(reward))

                    obs_list, done = next_obs_list, done
                    step += 1
            print('collect_time: ', time.time() - start_time)
            # Bootstrap value if not done
            done = np.ones_like(done, dtype=np.float32)
            value = np.zeros_like(value, dtype=np.float32)
            rollout.compute_adv_and_return('reward', 'value', 'done',
                value=np.expand_dims(value, axis=-1),
                done=np.expand_dims(done, axis=-1),
                gamma=setting.solver['ppo']['gamma'],
                gae_lambda=setting.solver['ppo']['gae_lambda'])
            print('try to learn....')
            learn_start_time = time.time()
            # Optimizing the policy and value network
            torch.set_num_threads(self._num_threads)
            v_loss, pg_loss, entropy_loss = ppo.learn(rollout,
                minibatch_size=setting.solver['ppo']['minibatch_size'],
                batch_size=train_env.episode_length*env_nums,
                update_epochs=update_epochs)
            print('learn_time: ', time.time() - learn_start_time)

            # tensorboard
            end_time = time.time()
            self._writer.add_scalar('train/v_loss', v_loss, iteration)
            self._writer.add_scalar('train/pg_loss', pg_loss, iteration)
            self._writer.add_scalar('train/entropy_loss',  entropy_loss, iteration)
            self._writer.add_scalar('train/action_entropy', np.mean(action_entropy_list), iteration)
            self._writer.add_scalar('train/train_placement_return', total_placement_return / (collect_nums), iteration)
            self._writer.add_scalar('train/train_heuristic_return', total_heuristic_return / (collect_nums), iteration)
            self._writer.add_scalar('train/train_episode_return', (total_placement_return + total_heuristic_return) / (collect_nums), iteration)
            for score_key, score_value in self._extract_score_from_info(iteration_info).items():
                self._writer.add_scalar('train/{}'.format(score_key), score_value, iteration)


            # evaluate
            if setting.rl_config['evaluate'] == True and iteration % setting.rl_config['evaluate_interval'] == 0:
                print('start to evaluate: ')
                evaluate_return, score_dict = self._evaluate(evaluate_env, ppo.get_agent(), device, iteration, setting.rl_config['evaluate_nums'])
                self._writer.add_scalar('evaluate/evaluate_return', evaluate_return, iteration)
                for score_key, score_value in score_dict.items():
                    self._writer.add_scalar('evaluate/' + score_key, score_value, iteration)

            # save checkpoint
            if iteration % setting.rl_config['save_interval'] == 0:
                print('save_check point: ', iteration)
                self._save_checkpoint(ppo.get_agent(), iteration)

            # log
            print('episode {}, placement-return: {:.5f}, time: {:.2f} secs'.format(iteration, total_placement_return / (collect_nums), end_time-start_time))
            print('episode {}, heuristic-return: {:.5f}, time: {:.2f} secs'.format(iteration, total_heuristic_return / (collect_nums), end_time-start_time))
            print('episode {}, train-return: {:.5f}, time: {:.2f} secs'.format(iteration, (total_placement_return + total_heuristic_return) / (collect_nums), end_time-start_time))
            self._rl_placer_logger.info('time: {} secs'.format(time.time() - self._start_time))
            self._rl_placer_logger.info('episode {}, placement-return: {:.5f}, time: {:.2f} secs'.format(iteration, total_placement_return / (collect_nums), end_time-start_time))
            self._rl_placer_logger.info('episode {}, heuristic-return: {:.5f}, time: {:.2f} secs'.format(iteration, total_heuristic_return / (collect_nums), end_time-start_time))
            self._rl_placer_logger.info('episode {}, train-return: {:.5f}, time: {:.2f} secs'.format(iteration, (total_placement_return + total_heuristic_return) / (collect_nums), end_time-start_time))
        rl_placer_report(log_dir=self._log_dir)


    def _extract_score_from_info(self, info):
        print('rl_placer info: ', info[0])
        episode_num = len(info)
        score_dict = defaultdict(float)
        for episode_info in info:
            lines = episode_info.split('\n')
            for line in lines:
                line = line.split(':')
                score_key = line[0]
                score_value = line[1]
                if score_key in ['wirelength', 'overflow']:
                    score_dict[score_key] += float(score_value)
        # calculate average score
        for key in score_dict.keys():
            score_dict[key] /= episode_num
        return score_dict

    def _set_log_dir(self):
        project_dir = os.environ['AIFP_PATH']
        log_dir = setting.log['log_dir']
        model_dir = setting.log['model_dir']
        case_select = setting.case_select
        run_num = setting.log['run_num']
        
        log_dir = '{}/{}/{}/run{}'.format(project_dir, log_dir, case_select, run_num)
        model_dir = '{}/{}/{}/run{}'.format(project_dir, model_dir, case_select, run_num)
        self._log_dir = log_dir
        self._model_dir = model_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self._rl_placer_logger = logging.getLogger()
        self._rl_placer_logger.setLevel(logging.INFO)
        self._rl_placer_logger.addHandler(logging.FileHandler(log_dir + '/RLPlacer.log'))
        self._rl_placer_logger.root.name = 'aifp:RLPlacer'
        self._writer = SummaryWriter(log_dir=log_dir)
        self._log_dir = log_dir


    def _evaluate(self, env, agent, device, iteration, evaluate_num=1):
        return_list = []
        for i in range(evaluate_num):
            episode_return = 0.0
            obs_list = env.reset(iteration)
            while True:
                tensor_obs_list = [torch.from_numpy(obs).unsqueeze(0).to(device) for obs in obs_list]
                action = agent.predict(*tensor_obs_list).squeeze()
                next_obs_list, reward, done, info = env.step(action)
                episode_return += reward
                if done:
                    score_dict = env.get_score_dict()
                    return_list.append(episode_return)
                    break

                obs_list, done = next_obs_list, done

        average_return = np.mean(return_list)
        return average_return, score_dict
    
    def _load_checkpoint(self, agent):
        # check exist runs
        if os.path.exists(self._model_dir + '/model.pt'):
            last_iteration = agent.load_model(self._model_dir + '/model.pt')
            return last_iteration
        else:
            return -1
    
    def _save_checkpoint(self, agent, last_iteration):
        agent.save_model(self._model_dir + '/model.pt', last_iteration)