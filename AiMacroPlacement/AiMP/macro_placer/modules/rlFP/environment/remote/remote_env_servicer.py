from aifp.operation.macro_placer.rl_placer.environment.local_env import LocalEnv
import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2 as remote_env_pb2
import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2_grpc as remote_env_pb2_grpc


class RemoteEnvServicer(remote_env_pb2_grpc.RemoteEnvServicer):
    def __init__(
        self,
        case_select):

        self._base_env = LocalEnv(is_evaluate=False)

    def reset(self, episode:remote_env_pb2.Action, unused_context) -> remote_env_pb2.Ret:
        [np_obs, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask] = self._base_env.reset(episode.action)
        return remote_env_pb2.Ret(
            macro_idx_to_place = macro_idx_to_place.tobytes(),
            np_obs = np_obs.tobytes(),
            action_mask = action_mask.tobytes(),
            sparse_adj_i = sparse_adj_i.tobytes(),
            sparse_adj_j = sparse_adj_j.tobytes(),
            sparse_adj_weight = sparse_adj_weight.tobytes()
        )
    
    def step(self, action:remote_env_pb2.Action, unused_contest) -> remote_env_pb2.Ret:
        int_action = action.action
        [np_obs, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask], reward, done, info = self._base_env.step(int_action)
        return remote_env_pb2.Ret(
            macro_idx_to_place = macro_idx_to_place.tobytes(),
            np_obs = np_obs.tobytes(),
            action_mask = action_mask.tobytes(),
            sparse_adj_i = sparse_adj_i.tobytes(),
            sparse_adj_j = sparse_adj_j.tobytes(),
            sparse_adj_weight = sparse_adj_weight.tobytes(),
            reward = reward.tobytes(),
            done = done.tobytes(),
            info = info
        )
    
    def episode_length(self, unused_message:remote_env_pb2.Number, unused_contest) -> remote_env_pb2.Number:
        return remote_env_pb2.Number(num=self._base_env.episode_length)

    def macro_nums(self, unused_message:remote_env_pb2.Number, unused_contest) -> remote_env_pb2.Number:
        return remote_env_pb2.Number(num=self._base_env.macro_nums)
    
    def node_nums(self, unused_message:remote_env_pb2.Number, unused_contest) -> remote_env_pb2.Number:
        return remote_env_pb2.Number(num=self._base_env.node_nums)

    def edge_nums(self, unused_message:remote_env_pb2.Number, unused_contest) -> remote_env_pb2.Number:
        return remote_env_pb2.Number(num=self._base_env.edge_nums)

    def set_log_dir(self, run_port:remote_env_pb2.Number, unused_contest) -> remote_env_pb2.Number:
        self._base_env.set_log_dir(run_port.num)
        return remote_env_pb2.Response(flag=True)
