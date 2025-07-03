import os
import sys

import time
import signal
import subprocess
import time
import pickle
import fabric
import psutil

os.environ["AIFP_PATH"] = '/root/reconstruct-aifp/'
sys.path.append('..')
sys.path.append(os.environ['AIFP_PATH'])
from aifp import setting
from aifp.utility.network import get_hostname, connect_and_get_response, connect_and_send, get_pid

class EnvClusterController:
    def __init__(self):
        pass

    def start_cluster(self, host:str, port:int, env_nums:int):
        flag = connect_and_send(host=host, port=port, request='start_servers {}'.format(env_nums).encode(), time_out=5)
        if flag == False:
            raise RuntimeError('fail to start remote cluster...')

    def stop_cluster(self, host:str, port:int):
        flag = connect_and_send(host=host, port=port, request='stop_servers'.encode(), time_out=5)
        if flag == False:
            raise RuntimeError('fail to start remote cluster...')

    def get_env_ports(self, host:str, port:int, time_out=300):
        """waiting until cluster started and get env ports or until time_out"""
        start_time = time.time()
        while time.time() - start_time < time_out:
            cluster_status = connect_and_get_response(host=host, port=port, request=b'get_status', time_out=5)
            if cluster_status.decode() == 'started':
                env_server_ports = pickle.loads(connect_and_get_response(host=host, port=port, request='get_env_server_ports'.encode()))
                return env_server_ports
            else:
                time.sleep(1)

    def start_clusters(self)->list:
        """start all remote env-server-clusters, return env_socket_list"""
        self._env_socket_list = []
        for hostname, config in setting.env_server.items():
            host = 'localhost' if hostname == get_hostname() else config['ip_address']
            self.start_cluster(host=host, port=config['controller_port'], env_nums=config['env_nums'])

        for hostname, config in setting.env_server.items():
            host = 'localhost' if hostname == get_hostname() else config['ip_address']
            env_ports = self.get_env_ports(host=host, port=config['controller_port'], time_out=200)
            for env_port in env_ports:
                self._env_socket_list.append('{}:{}'.format(host, env_port))
        return self._env_socket_list
    
    def stop_clusters(self):
        for hostname, config in setting.env_server.items():
            host = 'localhost' if hostname == get_hostname() else config['ip_address']
            self.stop_cluster(host=host, port=config['controller_port'])
        print('all clusters stopped')

    def get_env_sockets(self)->list:
        """get env server ports from all env clusters"""
        return self._env_socket_list

    def __del__(self):
        # print('env_cluster_del_call')
        self.stop_clusters()

if __name__ == '__main__':
    env_cluster_controller = EnvClusterController()
    env_server_ports = env_cluster_controller.start_clusters()
    print('env_server_controller clusters started')
    time.sleep(100)
    # env_cluster_controller.stop()