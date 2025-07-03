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
from aifp.utility.network import get_hostname, connect_and_get_response, get_pid

class EnvClusterController:
    def __init__(self):
        pass
    
    def start_remote_cluster(self, host:str, username:str, password:str, project_dir:str, controller_port:int, env_nums:int, env:set, cluster_name:str):
        conn = fabric.Connection(host=host, user=username, connect_kwargs={'password': password})
        start_cmd = 'python {}/aifp/operation/macro_placer/rl_placer/environment/env_server_controller.py {} {} {}&'\
            .format(project_dir, env_nums, controller_port, cluster_name)
        result = conn.run(start_cmd, env=env)
        conn.close()
        print('cluster {}:{} startted'.format(host, cluster_name))
    
    def stop_remote_cluster(self, host:str, username:str, password:str, cluster_name:str):
        conn = fabric.Connection(host=host, user=username, connect_kwargs={'password': password})
        # stop_cmd = 'pkill -f \".*{}\"'.format(cluster_name)
        pid = self.find_pid(cluster_name)
        if pid == None:
            print('pid not found...')
        else:
            stop_cmd = 'kill {} -SIGTERM'.format(pid)
            conn.run(stop_cmd)
            print('pid found and killed')
        conn.close()
        print('cluster {}:{} stopped'.format(host, cluster_name))
    
    def start_local_cluster(self, project_dir:str, env_nums:int, controller_port:int, cluster_name:str):
        print('starting env cluster, env_nums: {}'.format(env_nums))
        start_cmd = 'python {}/aifp/operation/macro_placer/rl_placer/environment/env_server_controller.py {} {} {}&'\
            .format(project_dir, env_nums, controller_port, cluster_name)
        subprocess.Popen(start_cmd, shell=True, preexec_fn=os.setsid)
        print('cluster {}:{} startted'.format('localhost', cluster_name))
        time.sleep(1)
        cluster_pid = get_pid(cluster_name)
        return cluster_pid
    
    def stop_local_cluster(self, cluster_pid:int): #cluster_name:str):
        os.kill(cluster_pid, signal.SIGTERM)
        print('cluster pid {} found and killed'.format(cluster_pid))
        print('cluster {}:{} stopped'.format('localhost', cluster_pid))

    def start_clusters(self)->list:
        self._cluster_names = []
        self._cluster_hosts = []
        self._cluster_controller_ports = []
        self._cluster_pids = []
        for hostname, config in setting.env_server.items():
            if hostname != get_hostname(): # start remote clusters
                raise NotImplementedError('start remote cluster not implemented...')
                # self._cluster_hosts.append(config['ip_address'])
                # cluster_name = 'remote_cluster_{}'.format(time.time()) # give it a unique name
                # self.start_remote_cluster(
                #     host=config['ip_address'],
                #     username=config['username'],
                #     password=config['password'],
                #     project_dir=config['project_dir'],
                #     controller_port=config['controller_port'],
                #     env_nums=config['env_nums'],
                #     env=config['env'],
                #     cluster_name=cluster_name)
            else: # start local cluster
                cluster_name = 'local_cluster_{}'.format(time.time())
                self._cluster_hosts.append('localhost')
                cluster_pid = self.start_local_cluster(
                    project_dir=config['project_dir'],
                    env_nums=config['env_nums'],
                    controller_port=config['controller_port'],
                    cluster_name=cluster_name
                )
            self._cluster_pids.append(cluster_pid)
            self._cluster_names.append(cluster_name)
            self._cluster_controller_ports.append(config['controller_port'])

        self._env_socket_list = self.get_env_sockets()
        print('cluster_names: ', self._cluster_names)
        print('cluster_ports: ', self._cluster_hosts)
        print('cluster_controller_ports: ', self._cluster_controller_ports)
        print('env_sockets: ', self._env_socket_list)
        return self._env_socket_list
    
    def stop_clusters(self):
        for idx in range(len(self._cluster_pids)):
            if self._cluster_names[idx].startswith('local_cluster'):
                self.stop_local_cluster(self._cluster_pids[idx])
            else:
                self.stop_remote_cluster(self._cluster_pids[idx])
        print('all clusters stopped')

    def get_env_sockets(self)->list:
        """get env server ports from all env clusters"""
        env_socket_list = []
        for i in range(len(self._cluster_names)):
            server_ports = pickle.loads(connect_and_get_response(
                host=self._cluster_hosts[i],
                port=self._cluster_controller_ports[i],
                request=b"get_env_server_ports",
                time_out=4000))
            for port in server_ports:
                env_socket_list.append('{}:{}'.format(self._cluster_hosts[i], port))
        # print('env_socket_list: ', env_socket_list)
        return env_socket_list

    def __del__(self):
        print('env_cluster_del_call')
        self.stop_clusters()

if __name__ == '__main__':
    env_cluster_controller = EnvClusterController()
    env_cluster_controller.start_clusters()
    print('env_server_controller clusters started')
    time.sleep(100)
    # env_cluster_controller.stop()