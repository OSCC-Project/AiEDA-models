import os
import sys

import time
import signal
import subprocess
import time
import socket
import pickle
import logging
import signal
import threading

os.environ["AIFP_PATH"] = '/root/reconstruct-aifp/'
sys.path.append('..')
sys.path.append(os.environ['AIFP_PATH'])
from aifp import setting

from aifp.utility.network import get_hostname, get_free_ports, is_port_open, listen_and_response
from aifp.operation.macro_placer.rl_placer.environment.remote.run_env_server import start_server

class EnvServerController:
    def __init__(self):
        self._started = False
    
    def start(self, env_num, not_use=None):
        print('==============  starting remote env servers at {} =============='.format(get_hostname()))
        logging.info('==============  starting remote env servers at {} =============='.format(get_hostname()))
        self._env_server_ports = get_free_ports(num_ports=env_num, not_use=not_use)
        logging.info('get free ports')
        print('get free ports ', self._env_server_ports)
        self._env_server_process_list = []
        for port in self._env_server_ports:
            start_cmd = 'python {}/aifp/operation/macro_placer/rl_placer/environment/remote/run_env_server.py {} &'.format(os.environ['AIFP_PATH'], port)
            logging.info(start_cmd)
            self._env_server_process_list.append(subprocess.Popen(start_cmd, shell=True, preexec_fn=os.setsid))
        # return when all servers started
        logging.info('before wait_until_all_servers_started')
        self._wait_until_all_servers_started()
        logging.info('after wait_until_all_servers_started')
        # time.sleep(5)
        print('====== all servers started =======')
        self._started = True
        return self._env_server_ports

    def stop(self):
        for process in self._env_server_process_list:
            # if process.is_alive():
            try:
                process.terminate()
                process.wait()
                os.killpg(process.pid, signal.SIGTERM) # kill relavant process group
            except:
                # print('process {} has terminated'.format(process))
                pass
        self._started = False
        print('============ env servers on {} terminated ============'.format(get_hostname()))

    def get_env_ports(self)->list:
        if self._started:
            return self._env_server_ports
        else:
            return None

    def _wait_until_all_servers_started(self):
        while True:
            if self._is_all_servers_started(self._env_server_ports):
                return
            else:
                time.sleep(5)

    def _is_all_servers_started(self, server_ports):
        for port in server_ports:
            if not is_port_open(host='localhost', port=port):
                return False
        return True

    def __del__(self):
        # logging.info('entering env server controller destructor')
        print('entering env server controller destructor')
        self.stop()
    

# def listen_and_response(listen_port:str, valid_request:bytes):
#     """listen and response on given port if request-message == valid_requset. WARNNING: this function will never return"""
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind(('localhost', listen_port))
#     server_socket.listen(1)
#     while True:
#     # try:
#         client_socket, _ = server_socket.accept()
#         request = client_socket.recv(1024)

#         if request.decode() == 'get_env_server_ports':
#             if env_server_controller._started:
#                 client_socket.sendall(pickle.dumps(env_server_controller.get_env_ports()))

#         elif request.decode().startswith('start_servers'):  # format  'start_servers env_nums'
#             if env_server_controller != None:
#                 env_server_controller.stop()
#             env_server_controller.start(env_nums=int(request.decode().split(' ')[1]))
        
#         elif request.decode() == 'stop_servers':
#             if env_server_controller != None:
#                 env_server_controller.stop()
#             env_server_controller.start()

#         elif request.decode() == 'get_status':
#             if env_server_controller._started:
#                 client_socket.sendall(b'started')
#             else:
#                 client_socket.sendall(b'closed')



if __name__ == '__main__':
    def signal_handler(signum, frame):
        """when receiving terminate signal, clean up ( stop all environment servers) and exit program"""
        vars = globals()
        if 'env_server_controller' in vars:
            env_server_controller.stop()
        exit()

    signal.signal(signal.SIGTERM, signal_handler)

    for hostname, config in setting.env_server.items():
        if hostname == get_hostname():
            env_num = config['env_nums']
            controller_port = config['controller_port']

    env_server_controller = EnvServerController()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', controller_port))
    server_socket.listen(1)
    print('env_servers started....')

    while True:
        print('waiting for connection...')
        client_socket, _ = server_socket.accept()
        print('got connection...')
        request = client_socket.recv(1024)
        print('got request...')

        if request.decode() == 'get_status':
            if env_server_controller._started:
                client_socket.sendall(b'started')
            else:
                client_socket.sendall(b'closed')
            print('send status...')

        elif request.decode() == 'get_env_server_ports':
            if env_server_controller._started:
                client_socket.sendall(pickle.dumps(env_server_controller.get_env_ports()))
            print('send env server ports...')

        elif request.decode().startswith('start_servers'):  # format  'start_servers env_nums'
            if env_server_controller._started:
                env_server_controller.stop()
            env_server_controller.start(env_num=int(request.decode().split(' ')[1]))
            print('servers started...')

        elif request.decode() == 'stop_servers':
            if env_server_controller._started:
                env_server_controller.stop()
            print('servers stopped...')

        else:
            continue