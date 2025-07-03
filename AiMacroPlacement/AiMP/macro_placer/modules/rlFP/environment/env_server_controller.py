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

from aifp.utility.network import get_hostname, get_free_ports, is_port_open, listen_and_response
from aifp.operation.macro_placer.rl_placer.environment.remote.run_env_server import start_server

class EnvServerController:
    def __init__(self):
        pass
    
    def start(self, env_num, not_use):
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
        print('============ env servers on {} terminated ============'.format(get_hostname()))

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

if __name__ == '__main__':
    def signal_handler(signum, frame):
        """when receiving terminate signal, clean up ( stop all environment servers) and exit program"""
        # print('entering env_server_controller signal_handler...')
        vars = globals()
        if 'env_server_controller' in vars:
            env_server_controller.stop()
        # if 'response_thread' in vars:
        #     response_thread.stop()
        #     response_thread.join()
        exit()

    signal.signal(signal.SIGTERM, signal_handler)


    
    env_num = int(sys.argv[1])
    controller_port = int(sys.argv[2])
    env_server_controller = EnvServerController()
    env_server_ports = env_server_controller.start(env_num, not_use=controller_port)
    response_thread = threading.Thread(target=listen_and_response, args=(controller_port, b"get_env_server_ports", pickle.dumps(env_server_ports)))
    response_thread.start()
    while True:  # waiting for signal.SIGUSR1
        time.sleep(1)

# class EnvServerController:
#     def __init__(self):
#         pass

#     def start(self, env_num):
#         "start env_num env servers, return server port list"
#         print('==============  starting remote env servers at {} =============='.format(get_hostname()))
#         self._server_ports = get_free_ports(num_ports=env_num)
#         self._env_server_process_list = []
#         # multiprocessing.set_start_method('spawn') 
#         for port in self._server_ports:
            
#             server_process = multiprocessing.Process(target=start_server, args=(port, 3), daemon=False) # a dameon process can't create subprocess
#             self._env_server_process_list.append(server_process)
#             server_process.start()
#         # return when all servers started
#         while True:
#             if self._is_all_servers_started(self._server_ports):
#                 break
#             else:
#                 time.sleep(5)
#         time.sleep(10)
#         print('all servers started')
#         return self._server_ports

#     def stop(self):
#         for process in self._env_server_process_list:
#             if process.is_alive():
#                 process.terminate(）
#                 process.wait()
#         print('remote env servers terminated'))

#     def _is_all_servers_started(self, server_ports):
#         for port in server_ports:
#             if not is_port_open(host='localhost', port=port):
#                 return False
#         return True

#     def __del__(self):
#         print(' ============ __del__ in env_server_controller called ===========')
#         self.stop()