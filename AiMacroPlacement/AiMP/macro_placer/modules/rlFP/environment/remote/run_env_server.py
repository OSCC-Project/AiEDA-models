import sys
import os
os.environ["AIFP_PATH"] = '/root/reconstruct-aifp/'
sys.path.append(os.environ['AIFP_PATH'])
from concurrent import futures
# import grpc
import time

from aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_servicer import RemoteEnvServicer
import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2 as remote_env_pb2
import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2_grpc as remote_env_pb2_grpc
from aifp import setting
import signal

def start_server(port, max_workers=3):
    def signal_handler(signum, frame):
        """when receiving terminate signal, clean up"""
        exit() # run __del__ methods to clean up and exit
    signal.signal(signal.SIGTERM, signal_handler)


    case_select = setting.case_select
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                options=[('grpc.max_send_message_length', setting.env_train['grpc_max_message_length']),
                ('grpc.max_receive_message_length', setting.env_train['grpc_max_message_length'])])

    remote_env_pb2_grpc.add_RemoteEnvServicer_to_server(
        RemoteEnvServicer(case_select), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()

    time.sleep(1)
    with grpc.insecure_channel('localhost:{}'.format(port)) as channel:
        local_stub = remote_env_pb2_grpc.RemoteEnvStub(channel)
        local_stub.set_log_dir(remote_env_pb2.Number(num=port))

    print("Server started, listening on {}".format(port))
    server.wait_for_termination()

if __name__ == '__main__':
    port = int(sys.argv[1])
    start_server(port, 1)