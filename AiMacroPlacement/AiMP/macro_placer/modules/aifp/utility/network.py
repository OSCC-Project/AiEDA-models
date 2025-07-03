from pickle import TRUE
import socket
import telnetlib
import time
import psutil
from typing import List

def get_hostname()->str:
    return socket.gethostname()

def get_free_ports(num_ports:int, not_use:int=None)->List[int]:
    """return num_ports number of free ports except not_use port-number"""
    ports:List[int] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        while len(ports) < num_ports:
            s.listen(1)
            port = s.getsockname()[1]
            if port != not_use:
                ports.append(port)
            s.close()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
    return ports

def is_port_open(host:str, port:int)->bool:
    """check if host:port is opening"""
    try:
        telnet = telnetlib.Telnet(host=host, port=port, timeout=0.1)
        telnet.close()
        return True
    except:
        return False

def listen_and_response(listen_port:str, valid_request:bytes, response:bytes):
    """listen and response on given port if request-message == valid_requset. WARNNING: this function will never return"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', listen_port))
    server_socket.listen(1)
    while True:
    # try:
        client_socket, _ = server_socket.accept()
        request = client_socket.recv(1024)
        if request == valid_request:
            client_socket.sendall(response)
            client_socket.close()
            # return
        # except:
        #     print('got exception at listen and response, ignore...')
        #     continue


def connect_and_send(host:str, port:int, request:bytes, time_out:int=300)->bool:
    """try to connect and send request until time_out. if time out, return False, else return True"""
    start_time = time.time()
    while (time.time() - start_time < time_out):
        if is_port_open(host, port):
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            client_socket.sendall(request)
            return True
        else:
            time.sleep(1)
    return False

def connect_and_get_response(host:str, port:int, request:bytes, time_out:int=300)->bytes:
    """try to connect and get response until time_out. if time out, return None"""
    start_time = time.time()
    while (time.time() - start_time < time_out):
        if is_port_open(host, port):
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            client_socket.sendall(request)
            response = client_socket.recv(10240)
            return response
        else:
            time.sleep(1)
    return None

def get_pid(cmd_str:str)->int:
    """find pid by command-line-string, cmd_str can be command-name, or param=name, if no matching process, return None"""
    for proc in psutil.process_iter():
        if cmd_str in proc.cmdline():
            return proc.pid
    return None