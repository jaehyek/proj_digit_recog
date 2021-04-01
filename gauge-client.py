import socket
from env_const import *
import json

HOST = '127.0.0.1'
PORT = 9999

def gauge_client_cmd(message):
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    
    client_socket.send(message.encode())
    data = client_socket.recv(1024)

    print('Received from the server :',repr(data.decode()))
    client_socket.close()

if __name__ == '__main__' :
    # cmd = json.dumps(dict_cmd_makeImageFolder)
    # cmd = json.dumps(dict_cmd_imageAugmentation)
    # cmd = json.dumps(dict_cmd_makeTrainValidFromDigitClass)
    # cmd = json.dumps(dict_cmd_DigitRecogModel)
    # cmd = json.dumps(dict_cmd_getValueFromJson)
    cmd = json.dumps(dict_cmd_serverclientTest)
    print(cmd)
    gauge_client_cmd(cmd)