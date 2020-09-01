import zmq
import time
import numpy as np
import sys

port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: ", message)
    time.sleep (1)  
    socket.send_pyobj({'message': "World from %s" % port, 'data': np.random.uniform(size=10)})
