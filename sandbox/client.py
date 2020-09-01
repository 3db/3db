import zmq
from uuid import uuid4
import sys

port = "5555"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)

context = zmq.Context()
print("Connecting to server...")
socket = context.socket(zmq.REQ)
socket.connect ("tcp://localhost:%s" % port)

worker_id = str(uuid4())

socket.send_pyobj({
    'kind': 'connect',
    'worker_id': worker_id
})

message = socket.recv_pyobj()
print(message)

