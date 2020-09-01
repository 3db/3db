import zmq
import random

def schedule_work(policy_controllers, port):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    workers = {}

    controllers_to_start = policy_controllers[:]
    running_policies = set()
    done_policies = set()

    while True:
        if len(done_policies) == len(policy_controllers):
            break  # We finished all the policies
        message = socket.recv_pyobj()

        wid = message['worker_id']

        if message['kind'] == 'connect':
            if len(controllers_to_start):
                selected_policy = controllers_to_start.pop()
                selected_policy.run()
                running_policies.add(selected_policy)
            else:
                selected_policy = random.choice(running_policies)

            assert(wid not in workers)
            workers[wid] = selected_policy
            socket.send_pyobj({
                'kind': 'assignment',
                'environment': selected_policy.env_file,
                'model': selected_policy.model_name
            })

        elif message['kind'] == 'pull':
            selected_policy = workers[wid]
            bs = message['batch_size']
            result = []
            for _ in range(bs):
                pulled = selected_policy.pull_work()
                if pulled is None:
                    break
                result.append(pulled)

                socket.send_pyobj({
                    'environment': selected_policy.env_file,
                    'model': selected_policy.model_name,
                    'params_to_render': result
                })

        elif message['kind'] == 'push':
            selected_policy = workers[wid]
            selected_policy.push_result((message['job'], message['result']))

        for policy in list(running_policies):
            if not policy.is_alive():
                running_policies.remove(policy)
                done_policies.add(policy)
