import zmq
import random
import json


def schedule_work(policy_controllers, port, list_envs, list_models,
                  render_args, inference_args):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    workers = {}

    controllers_to_start = policy_controllers[:]
    running_policies = set()
    done_policies = set()

    # Load the mapping for UIDs to logits indices
    with open(inference_args['uid_to_logits'], 'r') as f:
        uid_to_logits = json.load(f)

    while True:
        if len(done_policies) == len(policy_controllers):
            break  # We finished all the policies
        message = socket.recv_pyobj()

        wid = message['worker_id']

        if message['kind'] == 'info':
            socket.send_pyobj({
                'kind': 'info',
                'environments': list_envs,
                'models': list_models
            })
        elif message['kind'] == 'connect':
            if len(controllers_to_start):
                selected_policy = controllers_to_start.pop()
                selected_policy.start()
                running_policies.add(selected_policy)
            else:
                selected_policy = random.choice(running_policies)

            assert(wid not in workers)
            workers[wid] = selected_policy
            socket.send_pyobj({
                'kind': 'assignment',
                'environment': selected_policy.env_file,
                'model': selected_policy.model_name,
                'uid_to_logits': uid_to_logits,
                'inference': inference_args
            })

        elif message['kind'] == 'pull':
            selected_policy = workers[wid]
            if selected_policy in done_policies:
                del workers[wid]
                socket.send_pyobj({
                    'kind': 'done'
                })
            else:
                bs = message['batch_size']
                result = []
                for _ in range(bs):
                    pulled = selected_policy.pull_work(wid)
                    if pulled is None:
                        break
                    result.append(pulled)

                socket.send_pyobj({
                    'kind': 'work',
                    'environment': selected_policy.env_file,
                    'model': selected_policy.model_name,
                    'params_to_render': result,
                    'render_args': render_args
                })

        elif message['kind'] == 'push':
            selected_policy = workers[wid]
            selected_policy.push_result(message['job'], message['result'])
            socket.send_pyobj({
                'kind': 'ack'
            })

        else:
            socket.send_pyobj({
                'kind': 'bad_query'
            })

        for policy in list(running_policies):
            if not policy.is_alive():
                running_policies.remove(policy)
                done_policies.add(policy)
