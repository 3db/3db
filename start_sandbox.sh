tmux kill-session -t main_session
tmux kill-session -t client_session
tmux kill-session -t tb_session

tmux start-server
rm -rf /logs
mkdir /logs

tmux new-session -d -s tb_session zsh
tmux send-keys -t tb_session.0 'tensorboard --logdir /logs --host 0.0.0.0' ENTER

tmux new-session -d -s main_session zsh
# tmux send-keys -t main_session.0 "cd /code && python sandbox/main.py /data examples/very_simple_random_search.yaml 5555 --logdir /logs --loggers JSONLogger,ImageLogger --single-model" ENTER
tmux send-keys -t main_session.0 "cd /code && python sandbox/main.py /data examples/very_simple_random_search.yaml 5555 --logdir /logs --single-model" ENTER

tmux new-session -d -s client_session zsh
# tmux send-keys -t client_session.0 "/blender/blender --python-use-system-env -b -P /code/sandbox/client.py -- /data/ --master-address localhost:5555" ENTER
tmux send-keys -t client_session.0 "cd /code && python sandbox/client.py -- /data/ --master-address localhost:5555" ENTER