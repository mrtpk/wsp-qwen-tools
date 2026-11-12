set -e
V2=@ROOT@/v2
cd $V2
OPENWSP_WORK=$V2/work3 ./wsp export edits/run3-v8 > tmuxlogs/r3-export8.log 2>&1
PYTHONPATH=openwsp/src ../.venv/bin/python eval/build_checkpoint.py @BASE@ edits/run3-v8 models/qwen35-wsp8 > tmuxlogs/r3-ckpt8.log 2>&1
tmux new-window -d -t v2 -n srv8a "bash eval/serve.sh $V2/models/qwen35-wsp8 qwen35-wsp8 8301 0.4 262144 0 > tmuxlogs/bfcl8-srv1.log 2>&1"
until curl -sf localhost:8301/v1/models >/dev/null; do sleep 10; done
tmux new-window -d -t v2 -n srv8b "bash eval/serve.sh $V2/models/qwen35-wsp8 qwen35-wsp8-s2 8302 0.4 262144 1 > tmuxlogs/bfcl8-srv2.log 2>&1"
until curl -sf localhost:8302/v1/models >/dev/null; do sleep 10; done
date +%H:%M
cd eval
export QWEN35_EXTRA_MODELS=qwen35-wsp2,qwen35-wsp2-s2,qwen35-wsp4,qwen35-wsp4-s2,qwen35-wsp6,qwen35-wsp6-s2,qwen35-wsp7,qwen35-wsp7-s2,qwen35-wsp8,qwen35-wsp8-s2
./gen.sh qwen35-wsp8-nothink-FC 8301 single_turn,multi_turn,memory > logs-gen8-nothink-s1.log 2>&1 &
./gen.sh qwen35-wsp8-s2-nothink-FC 8302 single_turn,multi_turn,memory > logs-gen8-nothink-s2.log 2>&1 &
wait
date +%H:%M
echo __EXIT__
