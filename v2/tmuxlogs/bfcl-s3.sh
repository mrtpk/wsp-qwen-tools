set -e
V2=@ROOT@/v2
BASE=@BASE@
cd $V2
tmux new-window -d -t v2 -n srvs3a "bash eval/serve.sh $BASE qwen35-base-s3 8301 0.4 262144 2 > tmuxlogs/bfcls3-srv1.log 2>&1"
until curl -sf localhost:8301/v1/models >/dev/null; do sleep 10; done
tmux new-window -d -t v2 -n srvs3b "bash eval/serve.sh $V2/models/qwen35-wsp7 qwen35-wsp7-s3 8302 0.4 262144 2 > tmuxlogs/bfcls3-srv2.log 2>&1"
until curl -sf localhost:8302/v1/models >/dev/null; do sleep 10; done
date +%H:%M
cd eval
export QWEN35_EXTRA_MODELS=qwen35-wsp2,qwen35-wsp2-s2,qwen35-wsp4,qwen35-wsp4-s2,qwen35-wsp6,qwen35-wsp6-s2,qwen35-wsp7,qwen35-wsp7-s2,qwen35-wsp8,qwen35-wsp8-s2,qwen35-wsp9,qwen35-wsp9-s2,qwen35-base-s3,qwen35-wsp7-s3
./gen.sh qwen35-base-s3-think-FC 8301 single_turn,multi_turn,memory > logs-gens3-base-think.log 2>&1 &
./gen.sh qwen35-wsp7-s3-think-FC 8302 single_turn,multi_turn,memory > logs-gens3-wsp7-think.log 2>&1 &
wait
date +%H:%M
echo __EXIT__
