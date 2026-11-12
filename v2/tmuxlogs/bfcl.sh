cd @ROOT@/v2/eval
export QWEN35_EXTRA_MODELS=qwen35-wsp2,qwen35-wsp2-s2
wait_up() { until curl -sf localhost:$1/v1/models > /dev/null; do sleep 10; done; }
wait_up 8301; wait_up 8302
./gen.sh qwen35-wsp2-think-FC 8301 single_turn,multi_turn,memory > logs-gen-think-s1.log 2>&1 &
./gen.sh qwen35-wsp2-nothink-FC 8301 single_turn,multi_turn,memory > logs-gen-nothink-s1.log 2>&1 &
./gen.sh qwen35-wsp2-s2-think-FC 8302 single_turn,multi_turn,memory > logs-gen-think-s2.log 2>&1 &
./gen.sh qwen35-wsp2-s2-nothink-FC 8302 single_turn,multi_turn,memory > logs-gen-nothink-s2.log 2>&1 &
wait
echo __EXIT__
