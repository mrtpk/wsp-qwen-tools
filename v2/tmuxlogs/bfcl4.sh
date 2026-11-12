cd @ROOT@/v2/eval
export QWEN35_EXTRA_MODELS=qwen35-wsp2,qwen35-wsp2-s2,qwen35-wsp4,qwen35-wsp4-s2
./gen.sh qwen35-wsp4-think-FC 8301 single_turn,multi_turn,memory > logs-gen4-think-s1.log 2>&1 &
./gen.sh qwen35-wsp4-nothink-FC 8301 single_turn,multi_turn,memory > logs-gen4-nothink-s1.log 2>&1 &
./gen.sh qwen35-wsp4-s2-think-FC 8302 single_turn,multi_turn,memory > logs-gen4-think-s2.log 2>&1 &
./gen.sh qwen35-wsp4-s2-nothink-FC 8302 single_turn,multi_turn,memory > logs-gen4-nothink-s2.log 2>&1 &
wait
echo __EXIT__
