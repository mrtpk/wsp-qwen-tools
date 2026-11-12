cd @ROOT@/v2/eval
export QWEN35_EXTRA_MODELS=qwen35-wsp2,qwen35-wsp2-s2,qwen35-wsp4,qwen35-wsp4-s2,qwen35-wsp6,qwen35-wsp6-s2,qwen35-wsp7,qwen35-wsp7-s2,qwen35-wsp8,qwen35-wsp8-s2
date +%H:%M
./gen.sh qwen35-wsp8-s2-nothink-FC 8302 single_turn,multi_turn,memory > logs-gen8-nothink-s2b.log 2>&1
date +%H:%M
echo __EXIT__
