cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work-smoke OPENWSP_PASS_BUDGET=5
./wsp forward cells/smoke/req-gen.json && ./wsp forward cells/smoke/req-gen.json | grep -E '"cached"|"pass"' && ./wsp forward cells/smoke/req-tf.json
echo __EXIT__ $?
