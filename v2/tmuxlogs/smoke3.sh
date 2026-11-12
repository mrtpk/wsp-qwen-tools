cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work-smoke OPENWSP_PASS_BUDGET=5
./wsp run cells/smoke/c2.py && ./wsp forward cells/smoke/req-gen2.json | tail -8 && ./wsp status
echo __EXIT__ $?
