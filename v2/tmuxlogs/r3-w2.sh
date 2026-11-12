cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work3
./wsp run cells/run3/c26_probe_wA_req.py 2>&1 | grep -vi warn
./wsp forward work3/req-probe-wA-think.json > tmuxlogs/r3-probe-wA.log 2>&1
grep -E '"pass"|"seconds"|rror' tmuxlogs/r3-probe-wA.log | tail -3
./wsp run cells/run3/c27_edit_v10.py 2>&1 | grep -vi warn
touch tmuxlogs/r3-w2.done
