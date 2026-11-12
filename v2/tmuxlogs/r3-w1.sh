cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work3
./wsp run cells/run3/c25_gen_wA_req.py 2>&1 | grep -vi warn
./wsp forward work3/req-gen-wA-think.json > tmuxlogs/r3-gen-wA.log 2>&1
grep -E '"pass"|"seconds"|rror' tmuxlogs/r3-gen-wA.log | tail -3
touch tmuxlogs/r3-w1.done
