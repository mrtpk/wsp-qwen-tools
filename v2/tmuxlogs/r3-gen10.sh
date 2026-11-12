cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work3
for r in tB-think-v5 mB-think-v5 gB-think-v5 wB-think-v10; do
  ./wsp forward work3/req-gen-$r.json > tmuxlogs/r3-gen10-$r.log 2>&1
  grep -E '"pass"|"seconds"|rror' tmuxlogs/r3-gen10-$r.log | tail -3
done
./wsp run cells/run3/c28_validate_v10.py 2>&1 | grep -vi warn
touch tmuxlogs/r3-gen10.done
