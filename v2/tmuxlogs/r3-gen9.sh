cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work3
for r in tB-think-v5 mB-think-v5 gB-think-v5; do
  ./wsp forward work3/req-gen-$r.json > tmuxlogs/r3-gen9-$r.log 2>&1
  grep -E '"pass"|"seconds"|rror' tmuxlogs/r3-gen9-$r.log | tail -3
done
touch tmuxlogs/r3-gen9.done
