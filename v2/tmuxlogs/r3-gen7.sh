cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work3
for r in tB-think mB-think tB-nothink gB-think; do
  ./wsp forward work3/req-gen-$r-v5.json > tmuxlogs/r3-gen7-$r.log 2>&1
  grep -E '"pass"|"seconds"|rror' tmuxlogs/r3-gen7-$r.log | tail -3
done
touch tmuxlogs/r3-gen7.done
