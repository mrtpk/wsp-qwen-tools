cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work3
for r in tB-nothink-v5 mB-nothink-v8 gB-nothink-v8; do
  ./wsp forward work3/req-gen-$r.json > tmuxlogs/r3-gen8c-$r.log 2>&1
  grep -E '"pass"|"seconds"|rror' tmuxlogs/r3-gen8c-$r.log | tail -3
done
touch tmuxlogs/r3-gen8.done
