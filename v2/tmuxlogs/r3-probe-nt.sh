cd @ROOT@/v2
export OPENWSP_WORK=$PWD/work3
for s in tA mA; do
  ./wsp forward work3/req-probe-$s-nothink.json > tmuxlogs/r3-probe-$s-nothink.log 2>&1
  grep -E '"pass"|"seconds"|rror' tmuxlogs/r3-probe-$s-nothink.log | tail -3
done
touch tmuxlogs/r3-probe-nt.done
