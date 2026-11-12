cd @ROOT@/v2
for r in st_A-think st_A-nothink mt_A-think mt_A-nothink; do
  ./wsp forward work/req-probe-$r.json 2>&1 | grep -E '"pass"|"seconds"|Error|error' 
done
echo __EXIT__ $?
