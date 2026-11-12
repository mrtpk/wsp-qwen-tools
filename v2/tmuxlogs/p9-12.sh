cd @ROOT@/v2
for K in 0 1 4 16; do
  ./wsp run cells/run1/c09_k$K.py 2>&1 | grep -v -i warn
  ./wsp forward work/req-cal.json 2>&1 | grep -E '"pass"|"seconds"|"weights"|Error|error'
done
echo __EXIT__ $?
