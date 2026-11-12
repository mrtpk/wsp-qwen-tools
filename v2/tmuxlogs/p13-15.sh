cd @ROOT@/v2
for K in 0p1 0p25 0p5; do
  ./wsp run cells/run1/c11_k$K.py 2>&1 | grep -v -i warn
  ./wsp forward work/req-cal.json 2>&1 | grep -E '"pass"|Error|error'
done
echo __EXIT__ $?
