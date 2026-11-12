cd @ROOT@/v2
for r in st_A-think st_A-nothink mt_A-think mt_A-nothink; do
  ./wsp forward cells/run1/req-$r.json 2>&1 | grep -v -i "warn" | tail -25
done
echo __EXIT__ $?
