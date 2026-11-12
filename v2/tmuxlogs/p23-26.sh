cd @ROOT@/v2
for r in st_B-think st_B-nothink mt_B-think mt_B-nothink; do
  ./wsp forward cells/run1/req-$r-both.json 2>&1 | grep -E '"pass"|with_tool_call|Error|error'
done
echo __EXIT__ $?
