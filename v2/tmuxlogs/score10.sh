cd @ROOT@/v2/eval
export QWEN35_EXTRA_MODELS=qwen35-wsp2,qwen35-wsp2-s2,qwen35-wsp4,qwen35-wsp4-s2,qwen35-wsp6,qwen35-wsp6-s2,qwen35-wsp7,qwen35-wsp7-s2,qwen35-wsp8,qwen35-wsp8-s2,qwen35-wsp9,qwen35-wsp9-s2,qwen35-wsp10,qwen35-wsp10-s2,qwen35-base-s3,qwen35-wsp7-s3
export BFCL_PROJECT_ROOT=$PWD OPENAI_API_KEY=EMPTY RETRY_LOG=$PWD/retry-log.jsonl
@ROOT@/v1/bfcl/.venv/bin/python run_bfcl_retry.py evaluate --model qwen35-wsp10-think-FC,qwen35-wsp10-s2-think-FC --test-category single_turn,multi_turn,memory
echo __EXIT__ $?
