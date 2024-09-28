DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
mkdir -p "$DATA_RUN_PATH"
DATA_RESULTS_PATH="$run_base/$run_path/$MODEL/RESULTS"

mkdir -p "$DATA_RESULTS_PATH"
		
RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_RESULTS_SUMMARY.log"
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | 
tee -- "$RESULTS_FILE"
		
OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"

python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- 
"$OPT_RESULTS_FILE" 



OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"

python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- 
"$OPT_ROC_RESULTS_FILE"
