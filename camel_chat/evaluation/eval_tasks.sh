 #!/bin/bash -l

MODEL_PATH=
MODEL_NAME=
MODEL_ARGS=pretrained=$MODEL_PATH,trust_remote_code=True,use_accelerate=True,dtype=float16

# ARC-Challenge 25 shots
SHOTS=25
subject=arc_challenge

JOB_NAME=$subject-$MODEL_NAME-eval
sbatch --job-name $JOB_NAME run.sh --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $subject --num_fewshot $SHOTS --output_path ./$MODEL_NAME-results/$MODEL_NAME-$subject-$SHOTS-shots.json

# HellaSwag 10 shots
SHOTS=10
subject=hellaswag

JOB_NAME=$subject-$MODEL_NAME-eval
sbatch --job-name $JOB_NAME run.sh --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $subject --num_fewshot $SHOTS --output_path ./$MODEL_NAME-results/$MODEL_NAME-$subject-$SHOTS-shots.json

# MMLU 5 shots
SHOTS=5
declare -a TASK_ARRAY=(hendrycksTest-abstract_algebra hendrycksTest-anatomy hendrycksTest-astronomy hendrycksTest-business_ethics hendrycksTest-clinical_knowledge hendrycksTest-college_biology hendrycksTest-college_chemistry hendrycksTest-college_computer_science hendrycksTest-college_mathematics hendrycksTest-college_medicine hendrycksTest-college_physics hendrycksTest-computer_security hendrycksTest-conceptual_physics hendrycksTest-econometrics hendrycksTest-electrical_engineering hendrycksTest-elementary_mathematics hendrycksTest-formal_logic hendrycksTest-global_facts hendrycksTest-high_school_biology hendrycksTest-high_school_chemistry hendrycksTest-high_school_computer_science hendrycksTest-high_school_european_history hendrycksTest-high_school_geography hendrycksTest-high_school_government_and_politics hendrycksTest-high_school_macroeconomics hendrycksTest-high_school_mathematics hendrycksTest-high_school_microeconomics hendrycksTest-high_school_physics hendrycksTest-high_school_psychology hendrycksTest-high_school_statistics hendrycksTest-high_school_us_history hendrycksTest-high_school_world_history hendrycksTest-human_aging hendrycksTest-human_sexuality hendrycksTest-international_law hendrycksTest-jurisprudence hendrycksTest-logical_fallacies hendrycksTest-machine_learning hendrycksTest-management hendrycksTest-marketing hendrycksTest-medical_genetics hendrycksTest-miscellaneous hendrycksTest-moral_disputes hendrycksTest-moral_scenarios hendrycksTest-nutrition hendrycksTest-philosophy hendrycksTest-prehistory hendrycksTest-professional_accounting hendrycksTest-professional_law hendrycksTest-professional_medicine hendrycksTest-professional_psychology hendrycksTest-public_relations hendrycksTest-security_studies hendrycksTest-sociology hendrycksTest-us_foreign_policy hendrycksTest-virology hendrycksTest-world_religions)

arraylength=${#TASK_ARRAY[@]}
# use for loop to read all values and indexes
for (( i=0; i<${arraylength}; i++ ));
do
    subject=${TASK_ARRAY[$i]}
	JOB_NAME=$subject-$MODEL_NAME-eval
	sbatch --job-name $JOB_NAME run.sh --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $subject --num_fewshot $SHOTS --output_path ./$MODEL_NAME-results/$MODEL_NAME-$subject-$SHOTS-shots.json
done

# TruthfulQA 0 shots
SHOTS=0
subject=truthfulqa_mc

JOB_NAME=$subject-$MODEL_NAME-eval
sbatch --job-name $JOB_NAME run.sh --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $subject --num_fewshot $SHOTS --output_path ./$MODEL_NAME-results/$MODEL_NAME-$subject-$SHOTS-shots.json