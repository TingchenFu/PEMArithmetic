#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=glue
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

export TRANSFORMERS_CACHE=/apdcephfs/share_916081/shared_info/tingchenfu/huggingface_cache
export HF_DATASETS_CACHE=/apdcephfs/share_916081/shared_info/tingchenfu/huggingface_cache
export HF_METRICS_CACHE=/apdcephfs/share_916081/shared_info/tingchenfu/huggingface_cache

cache_dir=${TRANSFORMERS_CACHE}

#offline mode
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

###### roberta-large ######
# model_name_or_path=roberta-large
# adapter_config="pfeiffer"
# task_names=("cola" "qnli" "sst2" "mrpc")
# #"mnli"
# # qqp rte  stsb
# pretrained_adapters=(
# "lingaccept/cola@ukp"
# "nli/qnli@ukp"
# "sentiment/sst-2@ukp"
# "sts/mrpc@ukp")
# # "sts/sts-b@ukp"
# # "nli/multinli@ukp"

#"houlsby"

###### roberta-base ######
model_name_or_path=roberta-base
adapter_config="lora"
# adapter_config="ia3"
# task_names=("cola" "sst2" "mrpc" "qqp" "stsb" "mnli" "qnli" "rte")
# pretrained_adapters=(
# "lingaccept/cola@ukp"
# "AdapterHub/roberta-base-pf-sst2"
# "AdapterHub/roberta-base-pf-mrpc"
# "AdapterHub/roberta-base-pf-qqp"
# "AdapterHub/roberta-base-pf-stsb"
# "AdapterHub/roberta-base-pf-mnli"
# "AdapterHub/roberta-base-pf-qnli"
# "AdapterHub/roberta-base-pf-rte"
# )

# export WANDB_PROJECT=glue.${TASK_NAME}
# export WANDB_WATCH="false"
# set to "wandb" to use weights & bias
report_to="none"

DATE=`date +%Y%m%d`

metric=accuracy

# ######## mnli split #######
task_names=("mnli" "mnli")
pretrained_adapters=(
    "/data/home/tingchenfu/PEMArithmetic/dump/glue/mnli/fft_train0.32.1e-5.4000.0.01/20230712"
    "/data/home/tingchenfu/PEMArithmetic/dump/glue/mnli/fft_train0.32.1e-5.4000.0.01/20230712"
)

# pretrained_adapters=(
#     "checkpoints/glue/mnli/20230113/ia3_try_train.13247713.0/glue/"
#     "checkpoints/glue/mnli/20230111/ia3_try_train.13210234.0/glue/"
# )

######## rte split #######
# task_names=("rte" "rte")
# pretrained_adapters=(
#     "checkpoints/glue/rte/20230101/lora_0_basic/glue/"
#     "checkpoints/glue/rte/20230101/lora_1_basic/glue/"
# )

# task_names=("mnli" "rte")
# pretrained_adapters=(
#     "AdapterHub/roberta-base-pf-mnli"
#     "AdapterHub/roberta-base-pf-rte"
# )
# "permutated-pfeiffer-mnli/glue_mnli/"
    # "AdapterHub/roberta-base-pf-mnli"

x=0
y=1
for set0 in $(seq 0 100 100) # $(seq 20 20 100)
do
    # echo $set0
    set1=`expr 100 - $set0`
    echo "($set0, $set1)"
    adapter="merged_adapters/simple"
    adapter="${adapter}/${adapter_config}/for_${task_names[$y]}/${task_names[$x]}0_${task_names[$y]}1_${set0}_${set1}"
    echo $adapter

    python -u merge.py \
    --model ${model_name_or_path} \
    --adatasks ${task_names[$x]} ${task_names[$y]} \
    --adapters ${pretrained_adapters[$x]} ${pretrained_adapters[$y]} \
    --adaconfig ${adapter_config} \
    --save_path ${adapter} \
    --merge_way simple \
    --merge_head true \
    --overwrite true \
    --search_step $set0 $set1

    adapter="${adapter}/${task_names[$x]}_${task_names[$y]}"
    task_name=${task_names[$y]}
    SAVE=checkpoints/glue/${task_name}/${adapter}/${DATE}/eval

    echo ${SAVE}

    python -u examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path ${model_name_or_path} \
    --train_file "split-train-set/mnli-1k-1.json" \
    --validation_file "split-train-set/mnli-eval.json" \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir ${SAVE} \
    --overwrite_output_dir \
    --train_adapter \
    --load_adapter ${adapter} \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --report_to ${report_to} \
    --adapter_config ${adapter_config} \
        2>&1 | tee ${SAVE}/log.txt

done