#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=glue
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=10g
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-39%4

# trap_handler () {
#    echo "Caught signal: " $1
#    # SIGTERM must be bypassed
#    if [ "$1" = "TERM" ]; then
#        echo "bypass sigterm"
#    else
#      # Submit a new job to the queue
#      echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
#      # SLURM_JOB_ID is a unique representation of the job, equivalent
#      # to above
#      scontrol requeue $SLURM_JOB_ID
#    fi
# }


# # Install signal handler
# trap 'trap_handler USR1' USR1
# trap 'trap_handler TERM' TERM

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}

#offline mode
# export TRANSFORMERS_OFFLINE=1
# export WANDB_MODE=offline

export TASK_NAME=mnli
#cola,mnli,mrpc,qnli,qqp,rte,sst2,stsb,wnli
metric="accuracy"
adapter_config="ia3"
#"pfeiffer"
#"houlsby"
#"lora"

# pretrained_adapter="merged_adapters/pure_lora/initialization_30_lora_rte/glue/"
# "AdapterHub/bert-base-uncased-pf-mnli"
# "permutated-pfeiffer-mnli/glue_mnli/"
# "AdapterHub/roberta-base-pf-mnli"/rte
#"lingaccept/cola@ukp"
#"merged_adapters/pfeiffer/for_sst2/cola_sst2"
#"sts/qqp@ukp"
#"lingaccept/cola@ukp"
#"try-save-adapters/_01/cola"
# "sts/sts-b@ukp"
#"nli/qnli@ukp"
#"nli/multinli@ukp"
#"sentiment/sst-2@ukp"
#
#"sts/mrpc@ukp"

export WANDB_ENTITY="adapter-merge"
export WANDB_PROJECT=glue.${TASK_NAME}
export WANDB_WATCH="all"
# set to "wandb" to use weights & bias
report_to="wandb"
# report_to="none"

DATE=`date +%Y%m%d`

declare -a lr_list=(1e-5 5e-5 1e-4 5e-4 1e-3)
declare -a steps=(1500 3000)
declare -a wd_list=(0.01)
# declare -a dr_list=(0.1 0.2)
declare -a bsz_list=(16 32 64 128)

taskid=${SLURM_ARRAY_TASK_ID}
len=${#lr_list[@]}
i=$(( taskid%len ))
lr=${lr_list[$i]}

taskid=$(( taskid/len ))
len=${#steps[@]}
i=$(( taskid%len ))
max_steps=${steps[$i]}

taskid=$(( taskid/len ))
len=${#wd_list[@]}
i=$(( taskid%len ))
weight_decay=${wd_list[$i]}

# taskid=$(( taskid/len ))
# len=${#dr_list[@]}
# i=$(( taskid%len ))
# dropout=${dr_list[$i]}

taskid=$(( taskid/len ))
len=${#bsz_list[@]}
i=$(( taskid%len ))
bsz=${bsz_list[$i]}

# declare -a tf_list=("split-train-set/mnli-1k-0.json", "split-train-set/mnli-1k-1.json")
# train_file=${tf_list[SLURM_ARRAY_TASK_ID]}
# train_file="split-train-set/mnli-1k-1.json"

debug=0
# bsz=32
gradient_steps=1
# lr=1e-2
max_grad_norm=1
# lr=5e-5
# weight_decay=0
# weight_decay=0.02
warmup_updates=0
warmup_ratio=0.06
# max_steps=800
num_train_epochs=25
max_tokens_per_batch=0
max_seq_length=512

dropout=0.1

lr_scheduler_type="polynomial"
max_eval_samples=1600
logging_steps=10

save_strategy="epoch"
eval_strategy="steps" #"epoch"
save_steps=100 #5000
eval_steps=16

if [ "${debug}" = 1 ];
then
    weight_decay=0
    max_grad_norm=1
    max_train_samples=1000
    max_eval_samples=150
    bsz=32 #10
    gradient_steps=1
    num_train_epochs=5
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples} --max_predict_samples 150"
    debug_str=".debug"
fi

rm checkpoints/hf_model/downloads/*.lock
rm checkpoints/hf_model/*.lock

# SAVE=checkpoints/glue/${TASK_NAME}/${DATE}/${pretrained_adapter}/lora_1
exp_name=ia3_tr_train_0.${lr}.${max_steps}.${weight_decay}.${dropout}.${bsz}.${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID} #initialization_30
SAVE=checkpoints/glue/${TASK_NAME}/${DATE}/${exp_name}
#  --do_train \
# roberta-large  roberta-base

#   --do_fisher \  
  #   --task_name $TASK_NAME \
  #--max_train_samples 1000 \
  #--max_eval_samples 1000 \
  #   --validation_file split-train-set/rte-0.json \
  # --train_file split-train-set/rte-1.json \
  # --seed 54 \
  
  # --load_adapter ${pretrained_adapter} \
  # --train_file split-train-set/mnli-0.json \
  # --validation_file split-train-set/mnli-eval.json \
  
  # --train_file split-train-set/rte-1k-0.json \
  # --validation_file split-train-set/rte-eval.json \


python -u examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --hidden_dropout_prob ${dropout} \
  --attention_probs_dropout_prob ${dropout} \
  --max_seq_length 128 \
  --per_device_train_batch_size ${bsz} \
  --per_device_eval_batch_size ${bsz} \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-6 \
  --gradient_accumulation_steps ${gradient_steps} \
  --max_steps ${max_steps} \
  --num_train_epochs ${num_train_epochs} \
  --learning_rate ${lr} \
  --lr_scheduler_type ${lr_scheduler_type} \
  --max_grad_norm ${max_grad_norm} \
  --weight_decay ${weight_decay} \
  --warmup_steps ${warmup_updates} \
  --warmup_ratio ${warmup_ratio} \
  --logging_steps ${logging_steps} \
  --save_total_limit 5 \
  --evaluation_strategy ${eval_strategy} \
  --save_strategy ${eval_strategy} \
  --save_steps ${save_steps} \
  --eval_steps ${save_steps} \
  --load_best_model_at_end \
  --report_to ${report_to} \
  --run_name ${TASK_NAME}.${DATE}.${exp_name} \
  --overwrite_output_dir \
  --train_adapter \
  --metric_for_best_model ${metric} \
  --greater_is_better "True" \
  --adapter_config ${adapter_config} \
  --disable_tqdm "True" \
  --output_dir ${SAVE} \
    2>&1 | tee ${SAVE}/log.txt
# done
