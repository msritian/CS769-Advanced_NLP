# # 1. Run experiments on SST
# # PREF='sst'
# # python classifier.py \
# #     --use_gpu \
# #     --option finetune \
# #     --lr 1e-5 \
# #     --hidden_dropout_prob 0.1 \
# #     --epochs 5 \
# #     --batch_size 32 \
# #     --grad_accumulation_steps 1 \
# #     --warmup_ratio 0.1 \
# #     --max_length 128 \
# #     --weight_decay 0.05 \
# #     --seed 42

# # 2.2 Fine-tune on CFIMDB with advanced techniques
# PREF='cfimdb'
# python classifier.py \
#     --use_gpu \
#     --option finetune \
#     --lr 2e-5 \
#     --hidden_dropout_prob 0.1 \
#     --epochs 4 \
#     --batch_size 16 \
#     --grad_accumulation_steps 2 \
#     --warmup_ratio 0.06 \
#     --max_length 256 \
#     --weight_decay 0.01 \
#     --seed 42
# CAMPUSID='9088915203'
# mkdir -p $CAMPUSID

# # Step 1: Domain adaptation through continued pre-training
# # First on SST
# PREF='sst'
# python classifier.py \
#     --use_gpu \
#     --option pretrain \
#     --lr 1e-4 \
#     --hidden_dropout_prob 0.1 \
#     --epochs 3 \
#     --batch_size 32 \
#     --grad_accumulation_steps 1 \
#     --warmup_ratio 0.1 \
#     --max_length 128 \
#     --weight_decay 0.01 \
#     --seed 42

# # Then on CFIMDB for domain adaptation
# PREF='cfimdb'
# python classifier.py \
#     --use_gpu \
#     --option pretrain \
#     --lr 1e-4 \
#     --hidden_dropout_prob 0.1 \
#     --epochs 3 \
#     --batch_size 16 \
#     --grad_accumulation_steps 2 \
#     --warmup_ratio 0.1 \
#     --max_length 256 \
#     --weight_decay 0.01 \
#     --seed 42

# # Step 2: Fine-tuning with gradual unfreezing and discriminative learning rates
# PREF='sst'
# python classifier.py \
#     --use_gpu \
#     --option finetune \
#     --lr 2e-5 \
#     --hidden_dropout_prob 0.2 \
#     --epochs 5 \
#     --batch_size 4 \
#     --grad_accumulation_steps 1 \
#     --warmup_ratio 0.1 \
#     --max_length 128 \
#     --weight_decay 0.05 \
#     --seed 42 \
#     --gradual_unfreeze \
#     --discriminative_lr

# PREF='cfimdb'
# python classifier.py \
#     --use_gpu \
#     --option finetune \
#     --lr 2e-5 \
#     --hidden_dropout_prob 0.2 \
#     --epochs 5 \
#     --batch_size 4 \
#     --grad_accumulation_steps 2 \
#     --warmup_ratio 0.1 \
#     --max_length 128 \
#     --weight_decay 0.05 \
#     --seed 42 \
#     --gradual_unfreeze \
#     --discriminative_lr

# # Step 3. Prepare submission:
# ##  3.1. Copy your code to the $CAMPUSID folder
# for file in *.py; do cp "$file" "${CAMPUSID}/"; done
# for file in *.sh; do cp "$file" "${CAMPUSID}/"; done
# for file in *.md; do cp "$file" "${CAMPUSID}/"; done
# for file in *.txt; do cp "$file" "${CAMPUSID}/"; done

# # Copy specific required files if they exist in parent directory
# cp sanity_check.data "${CAMPUSID}/" 2>/dev/null || :
# cp README.md "${CAMPUSID}/" 2>/dev/null || :
# cp structure.md "${CAMPUSID}/" 2>/dev/null || :
# cp setup.py "${CAMPUSID}/" 2>/dev/null || :

# ##  3.2. Compress the $CAMPUSID folder to $CAMPUSID.zip (containing only .py/.txt/.pdf/.sh files)
# python prepare_submit.py ${CAMPUSID} ${CAMPUSID}
# ##  3.3. Submit the zip file to Canvas! Congrats!


#!/bin/bash

# SST Dataset - Pretrain
echo "========================================="
echo "Training on SST - Pretrain Mode"
echo "========================================="
python3 classifier.py \
  --option pretrain \
  --epochs 15 \
  --lr 2e-5 \
  --train data/sst-train.txt \
  --dev data/sst-dev.txt \
  --test data/sst-test.txt \
  --dev_out sst-dev-output-pretrain.txt \
  --test_out sst-test-output-pretrain.txt \
  --filepath sst-pretrain-15-2e-5.pt \
  --batch_size 32 \
  --use_gpu \
  --patience 3 > sst-train-pretrain-log.txt 2>&1

# SST Dataset - Finetune
echo "========================================="
echo "Training on SST - Finetune Mode"
echo "========================================="
python3 classifier.py \
  --option finetune \
  --epochs 15 \
  --lr 2e-5 \
  --train data/sst-train.txt \
  --dev data/sst-dev.txt \
  --test data/sst-test.txt \
  --dev_out sst-dev-output-finetune.txt \
  --test_out sst-test-output-finetune.txt \
  --filepath sst-finetune-15-2e-5.pt \
  --batch_size 32 \
  --use_gpu \
  --discriminative_lr \
  --gradual_unfreeze \
  --patience 3 > sst-train-finetune-log.txt 2>&1

# CFIMDB Dataset - Finetune
echo "========================================="
echo "Training on CFIMDB - Finetune Mode"
echo "========================================="
python3 classifier.py \
  --option finetune \
  --epochs 15 \
  --lr 2e-5 \
  --train data/cfimdb-train.txt \
  --dev data/cfimdb-dev.txt \
  --test data/cfimdb-test.txt \
  --dev_out cfimdb-dev-output.txt \
  --test_out cfimdb-test-output.txt \
  --filepath cfimdb-finetune-15-2e-5.pt \
  --batch_size 8 \
  --use_gpu \
  --discriminative_lr \
  --gradual_unfreeze \
  --patience 3 > cfimdb-train-log.txt 2>&1

echo "========================================="
echo "All experiments completed!"
echo "========================================="