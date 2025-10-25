#!/bin/bash

# Step 0. Change this to your campus ID
CAMPUSID='9088915203'
# mkdir -p $CAMPUSID

# # Step 1. Domain-Adaptive MLM Pre-training (NEW - for 100 marks innovation)
# echo "=========================================="
# echo "Step 1: Domain-Adaptive MLM Pre-training"
# echo "=========================================="

# # Pre-train on SST
# python3 pretrain_sst_mlm.py \
#     --input_file data/sst-train.txt \
#     --output_model bert-sst-mlm.pt \
#     --epochs 2 \
#     --batch_size 16 \
#     --lr 1e-5

# # Pre-train on CFIMDB
# python3 pretrain_sst_mlm.py \
#     --input_file data/cfimdb-train.txt \
#     --output_model bert-cfimdb-mlm.pt \
#     --epochs 2 \
#     --batch_size 16 \
#     --lr 1e-5

# # Step 2. Train models on two datasets.
# ##  2.1. Run experiments on SST
# echo ""
# echo "=========================================="
# echo "Step 2.1: Fine-tune on SST"
# echo "=========================================="

# PREF='sst'
# python3 classifier.py \
#     --use_gpu \
#     --option finetune \
#     --epochs 3 \
#     --lr 2e-5 \
#     --seed 11711 \
#     --train "data/${PREF}-train.txt" \
#     --dev "data/${PREF}-dev.txt" \
#     --test "data/${PREF}-test.txt" \
#     --dev_out "${CAMPUSID}/${PREF}-dev-output.txt" \
#     --test_out "${CAMPUSID}/${PREF}-test-output.txt" \
#     --filepath "${CAMPUSID}/${PREF}-model.pt" \
#     --pretrained_model bert-sst-mlm.pt \
#     --discriminative_lr \
#     --gradual_unfreeze \
#     --batch_size 32 \
#     --patience 4 | tee ${CAMPUSID}/${PREF}-train-log.txt

# ##  2.2 Run experiments on CFIMDB
# echo ""
# echo "=========================================="
# echo "Step 2.2: Fine-tune on CFIMDB"
# echo "=========================================="

# PREF='cfimdb'
# python3 classifier.py \
#     --use_gpu \
#     --option finetune \
#     --epochs 4 \
#     --lr 2e-5 \
#     --seed 11711 \
#     --train "data/${PREF}-train.txt" \
#     --dev "data/${PREF}-dev.txt" \
#     --test "data/${PREF}-test.txt" \
#     --dev_out "${CAMPUSID}/${PREF}-dev-output.txt" \
#     --test_out "${CAMPUSID}/${PREF}-test-output.txt" \
#     --filepath "${CAMPUSID}/${PREF}-model.pt" \
#     --pretrained_model bert-cfimdb-mlm.pt \
#     --discriminative_lr \
#     --gradual_unfreeze \
#     --batch_size 32 \
#     --patience 6 | tee ${CAMPUSID}/${PREF}-train-log.txt

# # Step 3. Prepare submission:
# echo ""
# echo "=========================================="
# echo "Step 3: Prepare Submission"
# echo "=========================================="

# ##  3.1. Copy your code to the $CAMPUSID folder
# for file in *.py; do cp $file ${CAMPUSID}/ ; done
# for file in *.sh; do cp $file ${CAMPUSID}/ ; done
# for file in *.md; do cp $file ${CAMPUSID}/ ; done
# for file in *.txt; do cp $file ${CAMPUSID}/ ; done

# ##  3.2. Compress the $CAMPUSID folder to $CAMPUSID.zip
python3 prepare_submit.py ${CAMPUSID} ${CAMPUSID}

##  3.3. Submit the zip file to Canvas! Congrats!
echo ""
echo "✅ Done! Submission ready: ${CAMPUSID}.zip"