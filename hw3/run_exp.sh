# 1. Run experiments on SST
# PREF='sst'
# python classifier.py \
#     --use_gpu \
#     --option finetune \
#     --lr 1e-5 \
#     --hidden_dropout_prob 0.1 \
#     --epochs 5 \
#     --batch_size 32 \
#     --grad_accumulation_steps 1 \
#     --warmup_ratio 0.1 \
#     --max_length 128 \
#     --weight_decay 0.05 \
#     --seed 42

# 2.2 Fine-tune on CFIMDB with advanced techniques
PREF='cfimdb'
python classifier.py \
    --use_gpu \
    --option finetune \
    --lr 2e-5 \
    --hidden_dropout_prob 0.1 \
    --epochs 4 \
    --batch_size 16 \
    --grad_accumulation_steps 2 \
    --warmup_ratio 0.06 \
    --max_length 256 \
    --weight_decay 0.01 \
    --seed 42 \
    --use_vat \
    --gradual_unfreeze
CAMPUSID='9088915203'
mkdir -p $CAMPUSID

# Step 1: Domain adaptation through continued pre-training
# First on SST
PREF='sst'
python classifier.py \
    --use_gpu \
    --option pretrain \
    --lr 1e-4 \
    --hidden_dropout_prob 0.1 \
    --epochs 3 \
    --batch_size 32 \
    --grad_accumulation_steps 1 \
    --warmup_ratio 0.1 \
    --max_length 128 \
    --weight_decay 0.01 \
    --seed 42

# Then on CFIMDB for domain adaptation
PREF='cfimdb'
python classifier.py \
    --use_gpu \
    --option pretrain \
    --lr 1e-4 \
    --hidden_dropout_prob 0.1 \
    --epochs 3 \
    --batch_size 16 \
    --grad_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --max_length 256 \
    --weight_decay 0.01 \
    --seed 42

# Step 2: Enhanced fine-tuning with gradual unfreezing
## 2.1 Fine-tune on SST
PREF='sst'
python classifier.py \
    --use_gpu \
    --option finetune \
    --lr 2e-5 \
    --hidden_dropout_prob 0.1 \
    --epochs 5 \
    --batch_size 32 \
    --grad_accumulation_steps 1 \
    --warmup_ratio 0.1 \
    --max_length 128 \
    --weight_decay 0.01 \
    --seed 42 \
    --use_vat \
    --gradual_unfreeze \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_out "${CAMPUSID}/${PREF}-test-output.txt" \
    --filepath "${CAMPUSID}/${PREF}-model.pt" | tee ${CAMPUSID}/${PREF}-train-log.txt

##  2.2 Run experiments on CF-IMDB
PREF='cfimdb'
python classifier.py \
    --use_gpu \
    --option finetune \
    --lr 2e-5 \
    --hidden_dropout_prob 0.1 \
    --epochs 4 \
    --batch_size 8 \
    --grad_accumulation_steps 4 \
    --warmup_ratio 0.06 \
    --max_length 256 \
    --weight_decay 0.01 \
    --seed 42 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_out "${CAMPUSID}/${PREF}-test-output.txt" \
    --filepath "${CAMPUSID}/${PREF}-model.pt" | tee ${CAMPUSID}/${PREF}-train-log.txt



# Step 3. Prepare submission:
##  3.1. Copy your code to the $CAMPUSID folder
for file in *.py; do cp "$file" "${CAMPUSID}/"; done
for file in *.sh; do cp "$file" "${CAMPUSID}/"; done
for file in *.md; do cp "$file" "${CAMPUSID}/"; done
for file in *.txt; do cp "$file" "${CAMPUSID}/"; done

# Copy specific required files if they exist in parent directory
cp sanity_check.data "${CAMPUSID}/" 2>/dev/null || :
cp README.md "${CAMPUSID}/" 2>/dev/null || :
cp structure.md "${CAMPUSID}/" 2>/dev/null || :
cp setup.py "${CAMPUSID}/" 2>/dev/null || :

##  3.2. Compress the $CAMPUSID folder to $CAMPUSID.zip (containing only .py/.txt/.pdf/.sh files)
python prepare_submit.py ${CAMPUSID} ${CAMPUSID}
##  3.3. Submit the zip file to Canvas! Congrats!
