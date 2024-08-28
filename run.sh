gpu_n=$1
DATASET=$2

# seed=42 
# BATCH_SIZE=64
# SLIDE_WIN=16
# SLIDE_STRIDE=3
# test_ratio=0.2
# val_ratio=0.1
# dim=64
# out_layer_num=1
# topk=4
# out_layer_inter_dim=128
# decay=0.5

seed=42 
BATCH_SIZE=64
SLIDE_WIN=16
SLIDE_STRIDE=3
test_ratio=0.2
val_ratio=0.1
dim=64
out_layer_num=1
topk=4
out_layer_inter_dim=128
decay=0.5


path_pattern="${DATASET}"
COMMENT="${DATASET}"

EPOCH=100


report='best'

if [[ "$gpu_n" == "cpu" ]]; then
    python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -device 'cpu'
else
    CUDA_VISIBLE_DEVICES=$gpu_n  
    python main4classify.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk 
fi