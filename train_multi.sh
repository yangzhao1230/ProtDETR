# env variables for DDP training
[ -z "${MASTER_PORT}" ] && MASTER_PORT=12346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=0
[ -z "${GPUS}" ] && GPUS=$(nvidia-smi -L | wc -l)

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  DISTRIBUTED_ARGS=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                      --master_port $MASTER_PORT"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR"
    fi
fi

# training args
[ -z "${train_data}" ] && train_data=split100
[ -z "${enc_layers}" ] && enc_layers=3
[ -z "${dec_layers}" ] && dec_layers=3
[ -z "${heads}" ] && heads=4
[ -z "${max_labels}" ] && max_labels=10
[ -z "${lr}" ] && lr=1e-4
[ -z "${bs}" ] && bs=8
[ -z "${eos_coef}" ] && eos_coef=0
[ -z "${epoch}" ] && epoch=50
[ -z "${esm_layer}" ] && esm_layer=32
[ -z "${hidden_dim}" ] && hidden_dim=256

model_name="your_model_name" 

torchrun $DISTRIBUTED_ARGS train_multi.py \
    --model_name "${model_name}" \
    --train_data "${train_data}" \
    --lr "${lr}" \
    --batch_size "${bs}" \
    --num_queries "${max_labels}" \
    --enc_layers "${enc_layers}" \
    --dec_layers "${dec_layers}" \
    --nheads "${heads}" \
    --eos_coef "${eos_coef}" \
    --epochs "${epoch}" \
    --esm_layer "${esm_layer}" \
    --hidden_dim "${hidden_dim}"
