#!/bin/bash
set -xe

function main {
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    if [ "${DATASET_DIR}" == "" ] || [ "${CKPT_DIR}" == "" ];then
        set +x
        echo "[ERROR] Please set DATASET_DIR and CKPT_DIR before launch"
        echo "  export DATASET_DIR=/path/to/dataset/dir"
        echo "  export CKPT_DIR=/path/to/checkpoint.pt"
        exit 1
        set -x
    fi
    pip install -r requirements.txt
    if [ "${device}" == "cuda" ];then
        pip install opencv-python==4.8.0.74
    fi	

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # cache
        if [ "${device}" != "cuda" ];then
            device_id='cpu'
        else
            device_id=$(nvidia-smi -L | grep "$CUDA_VISIBLE_DEVICES" -B 50 |grep '^GPU' |tail -1 |sed 's/:.*//;s/[^0-9]//g')
        fi
        python detect.py --nosave --device $device_id \
            --weights $CKPT_DIR --source $DATASET_DIR \
            --num_iter 3 --num_warmup 1 \
            --channels_last $channels_last --precision $precision --device_oob $device \
            ${addtion_options}
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean
            # generate launch script for multiple instance
            if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
                generate_core_launcher
            else
                generate_core
            fi
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
            device_id=$(nvidia-smi -L | grep "$CUDA_VISIBLE_DEVICES" -B 50 |grep '^GPU' |tail -1 |sed 's/:.*//;s/[^0-9]//g')
        fi
        printf " ${OOB_EXEC_HEADER} \
            python detect.py --nosave --device $device_id \
                --weights $CKPT_DIR --source $DATASET_DIR \
                --num_iter $num_iter --num_warmup $num_warmup \
                --channels_last $channels_last --precision $precision --device_oob $device \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#device_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            detect.py --nosave --device cpu \
                --weights $CKPT_DIR --source $DATASET_DIR \
                --num_iter $num_iter --num_warmup $num_warmup \
                --channels_last $channels_last --precision $precision --device_oob $device \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git

# Start
main "$@"
