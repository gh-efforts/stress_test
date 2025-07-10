#!/bin/bash

# 自动记录所有输出到日志文件
LOG_FILE="run_sglang_test.log"
#exec > >(tee -a "$LOG_FILE") 2>&1
exec > "$LOG_FILE" 2>&1

echo "脚本开始执行..."

# 节点信息
nodes=("root@10.122.181.97"
       "root@10.122.181.32")

# /media/nvme/models/deepseek-ai_DeepSeek-Coder-V2-Lite-Instruct
# /media/nvme/deepseek/DeepSeek-V3-0324 
# 公共命令
common_cmd="conda activate sglang; \
source /root/anaconda3/etc/profile.d/conda.sh; \
conda activate sglang; \
nohup python3 -m sglang.launch_server \
--model-path openchat/openchat_3.5 \
--tensor-parallel-size 16 \
--dist-init-addr 10.122.181.97:30001 \
--trust-remote-code \
--nnodes 2 \
--port 50000 \
--host 0.0.0.0 "

# 测试命令
test_command="conda activate sglang; \
source /root/anaconda3/etc/profile.d/conda.sh; \
conda activate sglang; \
nohup python3 -m sglang.bench_serving \
--backend sglang \
--dataset-path /media/nvme/ShareGPT_V3_unfiltered_cleaned_split.json \
--dataset-name sharegpt \
--num-prompts 5000 \
--max-concurrency 5000 \
--port 50000"

# 参数列表(在这里修改可变参数)
params=(
  "--schedule-policy lpm"
  "--schedule-policy random"
  "--schedule-policy fcfs"
  "--schedule-policy dfs-weight"
  "--attention-backend aiter"
  "--attention-backend fa3"
  "--attention-backend flashmla"
  "--attention-backend flashinfer"
  "--attention-backend intel_amx"
  "--attention-backend triton"
  "--attention-backend torch_native"
  "--attention-backend cutlass_mla"
  "--data-parallel-size 16 --enable-dp-attention --load-balance-method round_robin"
  "--data-parallel-size 16 --enable-dp-attention --load-balance-method round_robin shortest_queue"
)
export_cmd=(
  "export NCCL_DEBUG=INFO"
  )
fail_list=()

# 清理函数
clean_node() {
  # 清理端口占用
  echo "[$1] 清理端口占用..."
  ssh $1 "fuser -k 30001/tcp 2>/dev/null || kill -9 \$(lsof -t -i:30001) 2>/dev/null"
  ssh $1 "fuser -k 30002/tcp 2>/dev/null || kill -9 \$(lsof -t -i:30001) 2>/dev/null"
  ssh $1 "fuser -k 30006/tcp 2>/dev/null || kill -9 \$(lsof -t -i:30001) 2>/dev/null"
  ssh $1 "fuser -k 50000/tcp 2>/dev/null || kill -9 \$(lsof -t -i:50000) 2>/dev/null"
  
  # 清理GPU进程（匹配sglang相关进程）
  echo "[$1] 清理GPU进程..."
  ssh $1 "ps -ef | grep 'python3 -m sglang.launch_server' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || :"
  ssh $1 "ps -ef | grep 'run_sglang_muti_nodes_test.sh' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || :"
  
  # 二次清理残留CUDA进程
  echo "[$1] 清理残留CUDA进程..."
  ssh $1 "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || :"
}

# 等待“ready to roll”并触发测试
wait_for_ready_and_run_test() {
  local log_file="/var/log/node0.log"
  local param_key=$(echo "$1" | tr ' -' _)
  ssh ${nodes[0]} "mkdir -p test_sglang"
  local json_file="/var/log/test_sglang/${param_key}.jsonl"
  echo "$json_file"
  #timeout可能需要修改
  local timeout_sec=300 

  echo "[Node0] 开始监听日志中的 'The server is fired up and ready to roll!超时：300s"

  while true; do
    # 使用带超时的read监控日志更新
    if read -t $timeout_sec line < <(ssh ${nodes[0]} "tail -n 0 -F $log_file" 2>/dev/null); then
      echo "[日志] $line"
      
      # 检测就绪消息
      if timeout 2 ssh ${nodes[0]} "tail -F $log_file" | grep -m 1 "The server is fired up and ready to roll!"; then
        echo -e "\033[32m检测到服务就绪，开始测试...\033[0m"
        # 启动测试命令，后台运行
        ssh ${nodes[0]} "$test_command --output-file $json_file > /var/log/test0.log 2>&1 &"
        return 0
      fi
    else
      echo -e "\033[31m错误：等待服务启动超时。参数：$1\033[0m"
      return 1
    fi

  done
}

# 启动节点服务
start_node() {
  local export_cmd=$1
  local node=$2
  local rank=$3
  local param=$4
  clean_node $node
  echo "[$node] 启动服务进程... 参数：$param"
  ssh $node "$export_cmd $common_cmd  --node-rank $rank $param > /var/log/node$rank.log 2>&1 &"
}

# 取最大长度进行遍历
max_idx=$(( ${#params[@]} > ${#export_cmd[@]} ? ${#params[@]} : ${#export_cmd[@]} ))

a=0
b=0
# 遍历参数列表并依次启动服务
for (( idx=0; idx<max_idx; idx++ )); do
  # 获取当前 param，如果不存在就用最后一个
  if (( idx < ${#params[@]} )); then
    param=${params[$idx]}
  else
    param=${params[-1]}
  fi

  # 获取当前 export_str，如果不存在就用最后一个
  if (( idx < ${#export_cmd[@]} )); then
    export_cmd=${export_cmd[$idx]}
  else
    export_cmd=${export_cmd[-1]}
  fi

  echo "-------------------- 正在测试参数: $param --------------------"

  # 并行启动所有节点
  for i in "${!nodes[@]}"; do
    start_node "$export_cmd" "${nodes[$i]}" $i "$param" &
    echo -e "\n\033[32mAll nodes started successfully!\033[0m"
    echo "Node$i logs: ssh "${nodes[$i]}" 'tail -f /var/log/node$i.log'"
  done

  echo "logs: 'tail -f /var/log/node0.log'"

  # 等待服务就绪或超时
  if wait_for_ready_and_run_test "$param"; then
    a=$((a+1))
    echo "检测测试日志中... "
    test_file="/var/log/test0.log"
    #if timeout 2000 ssh ${nodes[0]} "tail -F $test_file" | grep -m 1 "Serving Benchmark Result"; then
    while true; do
      # 使用带超时的read监控日志更新
      if read -t $log_timeout line < <(ssh ${nodes[0]} "tail -n 0 -F "/var/log/node0.log"" 2>/dev/null); then
        echo "[日志] $line"
        # 检测测试状态
        if timeout $test_timeout ssh ${nodes[0]} "tail -F "/var/log/test0.log"" | grep -m 1 "Serving Benchmark Result"; then
          echo "测试已经结束，10s后关闭服务..."
          sleep 10

          # Append export_cmd to the last JSON line
          param_key=$(echo "$param" | tr ' -' _)
          json_file="/var/log/test_sglang/${param_key}.jsonl"

          # Modify the last line: append export_cmd to JSON
          ssh ${nodes[0]} "tmpfile=\$(mktemp); \
            head -n -1 $json_file > \$tmpfile || true; \
            last_line=\$(tail -n 1 $json_file); \
            python3 -c \"import json; \
            line=json.loads('\$last_line'); \
            line.update({'export_cmd': '$export_cmd', 'param': '$param'}); \
            print(json.dumps(line))\" >> \$tmpfile; \
            mv \$tmpfile $json_file"
          break
        fi
      else
        b=$((b+1))
        fail_list+=("$param")
        echo -e "\033[31m服务启动失败，清理并继续下一个参数...\033[0m"
        # clean_node ${nodes[0]}
        break
      fi
    done
    continue
  else
    b=$((b+1))
    fail_list+=("$param")
    echo -e "\033[31m服务启动失败，清理并继续下一个参数...\033[0m"
    # clean_node ${nodes[0]}
    continue
  fi
  
done

# 等待后台任务提交完成
echo -e "\n\033[32mSuccess:$a;Fail:$b,Fail list:"${fail_list[@]}"\033[0m"
echo -e "\n\033[32m测试数据保存在“/root/test_sglang/”文件夹下\033[0m"
echo -e "\n\033[32m所有参数的服务测试完成！\033[0m"

# echo "[$1] 清理run_sglang_muti_nodes_test.sh进程..."
ssh ${nodes[0]} "ps -ef | grep 'run_sglang_muti_nodes_test.sh' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || :"