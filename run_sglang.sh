#!/bin/bash

# 在本地执行此脚本，确保已配置SSH免密登录到两个节点

# 定义节点信息

nodes=("root@14.103.168.91"
       "root@14.103.172.125")

# 公共参数
common_cmd="source /media/nvme/anaconda3/etc/profile.d/conda.sh; \
conda activate sglang; \
export GLOO_SOCKET_IFNAME=eth0; \
export SGL_ENABLE_JIT_DEEPGEMM=0; \
nohup python3 -m sglang.launch_server \
--model-path /media/nvme/deepseek/DeepSeek-V3-0324 \
--tensor-parallel-size 16 \
--dist-init-addr 172.31.16.2:30001 \
--trust-remote-code \
--attention-backend flashinfer \
--data-parallel-size 4 \
--enable-dp-attention \
--schedule-conservativeness 0.01 \
--cuda-graph-max-bs 4096 \
--tool-call-parser deepseekv3 \
--grammar-backend outlines \
--chat-template /media/nvme/workspace/sglang-latest/examples/chat_template/tool_chat_template_deepseekv3.jinja \
--enable-metrics \
--enable-cache-report \
--port 50000 \
--host 0.0.0.0"


# 清理函数
clean_node() {
  # 清理端口占用
  echo "[$1] 清理端口占用..."
  ssh $1 "fuser -k 30001/tcp 2>/dev/null || kill -9 \$(lsof -t -i:30001) 2>/dev/null"
  ssh $1 "fuser -k 50000/tcp 2>/dev/null || kill -9 \$(lsof -t -i:50000) 2>/dev/null"
  
  # 清理GPU进程（匹配sglang相关进程）
  echo "[$1] 清理GPU进程..."
  ssh $1 "ps -ef | grep 'python3 -m sglang.launch_server' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || :"
  
  # 二次清理残留CUDA进程
  echo "[$1] 清理残留CUDA进程..."
  ssh $1 "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || :"
}

# 启动节点服务
start_node() {
  local node=$1
  local rank=$2
  clean_node $node
  echo "[$node] 启动服务进程..."
  ssh $node "$common_cmd  --node-rank $rank > /var/log/node$rank.log 2>&1 &"
}

# 并行启动节点
for i in "${!nodes[@]}"; do
    start_node "${nodes[$i]}" $i  &
    echo -e "\n\033[32mAll nodes started successfully!\033[0m"
    echo "Node$i logs: ssh "${nodes[$i]}" 'tail -f /var/log/node$i.log'"
  done

echo "logs: 'tail -f /var/log/node0.log'"


# 等待后台任务提交完成
wait

echo -e "\n\033[32mBoth nodes started successfully!\033[0m"
echo "Node1 logs: ssh ${nodes[0]} 'tail -f /var/log/node0.log'"
