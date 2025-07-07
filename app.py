from flask import Flask, request, render_template, jsonify
from collections import Counter
from pathlib import Path
import subprocess
import threading
import json
import os

app = Flask(__name__)
last_used_nodes = []
test_running = False

# ----------- DEFAULT VALUES FOR FORM -----------

default_nodes = "root@14.103.168.91|root@14.103.172.125"

default_params = (
    "--attention-backend flashinfer|"
    "--data-parallel-size n --enable-dp-attention|"
    "--data-parallel-size 2 --enable-dp-attention|"
    "--data-parallel-size 4 --enable-dp-attention|"
    "--data-parallel-size 8 --enable-dp-attention"
)

default_export_cmd = (
    "export GLOO_SOCKET_IFNAME=eth0|"
    "export GLOO_SOCKET_IFNAME=eth0 export SGL_ENABLE_JIT_DEEPGEMM=0|"
    "export GLOO_SOCKET_IFNAME=eth0 export SGLANG_HACK_DEEPEP_NEW_MODE=0|"
    "export GLOO_SOCKET_IFNAME=eth0"
)

default_common_cmd = """source /media/nvme/anaconda3/etc/profile.d/conda.sh; \
conda activate sglang; \
nohup python3 -m sglang.launch_server \
--model-path /media/nvme/deepseek/DeepSeek-V3-0324 \
--tensor-parallel-size ${#nodes[@]}*8 \
--dist-init-addr 172.31.16.2:30001 \
--trust-remote-code \
--nnodes ${#nodes[@]} \
--tool-call-parser deepseekv3 \
--grammar-backend outlines \
--chat-template /media/nvme/workspace/sglang-latest/examples/chat_template/tool_chat_template_deepseekv3.jinja \
--enable-metrics \
--enable-cache-report \
--port 50000 \
--host 0.0.0.0"""

default_test_cmd = """source /media/nvme/anaconda3/etc/profile.d/conda.sh; \
conda activate sglang; \
nohup python3 -m sglang.bench_serving \
--backend sglang \
--dataset-path /media/nvme/ShareGPT_V3_unfiltered_cleaned_split.json \
--dataset-name sharegpt \
--num-prompts 10 \
--max-concurrency 10 \
--port 50000"""

default_log_timeout = "300"
default_test_timeout = "2400"

# ----------- SCRIPT RUNNER -----------

def run_script(common_cmd, test_cmd, nodes, params, export_cmd, log_timeout, test_timeout):
    global test_running
    test_running = True

    try:    
        env = os.environ.copy()
        env['COMMON_CMD_ENV'] = common_cmd
        env['TEST_CMD_ENV'] = test_cmd
        env['NODES_ENV'] = nodes
        env['PARAMS_ENV'] = params
        env['EXPORT_CMD_ENV'] = export_cmd
        env['LOG_TIMEOUT_ENV'] = log_timeout
        env['TEST_TIMEOUT_ENV'] = test_timeout

        command = ['./run_sglang_muti_nodes_test.sh']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        process.wait()

    finally:
        test_running = False

# ----------- ROUTES -----------

@app.route('/')
def index():
    node_list = [node.strip() for node in default_nodes.split('|') if node.strip()]
    return render_template('index.html',
                           default_nodes=default_nodes,
                           default_common_cmd=default_common_cmd,
                           default_test_cmd=default_test_cmd,
                           default_params=default_params,
                           default_export_cmd=default_export_cmd,
                           default_log_timeout=default_log_timeout,
                           default_test_timeout=default_test_timeout,
                           node_list=node_list)

@app.route('/run', methods=['POST'])
def run():
    global last_used_nodes
    common_cmd = request.form.get("common_cmd", default_common_cmd)
    test_cmd = request.form.get("test_cmd", default_test_cmd)
    nodes = request.form.get("nodes", default_nodes)
    params = request.form.get("params", default_params)
    export_cmd = request.form.get("export_cmd", default_export_cmd)
    log_timeout = request.form.get("log_timeout", default_log_timeout)
    test_timeout = request.form.get("test_timeout", default_test_timeout)

    last_used_nodes = [n.strip() for n in nodes.split('|') if n.strip()]

    thread = threading.Thread(target=run_script, args=(common_cmd, test_cmd, nodes, params, export_cmd, log_timeout, test_timeout))
    thread.start()
    return jsonify({"status": "started"})

@app.route('/clean', methods=['POST'])
def clean():
    nodes_input = request.form.get("nodes", default_nodes)
    env = os.environ.copy()
    env['NODES_ENV'] = nodes_input

    command = ['./run_sglang_muti_nodes_test.sh', 'clean']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    return jsonify({"status": "cleaning_complete"})

@app.route('/status')
def status():
    return jsonify({"running": test_running})

@app.route('/logs')
def get_logs():
    global last_used_nodes
    requested_node = request.args.get("node")
    if not requested_node:
        return "Please specify a node", 400

    node_list = last_used_nodes if last_used_nodes else [n.strip() for n in default_nodes.split('|') if n.strip()]

    try:
        node_index = node_list.index(requested_node)
        log_file = f"/var/log/node{node_index}.log"
    except ValueError:
        return f"Node '{requested_node}' not found in current nodes. {node_list}", 400

    try:
        result = subprocess.run(
            ['ssh', requested_node, f'cat {log_file}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return f"Error retrieving log from node: {result.stderr}", 500

        return result.stdout
    except Exception as e:
        return f"Exception while retrieving log: {e}", 500
    

@app.route('/test-log')
def get_test_log():
    import subprocess

    # nodes = default_nodes.split('|')
    # node = nodes[0]
    node = (last_used_nodes or default_nodes.split('|'))[0]

    try:
        result = subprocess.run(
            ['ssh', node, 'cat /var/log/test0.log'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        return jsonify({"log": result.stdout})
    except Exception as e:
        return jsonify({"log": f"Failed to retrieve test log: {e}"}), 500
    
@app.route('/test_results')
def get_test_results():
    import subprocess

    node = (last_used_nodes or default_nodes.split('|'))[0]

    try:
        ssh_cmd = f'ssh {node} "cat /var/log/test_sglang/*.jsonl"'
        result = subprocess.run(
            ssh_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        lines = result.stdout.strip().splitlines()
        results = []

        for line in lines:
            try:
                result = json.loads(line)

                param = result.get("param", "")
                export_cmd = result.get("export_cmd", "")
                result["config"] = f"{param} + {export_cmd}"

                def clean_value(val):
                    if isinstance(val, float):
                        if val == float('inf'):
                            return "Infinity"
                        elif val == float('-inf'):
                            return "-Infinity"
                        elif val != val:  # NaN
                            return "NaN"
                    return val

                cleaned = {k: clean_value(v) for k, v in result.items()}
                results.append(cleaned)

            except json.JSONDecodeError:
                continue

        return app.response_class(
            response=json.dumps(results, ensure_ascii=False),
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/failure_counts')
def get_failure_counts():
    import subprocess

    node = (last_used_nodes or default_nodes.split('|'))[0]

    try:
        result = subprocess.run(
            ['ssh', node, 'cat /var/log/test_sglang/failures.json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        fails = json.loads(result.stdout)
        counter = Counter(fails)
        return jsonify(counter)
    except FileNotFoundError:
        return jsonify({})
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------- MAIN -----------
if __name__ == "__main__":
    app.run(debug=True)
