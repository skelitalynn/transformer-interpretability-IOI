"""
将本地的 Hugging Face 模型缓存上传到远端服务器
"""
import os
import paramiko
from pathlib import Path

# SSH 配置
ssh_config = {
    # --- 这里是关键的修改 ---
    "hostname": "connect.westb.seetacloud.com",  # 将 "host" 修改为 "hostname"
    "port": 21946,
    "username": "root",
    "password": "7COEO+tzdNN2",
    # -------------------------
    "python_bin": "python",
    "conda_env": "ioi",
    "setup_cmd": "eval \"$(conda shell.bash hook)\""
}

# 本地 Hugging Face 缓存路径
local_cache = Path.home() / ".cache" / "huggingface"
remote_cache = "/root/.cache/huggingface"

# --- 为了代码更健壮，只提取 connect 方法需要的参数 ---
connect_args = {
    "hostname": ssh_config["hostname"],
    "port": ssh_config["port"],
    "username": ssh_config["username"],
    "password": ssh_config["password"],
}
# ----------------------------------------------------

print("连接到远端服务器...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# --- 使用新的参数字典进行连接 ---
ssh.connect(**connect_args, timeout=30)
# ---------------------------------

print("创建远端缓存目录...")
sftp = ssh.open_sftp()
# ... (代码其余部分无需修改) ...
stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_cache}/hub")
stdout.channel.recv_exit_status()

print(f"\n本地缓存路径: {local_cache}")

# 查找 GPT-2 相关的模型文件
gpt2_dirs = []
hub_dir = local_cache / "hub"

if hub_dir.exists():
    for item in hub_dir.iterdir():
        if "gpt2" in item.name.lower():
            gpt2_dirs.append(item)

if not gpt2_dirs:
    print("错误: 本地未找到 GPT-2 缓存")
    print("请先在本地运行一次以下载模型:")
    print("  python -c \"from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('gpt2-small')\"")
    ssh.close()
    exit(1)

print(f"找到 {len(gpt2_dirs)} 个 GPT-2 缓存目录:")
for d in gpt2_dirs:
    print(f"  - {d.name}")

# 上传每个目录
for local_dir in gpt2_dirs:
    remote_dir = f"{remote_cache}/hub/{local_dir.name}"
    print(f"\n上传 {local_dir.name}...")

    # 创建远端目录
    stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_dir}")
    stdout.channel.recv_exit_status()

    # 递归上传文件
    def upload_directory(local_path, remote_path):
        for item in local_path.iterdir():
            local_item = local_path / item.name
            remote_item = f"{remote_path}/{item.name}"

            if item.is_file():
                try:
                    file_size = item.stat().st_size
                    print(f"  上传: {item.name} ({file_size/1024/1024:.1f} MB)")
                    sftp.put(str(item), remote_item)
                except Exception as e:
                    print(f"  跳过 {item.name}: {e}")
            elif item.is_dir():
                stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_item}")
                stdout.channel.recv_exit_status()
                upload_directory(item, remote_item)

    upload_directory(local_dir, remote_dir)

sftp.close()
ssh.close()

print("\n✓ 模型缓存上传完成！")
