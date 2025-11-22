"""
IOI 混合云编排器
根据 hybrid_config.json 中的 execution 配置，灵活选择每个GPU环节在本地或远端执行
记录详细计时（包含通信时间），输出格式与 IOI.ipynb 一致
"""
import os
import json
import time
import subprocess
from typing import Dict, Any

try:
    import paramiko
except ImportError:
    paramiko = None


def load_config(path: str = "hybrid_config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_local_task(task: str, input_file: str, output_file: str, timing_file: str = "timing_tmp.json") -> dict:
    """在本地执行任务"""
    cmd = ["python", "ioi_modules.py", "--task", task, "--input", input_file, "--output", output_file, "--timing-output", timing_file]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"本地任务 {task} 执行失败")
    
    with open(timing_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    os.remove(timing_file)
    return result


def ssh_connect(cfg: dict, max_retries: int = 3):
    if paramiko is None:
        raise RuntimeError("需要安装 paramiko: pip install paramiko")
    
    pkey = None
    if cfg.get("pkey_path"):
        pkey_path = os.path.expanduser(cfg["pkey_path"])
        if os.path.exists(pkey_path):
            pkey = paramiko.RSAKey.from_private_key_file(pkey_path)
    
    password = cfg.get("password") or None
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=cfg["host"],
                port=cfg.get("port", 22),
                username=cfg["username"],
                pkey=pkey,
                password=password,
                timeout=30,
            )
            print(f"SSH连接成功！")
            return ssh
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"连接失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"等待3秒后重试...")
                import time
                time.sleep(3)
            else:
                print(f"SSH连接失败，已重试 {max_retries} 次")
                raise


def sftp_put(ssh, local_path: str, remote_path: str):
    sftp = ssh.open_sftp()
    try:
        sftp.put(local_path, remote_path)
    finally:
        sftp.close()


def sftp_get(ssh, remote_path: str, local_path: str):
    sftp = ssh.open_sftp()
    try:
        sftp.get(remote_path, local_path)
    finally:
        sftp.close()


def ssh_run(ssh, command: str, cwd: str = None):
    # 注意：这里的环境初始化对于使用绝对路径的python来说不是必须的，但保留也无害
    command = f"source /etc/profile && source ~/.bashrc && {command}"
    if cwd:
        command = f"cd {cwd} && {command}"
    stdin, stdout, stderr = ssh.exec_command(command)
    rc = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    if rc != 0:
        print(f"远端命令失败:\n{command}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
        raise RuntimeError(f"远端命令返回码 {rc}")
    return rc, out, err


def ensure_remote_dir(ssh, path: str):
    ssh_run(ssh, f"mkdir -p {path}")


def run_remote_task(ssh, cfg: dict, task: str, local_input: str, local_output: str, remote_timing: str = "timing_remote_tmp.json") -> dict:
    """在远端执行任务，返回 {task_time, upload_time, download_time}"""
    # --- 添加诊断日志 ---
    print(f"\n--- 开始远程任务: {task} ---")

    ssh_cfg = cfg["ssh"]
    paths = cfg["paths"]
    remote_dir = paths["remote_workdir"]
    
    python_absolute_path = "/root/miniconda3/envs/ioi/bin/python"
    
    setup_cmd = ssh_cfg.get("setup_cmd", "")
    
    print(f"[诊断日志] 确保远程目录存在: {remote_dir}")
    ensure_remote_dir(ssh, remote_dir)
    print(f"[诊断日志] ...目录已确认。")

    # 1) 上传输入文件
    t_up_0 = time.time()
    remote_input = f"{remote_dir}/{os.path.basename(local_input)}"
    print(f"[诊断日志] 步骤 1/4: 正在上传输入文件 '{local_input}' -> '{remote_input}'...")
    sftp_put(ssh, local_input, remote_input)
    t_up_1 = time.time()
    upload_time = t_up_1 - t_up_0
    print(f"[诊断日志] ...输入文件上传完成 (耗时 {upload_time:.2f}s)。")

    # 2) 上传 ioi_modules.py
    remote_script_name = "ioi_modules.py" # 确保这个是您在远端要执行的脚本名
    remote_script_path = f"{remote_dir}/{remote_script_name}"
    print(f"[诊断日志] 步骤 2/4: 正在上传执行脚本 '{remote_script_name}' -> '{remote_script_path}'...")
    try:
        sftp_put(ssh, remote_script_name, remote_script_path)
        print(f"[诊断日志] ...执行脚本上传完成。")
    except Exception as e:
        print(f"[诊断日志] ...上传脚本失败或文件已存在，跳过。错误: {e}")
        pass
    
    # 3) 执行远端任务
    remote_output = f"{remote_dir}/{os.path.basename(local_output)}"
    
    # --- 核心修改 2: 使用绝对路径构建命令 ---
    remote_cmd = (
        f"{python_absolute_path} {remote_script_name} "
        f"--task {task} "
        f"--input {os.path.basename(local_input)} "
        f"--output {os.path.basename(local_output)} "
        f"--timing-output {remote_timing}"
    )

    if setup_cmd:
        remote_cmd = f"{setup_cmd} && {remote_cmd}"
    
    print(f"[诊断日志] 步骤 3/4: 准备执行远程命令。程序将在此等待，直到命令完成...")
    print(f"    远程工作目录: {remote_dir}")
    print(f"    将要执行的命令: {remote_cmd}")
    
    ssh_run(ssh, remote_cmd, cwd=remote_dir)
    
    print(f"[诊断日志] ...远程命令执行完毕！")

    # 4) 下载输出文件
    t_down_0 = time.time()
    print(f"[诊断日志] 步骤 4/4: 正在下载输出文件 '{remote_output}' -> '{local_output}'...")
    sftp_get(ssh, remote_output, local_output)
    t_down_1 = time.time()
    download_time = t_down_1 - t_down_0
    print(f"[诊断日志] ...输出文件下载完成 (耗时 {download_time:.2f}s)。")
    
    # 5) 下载计时文件
    sftp_get(ssh, f"{remote_dir}/{remote_timing}", remote_timing)
    with open(remote_timing, 'r', encoding='utf-8') as f:
        task_result = json.load(f)
    os.remove(remote_timing)
    
    print(f"--- 远程任务 {task} 完成 ---")
    
    return {
        "task_time": task_result["time"],
        "upload_time": upload_time,
        "download_time": download_time,
        "total_time": task_result["time"] + upload_time + download_time,
        "meta": task_result
    }


def run_orchestrator(config_path: str = "hybrid_config.json"):
    cfg = load_config(config_path)
    execution = cfg["execution"]
    paths = cfg["paths"]
    
    timing_report = {}
    ssh_conn = None
    
    # 需要远端执行时建立SSH连接
    need_remote = any(loc == "remote" for loc in execution.values())
    if need_remote:
        print("检测到需要远程执行的任务，正在建立SSH连接...")
        ssh_conn = ssh_connect(cfg["ssh"])
        print("SSH连接成功！\n")
    
    try:
        # 前置本地步骤（generate + check）
        print("=" * 60)
        print("阶段 0a: 生成数据")
        print("=" * 60)
        t0_gen_wall = time.time()
        subprocess.run(["python", "ioi_local_pre.py", "--step", "generate"], check=True)
        timing_report["generate_data_wall_time"] = time.time() - t0_gen_wall
        
        # 读取内部纯计算时间
        try:
            with open("local_pre_timing.json", 'r', encoding='utf-8') as f:
                pre_timing = json.load(f)
            timing_report["generate_data_time"] = pre_timing.get("local_generate_s", 0)
        except:
            pass
        
        print()
        
        print("=" * 60)
        print("阶段 0b: 结构校验")
        print("=" * 60)
        t0_check_wall = time.time()
        subprocess.run(["python", "ioi_local_pre.py", "--step", "check"], check=True)
        timing_report["check_structure_wall_time"] = time.time() - t0_check_wall
        
        # 读取内部纯计算时间
        try:
            with open("local_pre_timing.json", 'r', encoding='utf-8') as f:
                pre_timing = json.load(f)
            timing_report["check_structure_time"] = pre_timing.get("local_check_s", 0)
            os.remove("local_pre_timing.json")
        except:
            pass
        
        print()
        
        # 任务1: filter_gpt2
        print("=" * 60)
        print("阶段 1: GPT-2 样本筛选")
        print("=" * 60)
        t0_filter_wall = time.time()
        if execution["filter_gpt2"] == "local":
            print("[本地执行]")
            result = run_local_task("filter", paths["local_data_check1"], paths["local_data_check2"])
            timing_report["filter_gpt2_time"] = result["time"]
            timing_report["filter_gpt2_wall_time"] = time.time() - t0_filter_wall
            timing_report["filter_gpt2_location"] = "local"
        else:
            print("[远端执行]")
            result = run_remote_task(ssh_conn, cfg, "filter", paths["local_data_check1"], paths["local_data_check2"])
            timing_report["filter_gpt2_time"] = result["task_time"]
            timing_report["filter_gpt2_upload_time"] = result["upload_time"]
            timing_report["filter_gpt2_download_time"] = result["download_time"]
            timing_report["filter_gpt2_total_time"] = result["total_time"]
            timing_report["filter_gpt2_wall_time"] = time.time() - t0_filter_wall
            timing_report["filter_gpt2_location"] = "remote"
        print()
        
        # 任务2: collect_activations
        print("=" * 60)
        print("阶段 2: 缓存激活值")
        print("=" * 60)
        t0_collect_wall = time.time()
        if execution["collect_activations"] == "local":
            print("[本地执行]")
            result = run_local_task("collect", paths["local_data_check2"], paths["local_saved"])
            timing_report["collect_activations_time"] = result["time"]
            timing_report["collect_activations_wall_time"] = time.time() - t0_collect_wall
            timing_report["collect_activations_location"] = "local"
        else:
            print("[远端执行]")
            result = run_remote_task(ssh_conn, cfg, "collect", paths["local_data_check2"], paths["local_saved"])
            timing_report["collect_activations_time"] = result["task_time"]
            timing_report["collect_activations_upload_time"] = result["upload_time"]
            timing_report["collect_activations_download_time"] = result["download_time"]
            timing_report["collect_activations_total_time"] = result["total_time"]
            timing_report["collect_activations_wall_time"] = time.time() - t0_collect_wall
            timing_report["collect_activations_location"] = "remote"
        print()
        
        # 任务3: patch_activations
        print("=" * 60)
        print("阶段 3: 修补激活值")
        print("=" * 60)
        t0_patch_wall = time.time()
        if execution["patch_activations"] == "local":
            print("[本地执行]")
            result = run_local_task("patch", paths["local_saved"], paths["local_results"])
            timing_report["patch_activations_time"] = result["time"]
            timing_report["patch_activations_wall_time"] = time.time() - t0_patch_wall
            timing_report["patch_activations_location"] = "local"
        else:
            print("[远端执行]")
            result = run_remote_task(ssh_conn, cfg, "patch", paths["local_saved"], paths["local_results"])
            timing_report["patch_activations_time"] = result["task_time"]
            timing_report["patch_activations_upload_time"] = result["upload_time"]
            timing_report["patch_activations_download_time"] = result["download_time"]
            timing_report["patch_activations_total_time"] = result["total_time"]
            timing_report["patch_activations_wall_time"] = time.time() - t0_patch_wall
            timing_report["patch_activations_location"] = "remote"
        print()
        
        # 任务4: plot_heatmap
        print("=" * 60)
        print("阶段 4: 绘制热力图")
        print("=" * 60)
        t0_plot_wall = time.time()
        if execution["plot_heatmap"] == "local":
            print("[本地执行]")
            result = run_local_task("plot", paths["local_results"], paths["local_heatmap"])
            timing_report["plot_heatmap_time"] = result["time"]
            timing_report["plot_heatmap_wall_time"] = time.time() - t0_plot_wall
            timing_report["plot_heatmap_location"] = "local"
        else:
            print("[远端执行]")
            result = run_remote_task(ssh_conn, cfg, "plot", paths["local_results"], paths["local_heatmap"])
            timing_report["plot_heatmap_time"] = result["task_time"]
            timing_report["plot_heatmap_upload_time"] = result["upload_time"]
            timing_report["plot_heatmap_download_time"] = result["download_time"]
            timing_report["plot_heatmap_total_time"] = result["total_time"]
            timing_report["plot_heatmap_wall_time"] = time.time() - t0_plot_wall
            timing_report["plot_heatmap_location"] = "remote"
        print()
        
    finally:
        if ssh_conn:
            ssh_conn.close()
    
    # 保存计时报告
    with open(paths["timing_report"], 'w', encoding='utf-8') as f:
        json.dump(timing_report, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("全流程完成！计时报告已保存至", paths["timing_report"])
    print("=" * 60)
    
    # 打印汇总
    print("\n计时汇总：")
    for k, v in timing_report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}s")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hybrid_config.json", help="配置文件路径")
    args = parser.parse_args()
    
    run_orchestrator(args.config)