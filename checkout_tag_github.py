import os
import sys
import subprocess
import threading
import logging
import configparser
import json
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Callable, Any, NamedTuple, Union
import queue
import time
import webbrowser
from urllib.parse import urlparse
import re
import requests
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, END, filedialog
from tkinter.font import Font

# --- 版本与更新配置 ---
APP_VERSION = "5.0.0"
UPDATE_URL = "https://api.github.com/repos/LabiInYour/git_checkout_tool/releases/latest" #!TODO: 请替换为你的GitHub仓库地址

# --- 主题、配置和基本数据结构 ---

# 现代主题颜色配置
class ThemeColors:
    PRIMARY = "#2c3e50"
    PRIMARY_LIGHT = "#34495e"
    ACCENT = "#3498db"
    ACCENT_LIGHT = "#5dade2"
    SUCCESS = "#2ecc71"
    SUCCESS_LIGHT = "#58d68d"
    WARNING = "#f39c12"
    WARNING_LIGHT = "#f8c471"
    DANGER = "#e74c3c"
    DANGER_LIGHT = "#ec7063"
    BACKGROUND = "#ecf0f1"
    SURFACE = "#ffffff"
    SURFACE_VARIANT = "#f8f9fa"
    ON_SURFACE = "#2c3e50"
    ON_SURFACE_VARIANT = "#7f8c8d"
    BORDER = "#bdc3c7"
    BORDER_LIGHT = "#d5dbdb"
    HOVER = "#ecf0f1"
    PRESSED = "#d5dbdb"

@dataclass
class AppConfig:
    """应用配置数据类"""
    last_ref: str = ""
    last_selected_modules: List[str] = None
    window_geometry: str = "1000x750"
    window_state: str = "normal"
    theme: str = "modern"
    auto_save_logs: bool = True
    max_workers: int = 0

    def __post_init__(self):
        if self.last_selected_modules is None:
            self.last_selected_modules = []

    @classmethod
    def load(cls, config_file: Path) -> 'AppConfig':
        if not config_file.exists():
            return cls()
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"配置文件格式错误，使用默认配置: {e}")
            return cls()

    def save(self, config_file: Path):
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"保存配置失败: {e}")


class GitOperationError(Exception):
    """Git 操作相关异常"""
    def __init__(self, message: str, cmd: Optional[List[str]] = None, repo: Optional[Path] = None):
        super().__init__(message)
        self.cmd = cmd
        self.repo = repo


class RefType(Enum):
    BRANCH = "branch"
    TAG = "tag"
    DETACHED = "detached"


@dataclass
class GitRef:
    name: str
    ref_type: RefType

    def __str__(self) -> str:
        if self.ref_type == RefType.TAG:
            return f"标签: {self.name}"
        if self.ref_type == RefType.DETACHED:
            return "分离 HEAD"
        return f"分支: {self.name}"


@dataclass
class OperationResult:
    success: bool
    message: str
    path: Optional[str] = None
    duration: float = 0.0
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

class ModuleInfo(NamedTuple):
    """用于在线程间传递模块信息的结构体"""
    name: str
    relative_path: str
    current_ref: GitRef
    status_icon: str
    ref_display: str
    commit_hash: Optional[str] = None
    error_message: Optional[str] = None


class GitSubmoduleManager:
    """Git 子模块管理器 - 优化版"""
    EXPECTED_FETCH_CONFIG = '+refs/heads/*:refs/remotes/origin/*'
    REMOTE_SECTION = 'remote "origin"'
    BUILD_COMMANDS = [
        ['py', '-3.7', 'build.py', '-c', '-f'],
        ['py', '-3.7', 'build.py', '-a', 'adsp21593', '-p', 'ML12_695D', '-dl', '-b', 'release', '-l', 'warning', '-pb']
    ]

    def __init__(self, root_dir: Optional[str] = None, config: Optional[AppConfig] = None):
        self.root_dir = Path(root_dir or os.getcwd())
        self.platform_dir = self.root_dir / 'platform'
        self.config = config or AppConfig()
        self.logger = self._setup_logger()
        self._submodules_cache: Optional[List[Path]] = None
        self._last_cache_time = 0
        self.cache_ttl = 300

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            h.setFormatter(fmt)
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger

    def get_submodules(self, force_refresh: bool = False) -> List[Path]:
        current_time = time.time()
        if not force_refresh and self._submodules_cache is not None and current_time - self._last_cache_time < self.cache_ttl:
            return self._submodules_cache
        if not self.platform_dir.exists():
            self.logger.error(f"未找到 platform 目录: {self.platform_dir}")
            return []
        
        submodules = []
        for name in ('ap_blocks', 'components'):
            folder = self.platform_dir / name
            if not folder.exists():
                self.logger.warning(f"目录不存在: {folder}")
                continue
            for item in folder.iterdir():
                if name == 'components' and item.name == 'unit_test':
                    continue
                if item.is_dir() and (item / '.git').exists():
                    submodules.append(item)
        
        self._submodules_cache = sorted(submodules, key=lambda p: p.name)
        self._last_cache_time = current_time
        return self._submodules_cache

    def _run_git_command(self, repo: Path, cmd: List[str], capture_output: bool = True, timeout: int = 30, retry_count: int = 2) -> subprocess.CompletedProcess:
        full = ['git', '-C', str(repo)] + cmd
        retry_delay = 2
        
        actual_timeout = timeout
        if cmd[0] == 'fetch':
            actual_timeout = min(timeout * 4, 120)
        elif cmd[0] in ['clone', 'pull']:
            actual_timeout = min(timeout * 2, 90)
        elif cmd[:2] == ['tag', '--list']:
            actual_timeout = min(timeout, 10)

        for attempt in range(retry_count):
            try:
                start_time = time.time()
                result = subprocess.run(
                    full, capture_output=capture_output, text=True, check=False, 
                    timeout=actual_timeout, encoding='utf-8', errors='replace'
                )
                execution_time = time.time() - start_time
                if execution_time > 5:
                    self.logger.info(f"Git命令耗时 {execution_time:.1f}s: {' '.join(cmd)}")
                return result
            except subprocess.TimeoutExpired:
                if attempt < retry_count - 1:
                    self.logger.warning(f"Git命令超时(重试 {attempt + 1}): {' '.join(cmd)}")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    raise GitOperationError(f"Git 命令超时 (已重试 {retry_count - 1} 次)", cmd=full, repo=repo)
            except subprocess.SubprocessError as e:
                raise GitOperationError(f"Git 命令执行错误: {e}", cmd=full, repo=repo)
        
        raise GitOperationError("Git 命令执行失败", cmd=full, repo=repo)

    def update_fetch_config(self, repo: Path) -> bool:
        cfg = repo / '.git' / 'config'
        cp = configparser.ConfigParser(strict=False)
        try:
            cp.read(cfg)
        except configparser.ParsingError as e:
            self.logger.error(f"解析 config 失败: {cfg}，{e}")
            return False
        if self.REMOTE_SECTION not in cp:
            cp[self.REMOTE_SECTION] = {}
        cur = cp[self.REMOTE_SECTION].get('fetch', '')
        if cur != self.EXPECTED_FETCH_CONFIG:
            cp[self.REMOTE_SECTION]['fetch'] = self.EXPECTED_FETCH_CONFIG
            try:
                with open(cfg, 'w') as f:
                    cp.write(f)
                self.logger.info(f"更新 {repo.name} fetch 配置")
                return True
            except IOError as e:
                self.logger.error(f"写入 config 失败: {cfg}，{e}")
        return False
        
    def get_current_ref(self, repo: Path) -> GitRef:
        try:
            # 1. Check for a branch using symbolic-ref, which is the most reliable way.
            r = self._run_git_command(repo, ['symbolic-ref', '--short', 'HEAD'], timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return GitRef(r.stdout.strip(), RefType.BRANCH)

            # 2. If detached, try to get the name from `git branch` output first.
            r_branch = self._run_git_command(repo, ['branch'], timeout=5)
            if r_branch.returncode == 0:
                for line in r_branch.stdout.strip().splitlines():
                    if line.startswith('* (HEAD detached at'):
                        name = line[line.find(' at ')+4:-1].strip()
                        return GitRef(name, RefType.DETACHED)

            # 3. If not a special detached HEAD, get a descriptive name.
            # This will show tag name if on a tag, or a describe-like name.
            r_desc = self._run_git_command(repo, ['describe', '--all'], timeout=5)
            if r_desc.returncode == 0 and r_desc.stdout.strip():
                name = r_desc.stdout.strip()
                # The output could be 'heads/main' or 'tags/v1.0'. We clean it up.
                if name.startswith('heads/'):
                    name = name[len('heads/'):]
                elif name.startswith('tags/'):
                    name = name[len('tags/'):]
                    # If it's a tag, let's return it as a TAG type
                    return GitRef(name, RefType.TAG)
                return GitRef(name, RefType.DETACHED)

            # 4. As a fallback, get the short commit hash.
            r_hash = self._run_git_command(repo, ['rev-parse', '--short', 'HEAD'], timeout=5)
            if r_hash.returncode == 0 and r_hash.stdout.strip():
                return GitRef(r_hash.stdout.strip(), RefType.DETACHED)

            # 5. As a final fallback.
            return GitRef("无法获取状态", RefType.DETACHED)
        except GitOperationError as e:
            self.logger.error(f"获取 {repo.name} 状态时出错: {e}")
            return GitRef("Error", RefType.DETACHED)


    def switch_ref(self, repo: Path, ref: str) -> OperationResult:
        start_time = time.time()
        try:
            # 1. 先 pull 当前分支
            initial_ref = self.get_current_ref(repo)
            if initial_ref.ref_type == RefType.BRANCH:
                self.logger.info(f"在 {repo.name} 上为当前分支 '{initial_ref.name}' 执行 pull...")
                pull_before_result = self._run_git_command(repo, ['pull'])
                if pull_before_result.returncode != 0:
                    self.logger.warning(f"在 {repo.name} 切换前 pull 失败: {pull_before_result.stderr.strip()}")
                    # 不中断操作，仅记录警告

            # 2. fetch 远程更新
            fetch_result = self._run_git_command(repo, ['fetch', 'origin', '--tags', '--prune', '--no-auto-gc'])
            if fetch_result.returncode != 0:
                self.logger.warning(f"Fetch警告 {repo.name}: {fetch_result.stderr}")
            
            # 3. checkout 目标分支/标签
            co = self._run_git_command(repo, ['checkout', ref])
            if co.returncode != 0:
                return OperationResult(False, f"checkout 失败: {co.stderr.strip()}", str(repo), time.time() - start_time)
            
            # 4. 如果是分支，再 pull 一次
            cur = self.get_current_ref(repo)
            if cur.ref_type == RefType.BRANCH:
                self.logger.info(f"在 {repo.name} 上为新分支 '{cur.name}' 执行 pull...")
                # 使用 cur.name 而不是 ref，因为 ref 可能是 'main'，而实际分支可能是 'origin/main' 解析后的 'main'
                p = self._run_git_command(repo, ['pull', 'origin', cur.name])
                msg = f"切换并 pull 成功 ({cur.name})" if p.returncode == 0 else f"切换成功，但 pull 失败: {p.stderr.strip()}"
                return OperationResult(p.returncode == 0, msg, str(repo), time.time() - start_time)
            elif cur.ref_type == RefType.TAG:
                return OperationResult(True, f"切换到标签: {cur.name}", str(repo), time.time() - start_time)
            else:
                # 分离头模式下，显示更详细的信息
                return OperationResult(True, f"切换到分离 HEAD ({cur.name})", str(repo), time.time() - start_time)
        except GitOperationError as e:
            return OperationResult(False, f"Git 操作异常: {e}", str(repo), time.time() - start_time)
        
    def process_submodule(self, repo: Path, ref: str) -> OperationResult:
        start_time = time.time()
        try:
            self.update_fetch_config(repo)
            result = self.switch_ref(repo, ref)
            result.duration = time.time() - start_time
            return result
        except Exception as e:
            return OperationResult(False, f"异常: {e}", str(repo), time.time() - start_time)

    def check_working_tree_clean(self, repo: Path) -> bool:
        status = self._run_git_command(repo, ['status', '--porcelain'], timeout=10)
        return status.returncode == 0 and not status.stdout.strip()

    def discard_changes(self, repo: Path) -> OperationResult:
        try:
            reset = self._run_git_command(repo, ['reset', '--hard'])
            if reset.returncode != 0: return OperationResult(False, f"重置失败: {reset.stderr.strip()}", str(repo))
            clean = self._run_git_command(repo, ['clean', '-fd'])
            if clean.returncode != 0: return OperationResult(False, f"清理失败: {clean.stderr.strip()}", str(repo))
            return OperationResult(True, "已放弃所有更改", str(repo))
        except GitOperationError as e:
            return OperationResult(False, f"放弃更改时出错: {e}", str(repo))

    def stash_changes(self, repo: Path) -> OperationResult:
        try:
            stash = self._run_git_command(repo, ['stash', 'push', '--include-untracked'])
            if "No local changes to save" in stash.stdout or "No local changes to save" in stash.stderr:
                return OperationResult(True, "无需暂存", str(repo))
            if stash.returncode == 0:
                return OperationResult(True, "已暂存更改", str(repo))
            return OperationResult(False, f"暂存失败: {stash.stderr.strip()}", str(repo))
        except GitOperationError as e:
            return OperationResult(False, f"暂存更改时出错: {e}", str(repo))

    def _execute_parallel_operation(self, subs: List[Path], operation_func: Callable,
                                    progress_callback: Optional[Callable] = None, **kwargs) -> Dict[str, Any]:
        res = {'success': [], 'failure': [], 'dirty': []}
        total = len(subs)

        self.logger.info("开始预检查工作区状态...")
        dirty_repos = []
        max_workers_check = self.config.max_workers or min(total, 16)
        with ThreadPoolExecutor(max_workers=max_workers_check) as exe:
            futs = {exe.submit(self.check_working_tree_clean, s): s for s in subs}
            for i, fut in enumerate(as_completed(futs)):
                repo = futs[fut]
                if progress_callback:
                    progress = int((i + 1) / total * 20)
                    progress_callback(progress, f"检查状态: {repo.name}...")
                if not fut.result():
                    dirty_repos.append(repo)
        
        if dirty_repos:
            res['dirty'] = dirty_repos
            return res

        self.logger.info("预检查通过, 开始执行核心操作...")
        max_workers_op = self.config.max_workers or min(total, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=max_workers_op) as exe:
            futs = {exe.submit(operation_func, s, **kwargs): s for s in subs}
            completed = 0
            for fut in as_completed(futs):
                completed += 1
                repo = futs[fut]
                if progress_callback:
                    progress = 20 + int(completed / total * 80)
                    progress_callback(progress, f"处理中: {repo.name}...")
                try:
                    r = fut.result()
                    if r.success:
                        res['success'].append(r)
                        self.logger.info(f"✓ {repo.name}: {r.message} ({r.duration:.1f}s)")
                    else:
                        res['failure'].append(r)
                        self.logger.error(f"✗ {repo.name}: {r.message} ({r.duration:.1f}s)")
                except Exception as e:
                    error_result = OperationResult(False, f"异常: {e}", str(repo))
                    res['failure'].append(error_result)
                    self.logger.error(f"✗ {repo.name}: 异常 {e}")
        return res

    def build_project(self, progress_callback=None) -> OperationResult:
        if not self.platform_dir.exists():
            return OperationResult(False, f"platform 目录不存在: {self.platform_dir}")
        bp = self.root_dir / 'build.py'
        if not bp.exists():
            return OperationResult(False, f"找不到 build.py: {bp}")
        
        result_file = self.root_dir / 'result.txt'
        
        # 初始化进度
        if progress_callback:
            progress_callback(5, "开始编译...")
        
        try:
            # 执行第一条命令
            if progress_callback:
                progress_callback(10, f"运行: {' '.join(self.BUILD_COMMANDS[0])}")
            subprocess.run(self.BUILD_COMMANDS[0], cwd=self.root_dir, check=True)
            
            # 执行第二条命令并捕获输出
            if progress_callback:
                progress_callback(15, f"运行: {' '.join(self.BUILD_COMMANDS[1])}")
            
            with open(result_file, 'w', encoding='utf-8') as f:
                process = subprocess.Popen(
                    self.BUILD_COMMANDS[1], cwd=self.root_dir, stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8'
                )
                
                # 实时读取输出并解析进度信息
                total_projects = 140  # 默认总项目数，实际会从输出中获取
                
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    line = line.strip()
                    print(line)
                    f.write(line + '\n')
                    f.flush()
                    
                    # 检查是否包含进度信息
                    if "Processing project [" in line and "]:" in line:
                        # 解析进度信息，例如: INFO: -- -- Processing project [4/140]: c_sv_utility
                        try:
                            # 提取 [4/140] 部分
                            start = line.index("[")
                            end = line.index("]", start)
                            progress_part = line[start+1:end]  # 得到 "4/140"
                            current, total = map(int, progress_part.split("/"))
                            total_projects = total
                            
                            # 计算进度百分比 (15% 到 90% 之间)
                            progress = 15 + int((current / total_projects) * 75)
                            project_name = line.split("]:")[-1].strip().split()[0]  # 提取项目名
                            
                            if progress_callback:
                                progress_callback(min(90, progress), f"编译项目 [{current}/{total_projects}]: {project_name}")
                        except (ValueError, IndexError):
                            # 如果解析失败，跳过这一行
                            pass
                
                returncode = process.wait()
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, self.BUILD_COMMANDS[1], output=f"详见 {result_file}")
            
            # 编译完成
            if progress_callback:
                progress_callback(100, "编译完成")
                
        except subprocess.CalledProcessError as e:
            return OperationResult(False, f"编译命令失败: {self.BUILD_COMMANDS[1]}, 返回码 {e.returncode}, {getattr(e, 'output', '')}")
            
        return OperationResult(True, f"编译完成，输出已保存至 {result_file}")


class ModernGitSubmoduleGUI:
    """Git子模块管理GUI (v5.0)"""

    def __init__(self, manager: GitSubmoduleManager):
        self.manager = manager
        self.config_file = self.manager.root_dir / '.git_gui_config.json'
        self.config = AppConfig.load(self.config_file)
        self.manager.config = self.config

        self._init_root_window()
        self._setup_fonts()
        self._create_ui()
        self._setup_theme()
        self._bind_events()
        self._init_state()

        self.log(f"=== 应用已启动 (v{APP_VERSION}) ===")
        self.log(f"根目录: {self.manager.root_dir}")
        
        self.root.after(100, self.async_load_initial_data)
        self.root.after(2000, self.check_for_updates) # 2秒后检查更新
        self.root.mainloop()

    def check_for_updates(self):
        """在后台线程中检查更新"""
        self.log("正在检查更新...", "INFO")
        self._run_in_thread(self._update_check_worker, self._on_update_check_done)

    def _update_check_worker(self) -> Optional[Dict]:
        """[工作线程] 从远程URL获取最新版本信息"""
        try:
            if "YOUR_USERNAME" in UPDATE_URL:
                self.log("更新URL未配置，跳过检查。", "WARNING")
                return None

            response = requests.get(UPDATE_URL, timeout=10)
            response.raise_for_status()
            latest_release = response.json()
            
            latest_version = latest_release.get("tag_name", "").lstrip('v')
            current_version = APP_VERSION

            if latest_version and latest_version > current_version:
                assets = latest_release.get("assets", [])
                download_url = None
                for asset in assets:
                    if asset.get("name", "").endswith(".exe"):
                        download_url = asset.get("browser_download_url")
                        break
                
                if download_url:
                    return {
                        "latest_version": latest_version,
                        "download_url": download_url,
                        "release_notes": latest_release.get("body", "没有发布说明。")
                    }
                else:
                    self.log("在新版本中未找到.exe下载文件。", "WARNING")

        except requests.RequestException as e:
            self.log(f"检查更新失败: {e}", "WARNING")
        except (KeyError, IndexError) as e:
            self.log(f"解析更新数据失败: {e}", "WARNING")
        return None

    def _on_update_check_done(self, update_info: Optional[Dict]):
        """[UI线程] 收到更新信息后，提示用户"""
        if update_info:
            self.log(f"发现新版本: {update_info['latest_version']}", "SUCCESS")
            msg = (
                f"发现新版本: v{update_info['latest_version']} (当前版本: v{APP_VERSION})\n\n"
                f"更新内容:\n{update_info['release_notes']}\n\n"
                "是否立即自动下载并安装更新？"
            )
            if messagebox.askyesno("发现新版本", msg, parent=self.root):
                self._download_and_apply_update(update_info['download_url'])
        else:
            self.log("当前已是最新版本。", "INFO")

    def _download_and_apply_update(self, download_url: str):
        """启动后台线程以下载并应用更新"""
        if self._start_operation("应用更新") is False: return
        self.log(f"开始从 {download_url} 下载更新...", "INFO")
        self.notebook.select(0) # 切换到进度标签页
        self._run_in_thread(self._update_worker, self._on_update_finished, download_url)

    def _update_worker(self, url: str) -> str:
        """[工作线程] 执行下载、创建更新脚本并退出"""
        try:
            # 确定可执行文件名
            exe_path = Path(sys.executable)
            exe_name = exe_path.name
            update_tmp_path = exe_path.parent / "update.tmp"

            # 下载新版本
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                bytes_downloaded = 0
                with open(update_tmp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size > 0:
                            progress = (bytes_downloaded / total_size) * 100
                            self.update_progress_from_thread(progress, f"下载中... {bytes_downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB")
            
            self.log("下载完成，准备应用更新...", "SUCCESS")

            # 创建 updater.bat 脚本
            updater_script_path = exe_path.parent / "updater.bat"
            script_content = f"""
@echo off
echo Waiting for application to close...
timeout /t 3 /nobreak > nul
echo Replacing executable...
move /Y "{update_tmp_path.name}" "{exe_name}" > nul
echo Relaunching application...
start "" "{exe_name}"
echo Cleaning up...
del "{updater_script_path.name}"
"""
            with open(updater_script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            # 启动脚本并准备退出
            subprocess.Popen([str(updater_script_path)], creationflags=subprocess.DETACHED_PROCESS, shell=True)
            return "restarting"

        except Exception as e:
            self.log(f"更新过程中发生错误: {e}", "ERROR")
            return f"更新失败: {e}"

    def _on_update_finished(self, result: str):
        """[UI线程] 更新工作完成后，退出应用或显示错误"""
        if result == "restarting":
            self.log("应用即将重启以完成更新...", "INFO")
            self.root.after(1000, self.root.destroy) # 延迟1秒后关闭
        else:
            self._show_error(result)
            self._end_operation(0, 1)

    def _init_root_window(self):
        self.root = tk.Tk()
        self.root.title("Harman Git 子模块管理工具 v5.0 ")
        self.root.geometry(self.config.window_geometry or "1000x750")
        self.root.minsize(900, 650)
        self.root.configure(bg=ThemeColors.BACKGROUND)
        try:
            self.root.iconbitmap(self._resource_path("icon.ico"))
        except:
            pass
        self._center_window()
        
    def _center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_fonts(self):
        self.title_font = Font(family="Microsoft YaHei", size=16, weight="bold")
        self.subtitle_font = Font(family="Microsoft YaHei", size=12, weight="bold")
        self.button_font = Font(family="Microsoft YaHei", size=10)
        self.selected_tab_font = Font(family="Microsoft YaHei", size=11, weight="bold")
        self.log_font = Font(family="Consolas", size=10)
        self.status_font = Font(family="Microsoft YaHei", size=9)

    def _create_ui(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self._create_title_bar()
        self._create_toolbar()
        self._create_main_content()
        self._create_status_bar()

    def _create_title_bar(self):
        title_frame = ttk.Frame(self.main_container)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame, text="🚀 Harman Git 子模块管理工具", font=self.title_font, foreground=ThemeColors.PRIMARY).pack(side=tk.LEFT)
        ttk.Label(title_frame, text="v5.0 ", font=self.status_font, foreground=ThemeColors.ON_SURFACE_VARIANT).pack(side=tk.LEFT, padx=(10, 0))
        button_frame = ttk.Frame(title_frame)
        button_frame.pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="❓ 帮助", command=self._show_help, style="Secondary.TButton", width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="❌ 退出", command=self._on_closing, style="Danger.TButton", width=8).pack(side=tk.LEFT)

    def _create_toolbar(self):
        toolbar_frame = ttk.LabelFrame(self.main_container, text="🛠️ 操作工具栏", padding=(15, 10))
        toolbar_frame.pack(fill=tk.X, pady=(0, 15))
        buttons = [
            ("🔄 切换分支/标签", self.switch_refs, "Primary.TButton", "Ctrl+S"),
            ("🎯 一键切换Base", self.switch_base_branches, "Info.TButton", "Ctrl+D"),
            ("🔨 执行编译", self.build_project, "Success.TButton", "Ctrl+B"),
        ]
        for i, (text, command, style, shortcut) in enumerate(buttons):
            row, col = divmod(i, 3)
            btn = ttk.Button(toolbar_frame, text=text, command=command, style=style, width=20)
            btn.grid(row=row, column=col, padx=8, pady=5, sticky="ew")
            self._create_tooltip(btn, f"{text}\n快捷键: {shortcut}")
        for col in range(3):
            toolbar_frame.grid_columnconfigure(col, weight=1)

    def _create_main_content(self):
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self._create_progress_tab()
        self._create_log_tab()
        self._create_modules_tab()
        self.notebook.select(0)
    
    def _create_progress_tab(self):
        progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(progress_frame, text="📊 进度监控")
        main_progress_frame = ttk.LabelFrame(progress_frame, text="🎯 当前操作进度", padding=(20, 15))
        main_progress_frame.pack(fill=tk.X, padx=15, pady=15)
        self.operation_name_var = tk.StringVar(value="等待操作...")
        ttk.Label(main_progress_frame, textvariable=self.operation_name_var, font=self.subtitle_font, foreground=ThemeColors.PRIMARY).pack(anchor=tk.W, pady=(0, 10))
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(main_progress_frame, variable=self.progress_var, length=500, mode='determinate', style="Modern.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        progress_info_frame = ttk.Frame(main_progress_frame)
        progress_info_frame.pack(fill=tk.X)
        self.progress_percent_var = tk.StringVar(value="0%")
        ttk.Label(progress_info_frame, textvariable=self.progress_percent_var, font=self.button_font, foreground=ThemeColors.ACCENT).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(progress_info_frame, textvariable=self.status_var, font=self.status_font, foreground=ThemeColors.ON_SURFACE_VARIANT).pack(side=tk.RIGHT)
        
        stats_frame = ttk.LabelFrame(progress_frame, text="📈 操作统计", padding=(20, 15))
        stats_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        self.stats = {'total': tk.StringVar(value="0"), 'success': tk.StringVar(value="0"), 'failed': tk.StringVar(value="0"), 'time': tk.StringVar(value="0.0s")}
        stat_items = [("总计", self.stats['total'], ThemeColors.PRIMARY), ("成功", self.stats['success'], ThemeColors.SUCCESS), ("失败", self.stats['failed'], ThemeColors.DANGER), ("耗时", self.stats['time'], ThemeColors.WARNING)]
        for i, (label, var, color) in enumerate(stat_items):
            frame = ttk.Frame(stats_grid)
            frame.grid(row=0, column=i, padx=20, pady=5)
            ttk.Label(frame, text=label, font=self.status_font).pack()
            ttk.Label(frame, textvariable=var, font=self.subtitle_font, foreground=color).pack()
        for i in range(4):
            stats_grid.grid_columnconfigure(i, weight=1)

    def _create_log_tab(self):
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="📄 操作日志")
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
        ttk.Label(log_control_frame, text="📝 实时日志", font=self.subtitle_font).pack(side=tk.LEFT)
        log_btn_frame = ttk.Frame(log_control_frame)
        log_btn_frame.pack(side=tk.RIGHT)
        ttk.Button(log_btn_frame, text="🗑️ 清空", command=self._clear_log, style="Warning.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_btn_frame, text="💾 保存", command=self._save_log, style="Info.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_btn_frame, text="📋 复制", command=self._copy_log, style="Secondary.TButton").pack(side=tk.LEFT)
        log_content_frame = ttk.Frame(log_frame)
        log_content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.log_text = scrolledtext.ScrolledText(
            log_content_frame, wrap=tk.WORD, font=self.log_font, padx=15, pady=15, 
            bg=ThemeColors.SURFACE, fg=ThemeColors.ON_SURFACE, insertbackground=ThemeColors.ACCENT, 
            selectbackground=ThemeColors.ACCENT, selectforeground=ThemeColors.SURFACE
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.tag_configure("INFO", foreground=ThemeColors.PRIMARY)
        self.log_text.tag_configure("SUCCESS", foreground=ThemeColors.SUCCESS)
        self.log_text.tag_configure("WARNING", foreground=ThemeColors.WARNING)
        self.log_text.tag_configure("ERROR", foreground=ThemeColors.DANGER)
        self.log_text.tag_configure("TIMESTAMP", foreground=ThemeColors.ON_SURFACE_VARIANT)

    def _create_modules_tab(self):
        modules_frame = ttk.Frame(self.notebook)
        self.notebook.add(modules_frame, text="📦 子模块信息")
        modules_control_frame = ttk.Frame(modules_frame)
        modules_control_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
        ttk.Label(modules_control_frame, text="📦 子模块管理", font=self.subtitle_font).pack(side=tk.LEFT)
        
        control_btn_frame = ttk.Frame(modules_control_frame)
        control_btn_frame.pack(side=tk.RIGHT)
        
        # 将主库分支信息移动到这里
        self.main_branch_var = tk.StringVar(value="主库分支: 获取中...")
        main_branch_label = ttk.Label(control_btn_frame, textvariable=self.main_branch_var, font=self.status_font, foreground=ThemeColors.ACCENT, relief="solid", padding=(8, 4), borderwidth=1)
        main_branch_label.pack(side=tk.LEFT, padx=(0, 15))
        self._create_tooltip(main_branch_label, "点击查看主库 (Toyota_Apollo_DSP_GriffinXP) 的详细Git信息")
        main_branch_label.bind("<Button-1>", self._show_main_repo_details)
        
        ttk.Button(control_btn_frame, text="🔄 刷新列表", command=self.async_load_initial_data, style="Info.TButton").pack(side=tk.LEFT)
        
        self.modules_list_frame = ttk.LabelFrame(modules_frame, text="📋 子模块列表 (加载中...)", padding=(15, 10))
        self.modules_list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self._create_modules_tree(self.modules_list_frame)

    def _create_modules_tree(self, parent):
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.modules_tree = ttk.Treeview(tree_frame, columns=("path", "status", "branch"), show="tree headings", height=15)
        self.modules_tree.heading("#0", text="模块名称", anchor=tk.W)
        self.modules_tree.heading("path", text="路径", anchor=tk.W)
        self.modules_tree.heading("status", text="状态", anchor=tk.CENTER)
        self.modules_tree.heading("branch", text="分支/标签", anchor=tk.W)
        self.modules_tree.column("#0", width=200, minwidth=150, stretch=tk.NO)
        self.modules_tree.column("path", width=350, minwidth=250, stretch=tk.NO)
        self.modules_tree.column("status", width=80, minwidth=60, stretch=tk.NO)
        self.modules_tree.column("branch", width=600, minwidth=300, stretch=tk.NO)
        
        tree_scrollbar_v = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.modules_tree.yview)
        tree_scrollbar_h = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.modules_tree.xview)
        self.modules_tree.configure(yscrollcommand=tree_scrollbar_v.set, xscrollcommand=tree_scrollbar_h.set)
        
        self.modules_tree.grid(row=0, column=0, sticky="nsew")
        tree_scrollbar_v.grid(row=0, column=1, sticky="ns")
        tree_scrollbar_h.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        self.modules_tree.insert("", tk.END, text="🔄 正在加载子模块信息，请稍候...", values=("", "", ""), tags=('loading',))
        self.modules_tree.tag_configure('loading', foreground=ThemeColors.ON_SURFACE_VARIANT)

        self.modules_tree.bind("<Double-1>", self._on_module_double_click)
        self.modules_tree.bind("<Motion>", self._on_tree_motion)
        self.modules_tree.bind("<Leave>", self._on_tree_leave)
        self.tree_tooltip = None

    def _create_status_bar(self):
        self.status_frame = ttk.Frame(self.main_container)
        self.status_frame.pack(fill=tk.X)
        ttk.Separator(self.status_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 5))
        status_content = ttk.Frame(self.status_frame)
        status_content.pack(fill=tk.X, pady=(0, 5))

        # Configure grid columns
        status_content.grid_columnconfigure(0, weight=1) # Left status message (stretches)
        status_content.grid_columnconfigure(1, weight=0) # Submodule count
        status_content.grid_columnconfigure(2, weight=0) # Root dir name

        # Left status message
        self.status_left_var = tk.StringVar(value="就绪")
        ttk.Label(status_content, textvariable=self.status_left_var, font=self.status_font, foreground=ThemeColors.ON_SURFACE_VARIANT).grid(row=0, column=0, sticky="w")

        # Right-aligned items
        self.count_label_var = tk.StringVar(value="📦 0 个子模块")
        ttk.Label(status_content, textvariable=self.count_label_var, font=self.status_font, foreground=ThemeColors.ON_SURFACE_VARIANT).grid(row=0, column=1, sticky="e", padx=(10, 0))

        ttk.Label(status_content, text=f"📁 {self.manager.root_dir.name}", font=self.status_font, foreground=ThemeColors.ON_SURFACE_VARIANT).grid(row=0, column=2, sticky="e", padx=(10, 0))

    def _setup_theme(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=ThemeColors.BACKGROUND, foreground=ThemeColors.ON_SURFACE, borderwidth=0, focuscolor="none")
        style.configure('TFrame', background=ThemeColors.BACKGROUND, relief="flat", borderwidth=0)
        style.configure('TLabelframe', background=ThemeColors.BACKGROUND, borderwidth=1, relief="solid", bordercolor=ThemeColors.BORDER_LIGHT)
        style.configure('TLabelframe.Label', background=ThemeColors.BACKGROUND, foreground=ThemeColors.PRIMARY, font=self.subtitle_font)
        self._configure_button_styles(style)
        self._configure_progressbar_styles(style)
        self._configure_notebook_styles(style)
        self._configure_treeview_styles(style)
        self._configure_checkbutton_styles(style)

    def _configure_button_styles(self, style):
        base_config = {'font': self.button_font, 'padding': (12, 8), 'relief': "flat", 'borderwidth': 0, 'anchor': 'center'}
        style.configure('Primary.TButton', background=ThemeColors.PRIMARY, foreground=ThemeColors.SURFACE, **base_config)
        style.map('Primary.TButton', background=[('active', ThemeColors.PRIMARY_LIGHT), ('pressed', ThemeColors.PRIMARY)])
        style.configure('Success.TButton', background=ThemeColors.SUCCESS, foreground=ThemeColors.SURFACE, **base_config)
        style.map('Success.TButton', background=[('active', ThemeColors.SUCCESS_LIGHT), ('pressed', ThemeColors.SUCCESS)])
        style.configure('Warning.TButton', background=ThemeColors.WARNING, foreground=ThemeColors.SURFACE, **base_config)
        style.map('Warning.TButton', background=[('active', ThemeColors.WARNING_LIGHT), ('pressed', ThemeColors.WARNING)])
        style.configure('Danger.TButton', background=ThemeColors.DANGER, foreground=ThemeColors.SURFACE, **base_config)
        style.map('Danger.TButton', background=[('active', ThemeColors.DANGER_LIGHT), ('pressed', ThemeColors.DANGER)])
        style.configure('Info.TButton', background=ThemeColors.ACCENT, foreground=ThemeColors.SURFACE, **base_config)
        style.map('Info.TButton', background=[('active', ThemeColors.ACCENT_LIGHT), ('pressed', ThemeColors.ACCENT)])
        secondary_config = base_config.copy()
        secondary_config.update({'relief': "solid", 'borderwidth': 1})
        style.configure('Secondary.TButton', background=ThemeColors.SURFACE, foreground=ThemeColors.ON_SURFACE, bordercolor=ThemeColors.BORDER, **secondary_config)
        style.map('Secondary.TButton', background=[('active', ThemeColors.HOVER), ('pressed', ThemeColors.PRESSED)], bordercolor=[('active', ThemeColors.ACCENT), ('pressed', ThemeColors.ACCENT)])

    def _configure_progressbar_styles(self, style):
        style.configure("Modern.Horizontal.TProgressbar", background=ThemeColors.ACCENT, troughcolor=ThemeColors.BORDER_LIGHT, borderwidth=0, lightcolor=ThemeColors.ACCENT, darkcolor=ThemeColors.ACCENT, thickness=25)

    def _configure_notebook_styles(self, style):
        style.configure('TNotebook', background=ThemeColors.BACKGROUND, borderwidth=0)
        style.configure('TNotebook.Tab', background=ThemeColors.SURFACE_VARIANT, foreground=ThemeColors.ON_SURFACE, padding=(20, 8), font=self.button_font, borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', ThemeColors.ACCENT), ('active', ThemeColors.HOVER)],
                  foreground=[('selected', ThemeColors.SURFACE)],
                  padding=[('selected', (20, 15))],
                  font=[('selected', self.selected_tab_font)])

    def _configure_treeview_styles(self, style):
        style.configure('Treeview', background=ThemeColors.SURFACE, foreground=ThemeColors.ON_SURFACE, fieldbackground=ThemeColors.SURFACE, rowheight=25, borderwidth=1, relief="solid", bordercolor=ThemeColors.BORDER_LIGHT, font=self.status_font)
        style.configure('Treeview.Heading', background=ThemeColors.SURFACE_VARIANT, foreground=ThemeColors.PRIMARY, font=self.button_font, relief="flat", borderwidth=1, bordercolor=ThemeColors.BORDER_LIGHT)
        style.map('Treeview', background=[('selected', ThemeColors.ACCENT)], foreground=[('selected', ThemeColors.SURFACE)])

    def _configure_checkbutton_styles(self, style):
        # 配置复选框样式，确保选中时显示对号而不是叉号
        style.configure('TCheckbutton',
                       background=ThemeColors.BACKGROUND,
                       foreground=ThemeColors.ON_SURFACE,
                       font=self.button_font,
                       focuscolor='none')
        style.map('TCheckbutton',
                 background=[('active', ThemeColors.HOVER), ('pressed', ThemeColors.PRESSED)],
                 foreground=[('disabled', ThemeColors.ON_SURFACE_VARIANT)])
        
        # 创建自定义复选框样式，确保显示对号
        style.configure('Custom.TCheckbutton',
                       background=ThemeColors.BACKGROUND,
                       foreground=ThemeColors.ON_SURFACE,
                       font=self.button_font,
                       focuscolor='none',
                       indicatoron=1)
        style.map('Custom.TCheckbutton',
                 background=[('active', ThemeColors.HOVER)],
                 indicatorcolor=[('selected', ThemeColors.SUCCESS), ('!selected', ThemeColors.SURFACE)],
                 indicatorrelief=[('pressed', 'sunken'), ('!pressed', 'raised')])

    def _bind_events(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.bind('<Control-s>', lambda e: self.switch_refs())
        self.root.bind('<Control-d>', lambda e: self.switch_base_branches())
        self.root.bind('<Control-b>', lambda e: self.build_project())
        self.root.bind('<Control-q>', lambda e: self._on_closing())
        self.root.bind('<F1>', lambda e: self._show_help())
        self.root.bind('<F5>', lambda e: self.async_load_initial_data())
        self.root.bind('<Configure>', self._on_window_configure)
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def _on_tab_changed(self, event):
        """标签页切换事件处理"""
        try:
            selected_tab = event.widget.tab(event.widget.select(), "text")
            if selected_tab == "📦 子模块信息":
                # 当切换到子模块信息标签页时自动刷新
                self.root.after(100, self.async_load_initial_data)
        except Exception as e:
            self.log(f"标签页切换事件处理出错: {e}", "WARNING")

    def _init_state(self):
        self.is_operation_running = False
        self.current_operation = None
        self.operation_start_time = None
        self._reset_stats()
        self.submodules: List[Path] = []
        self.module_info_cache: List[ModuleInfo] = []

    def _reset_stats(self):
        self.stats['total'].set("0")
        self.stats['success'].set("0")
        self.stats['failed'].set("0")
        self.stats['time'].set("0.0s")
    
    def async_load_initial_data(self):
        """异步加载子模块信息以避免UI冻结"""
        if self.is_operation_running:
            self.log("操作进行中，请稍后刷新。", "WARNING")
            return
        
        self.log("开始刷新子模块列表...", "INFO")
        for item in self.modules_tree.get_children():
            self.modules_tree.delete(item)
        self.modules_tree.insert("", tk.END, text="🔄 正在加载子模块信息，请稍候...", values=("", "", ""), tags=('loading',))
        self.modules_list_frame.config(text="📋 子模块列表 (加载中...)")
        self.count_label_var.set("📦 加载中...")
        
        self._run_in_thread(self._load_module_data_worker, self._on_initial_data_loaded)

    def _load_module_data_worker(self) -> Tuple[List[ModuleInfo], str]:
        """[工作线程] 获取所有子模块的详细信息和主库分支信息"""
        self.submodules = self.manager.get_submodules(force_refresh=True)
        infos = []
        
        # 获取主库分支信息
        main_branch = self._get_main_repo_branch()
        
        if not self.submodules:
            return infos, main_branch

        max_workers = self.config.max_workers or min(len(self.submodules), 16)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sub = {executor.submit(self._get_single_module_info, sub): sub for sub in self.submodules}
            for future in as_completed(future_to_sub):
                try:
                    infos.append(future.result())
                except Exception as e:
                    sub = future_to_sub[future]
                    infos.append(ModuleInfo(
                        name=sub.name,
                        relative_path=str(sub.relative_to(self.manager.root_dir)),
                        current_ref=GitRef("Error", RefType.DETACHED),
                        status_icon="❌",
                        ref_display=f"⚠️ 错误",
                        commit_hash="N/A",
                        error_message=str(e)
                    ))
        
        infos.sort(key=lambda x: x.name)
        return infos, main_branch
        
    def _get_single_module_info(self, module: Path) -> ModuleInfo:
        """[工作线程] 获取单个模块的信息"""
        current_ref = self.manager.get_current_ref(module)
        is_clean = self.manager.check_working_tree_clean(module)
        status_icon = "✅" if is_clean else "⚠️"
        
        try:
            r = self.manager._run_git_command(module, ['rev-parse', 'HEAD'], timeout=5)
            commit_hash = r.stdout.strip() if r.returncode == 0 else "N/A"
        except GitOperationError:
            commit_hash = "N/A"

        if current_ref.ref_type == RefType.BRANCH:
            ref_display = f"🌿 {current_ref.name}"
        elif current_ref.ref_type == RefType.TAG:
            ref_display = f"🏷️ {current_ref.name}"
        else:
            ref_display = f"🔗 {current_ref.name}"
        
        return ModuleInfo(
            name=module.name,
            relative_path=str(module.relative_to(self.manager.root_dir)),
            current_ref=current_ref,
            status_icon=status_icon,
            ref_display=ref_display,
            commit_hash=commit_hash
        )

    def _on_initial_data_loaded(self, data: Tuple[List[ModuleInfo], str]):
        """[UI线程] 收到数据后更新UI"""
        module_infos, main_branch = data
        self.module_info_cache = module_infos
        for item in self.modules_tree.get_children():
            self.modules_tree.delete(item)
        
        if not module_infos and not self.submodules:
            messagebox.showerror("错误", f"在 '{self.manager.platform_dir}' 未找到任何子模块，请检查路径。")
            self.root.quit()
            return
        
        if not module_infos:
            self.modules_tree.insert("", tk.END, text="❌ 未找到任何子模块。", tags=('loading',))
            self.log("未找到任何子模块。", "WARNING")
        else:
            for info in module_infos:
                text_prefix = "📦 " if not info.error_message else "❌ "
                self.modules_tree.insert("", tk.END, iid=info.name, text=f"{text_prefix}{info.name}", values=(info.relative_path, info.status_icon, info.ref_display))
            self.log(f"模块列表刷新完成，共 {len(module_infos)} 个模块。", "SUCCESS")

        self.modules_list_frame.config(text=f"📋 子模块列表 (共 {len(module_infos)} 个)")
        self.count_label_var.set(f"📦 {len(module_infos)} 个子模块")
        self.main_branch_var.set(f"主库分支: {main_branch}")

    def _get_main_repo_branch(self) -> str:
        """获取主库(toyota_目录)的当前分支名"""
        try:
            main_repo_path = self.manager.root_dir
            if not (main_repo_path / '.git').exists():
                return "未知(非Git仓库)"
            
            current_ref = self.manager.get_current_ref(main_repo_path)
            if current_ref.ref_type == RefType.BRANCH:
                return f"🌿 {current_ref.name}"
            elif current_ref.ref_type == RefType.TAG:
                return f"🏷️ {current_ref.name}"
            else:
                return f"🔗 {current_ref.name}"
        except Exception as e:
            self.log(f"获取主库分支信息失败: {e}", "WARNING")
            return "获取失败"

    def _run_in_thread(self, fn, cb, *args, **kwargs):
        def t():
            try:
                r = fn(*args, **kwargs)
                self.root.after(0, lambda: cb(r))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda e=e: self._show_error(f"后台操作错误: {e}"))
        threading.Thread(target=t, daemon=True).start()
    
    def _bring_to_front(self):
        """强制将主窗口置于最前面"""
        try:
            # 保存当前topmost状态
            was_topmost = self.root.attributes('-topmost')
            
            # 临时设置为置顶
            self.root.attributes('-topmost', True)
            self.root.lift()
            self.root.focus_force()
            
            # 稍微延迟后恢复原状态（如果原来不是topmost的话）
            if not was_topmost:
                self.root.after(100, lambda: self.root.attributes('-topmost', False))
        except Exception as e:
            # 如果上述方法失败，尝试基本方法
            try:
                self.root.lift()
                self.root.focus_force()
            except:
                pass

    def _show_error(self, text: str, title: str = "错误"):
        self._bring_to_front()
        self.root.after(50, lambda: messagebox.showerror(title, text, parent=self.root))
        self.log(text, "ERROR")

    def _show_info(self, text: str, title: str = "信息"):
        self._bring_to_front()
        self.root.after(50, lambda: messagebox.showinfo(title, text, parent=self.root))
        self.log(text, "SUCCESS")

    def _show_warning(self, text: str, title: str = "警告"):
        self._bring_to_front()
        self.root.after(50, lambda: messagebox.showwarning(title, text, parent=self.root))
        self.log(text, "WARNING")

    def _force_window_to_front(self):
        """更强力的窗口置顶方法"""
        try:
            # Windows特定的置顶方法
            import sys
            if sys.platform == "win32":
                try:
                    import ctypes
                    from ctypes import wintypes
                    
                    # 获取窗口句柄
                    hwnd = self.root.winfo_id()
                    
                    # 使用Windows API强制置顶
                    ctypes.windll.user32.SetWindowPos(
                        hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040
                    )
                    
                    # 激活窗口
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                    ctypes.windll.user32.BringWindowToTop(hwnd)
                    
                except Exception:
                    # 如果Windows API失败，使用tkinter方法
                    pass
            
            # 通用方法
            self.root.attributes('-topmost', True)
            self.root.lift()
            self.root.focus_force()
            self.root.bell()  # 发出提示音
            
            # 短暂延迟后取消置顶（避免一直在最前面影响使用）
            self.root.after(3000, lambda: self.root.attributes('-topmost', False))
            
        except Exception as e:
            # 最基本的方法
            try:
                self.root.lift()
                self.root.focus_force()
                self.root.bell()
            except:
                pass

    def _show_build_success(self, message: str):
        """显示编译成功消息，强制置顶"""
        self._force_window_to_front()
        self._show_info(message, "🎉 编译成功")
        
    def _show_build_error(self, message: str):
        """显示编译失败消息，强制置顶"""
        self._force_window_to_front()
        self._show_error(message, "❌ 编译失败")
        
    def _resource_path(self, relative_path: str) -> str:
        """获取资源绝对路径，兼容PyInstaller"""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def _show_result(self, text: str, title: str = "结果"):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("1000x750")
        window.transient(self.root)
        window.grab_set()
        window.configure(bg=ThemeColors.BACKGROUND)
        try:
            window.iconbitmap(self._resource_path("icon.ico"))
        except:
            pass
        self._center_on_parent(window)
        frame = ttk.Frame(window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text=title, font=self.subtitle_font, foreground=ThemeColors.PRIMARY).pack(anchor=tk.W, pady=(0, 15))
        st = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=self.log_font, padx=15, pady=15, bg=ThemeColors.SURFACE, fg=ThemeColors.ON_SURFACE)
        st.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        st.insert(tk.END, text)
        st.see(tk.END)
        st.config(state=tk.DISABLED)
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="📋 复制", command=lambda: self._copy_result_text(text, window), style="Info.TButton").pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="💾 保存", command=lambda: self._save_result_text(text, title), style="Warning.TButton").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(btn_frame, text="❌ 关闭", command=window.destroy, style="Secondary.TButton").pack(side=tk.RIGHT)

    def _copy_result_text(self, text: str, parent_win: tk.Toplevel):
        try:
            parent_win.clipboard_clear()
            parent_win.clipboard_append(text)
            self._show_info("内容已复制到剪贴板")
        except Exception as e:
            self._show_error(f"复制失败: {e}")

    def _save_result_text(self, text: str, title: str):
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("日志文件", "*.log"), ("所有文件", "*.*")],
            title=f"保存 {safe_title}",
            initialfile=f"{safe_title}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                self._show_info(f"内容已保存到: {filename}")
            except Exception as e:
                self._show_error(f"保存失败: {e}")

    def _format_results(self, res: Dict[str, List[OperationResult]]) -> str:
        out = f"\n{'='*60}\n📊 操作结果总结\n{'='*60}\n\n"
        success_list = res.get('success', [])
        failure_list = res.get('failure', [])
        total = len(success_list) + len(failure_list)
        out += f"📈 总体统计:\n   • 总操作数: {total}\n   • 成功: {len(success_list)} ✅\n   • 失败: {len(failure_list)} ❌\n"
        if total > 0:
            out += f"   • 成功率: {(len(success_list) / total) * 100:.1f}%\n"
        out += "\n"
        if success_list:
            out += f"✅ 成功操作 ({len(success_list)} 项):\n{'-' * 40}\n"
            for r in sorted(success_list, key=lambda x: Path(x.path).name):
                out += f"   ✓ {Path(r.path).name}: {r.message} ({r.duration:.1f}s)\n"
            out += "\n"
        if failure_list:
            out += f"❌ 失败操作 ({len(failure_list)} 项):\n{'-' * 40}\n"
            for r in sorted(failure_list, key=lambda x: Path(x.path).name):
                out += f"   ✗ {Path(r.path).name}: {r.message} ({r.duration:.1f}s)\n"
            out += "\n"
        out += "="*60 + "\n"
        return out
    
    def _show_module_selection_dialog(self, title: str, mode: str) -> Union[Tuple[List[Path], str], List[Path], None]:
        if not self.module_info_cache:
            self._show_error("子模块列表为空，无法执行操作。")
            return None
        
        # 设置当前对话框模式，供_create_module_selection使用
        self.current_dialog_mode = mode
        
        result = {"subs": [], "ref": None}
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1000x700")
        win.transient(self.root)
        win.grab_set()
        win.configure(bg=ThemeColors.BACKGROUND)
        try:
            win.iconbitmap(self._resource_path("icon.ico"))
        except:
            pass
        self._center_on_parent(win)

        main_container = ttk.Frame(win, padding=20)
        main_container.pack(fill=tk.BOTH, expand=True)

        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame, text=title, font=self.title_font, foreground=ThemeColors.PRIMARY).pack(side=tk.LEFT)
        quick_btn_frame = ttk.Frame(title_frame)
        quick_btn_frame.pack(side=tk.RIGHT)
        ttk.Button(quick_btn_frame, text="✅ 全选", command=lambda: self._select_all_modules(True), style="Success.TButton").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_btn_frame, text="❌ 全不选", command=lambda: self._select_all_modules(False), style="Warning.TButton").pack(side=tk.LEFT)
        
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        content_frame.grid_columnconfigure(0, weight=2)
        content_frame.grid_columnconfigure(1, weight=3)
        content_frame.grid_rowconfigure(0, weight=1)
        
        left_frame = ttk.LabelFrame(content_frame, text="📦 选择子模块", padding=15)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._create_module_selection(left_frame)
        
        right_frame = ttk.LabelFrame(content_frame, text="🎯 预览", padding=15)
        right_frame.grid(row=0, column=1, sticky="nsew")

        if mode == "base_branch":
            self._create_base_branch_preview(right_frame)
        else:
            self._create_selected_preview(right_frame)

        bottom_frame = ttk.Frame(main_container)
        bottom_frame.pack(fill=tk.X)
        
        def on_confirm():
            selected = [p for v, p in self.sub_vars if v.get()]
            
            # 检查是否有任何选择（子模块或Toyota_Apollo_DSP_GriffinXP）
            has_sibling_selected = (hasattr(self, 'sibling_var') and
                                   self.sibling_var is not None and
                                   self.sibling_var.get())
            
            if not selected and not has_sibling_selected:
                self._show_warning("请至少选择一个子模块")
                return
            
            ref = self.ref_entry.get().strip() if mode == "standard" else "base_branch_mode"
            if not ref and mode == "standard":
                self._show_warning("请输入分支或标签名")
                return

            result["subs"] = selected
            # 保存sibling选择状态，供后续使用
            result["include_sibling"] = has_sibling_selected
            if mode == "standard":
                result["ref"] = ref
                self.config.last_ref = ref
            
            self.config.last_selected_modules = [s.name for s in selected]
            win.destroy()

        if mode == "standard":
            ref_frame = ttk.LabelFrame(bottom_frame, text="🎯 目标分支/标签", padding=15)
            ref_frame.pack(fill=tk.X, pady=(0, 15))
            ref_input_frame = ttk.Frame(ref_frame)
            ref_input_frame.pack(fill=tk.X)
            ttk.Label(ref_input_frame, text="分支/标签:", font=self.button_font).pack(side=tk.LEFT, padx=(0, 10))
            self.ref_entry = ttk.Entry(ref_input_frame, font=self.button_font, width=30)
            self.ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            self.ref_entry.focus_set()
            if self.config.last_ref:
                self.ref_entry.insert(0, self.config.last_ref)
            # 删除了 main、master、develop、release 按钮的创建代码
            
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        confirm_btn_text = "🚀 开始切换" if mode=="standard" else "🚀 一键切换Base分支"
        ttk.Button(button_frame, text=confirm_btn_text, command=on_confirm, style="Success.TButton").pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="❌ 取消", command=win.destroy, style="Secondary.TButton", width=10).pack(side=tk.RIGHT, padx=(0, 10))

        win.bind("<Return>", lambda e: on_confirm())
        win.bind("<Escape>", lambda e: win.destroy())
        win.wait_window()
        
        if result["ref"] is None and mode == "standard": return None
        if not result["subs"] and not result.get("include_sibling", False): return None
        
        if mode == "standard":
            return (result["subs"], result["ref"])
        else:
            # base_branch模式返回包含sibling信息的结果
            return {"modules": result["subs"], "include_sibling": result.get("include_sibling", False)}

    def _create_module_selection(self, parent):
        canvas = tk.Canvas(parent, bg=ThemeColors.SURFACE, highlightthickness=0, bd=0)
        v_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg=ThemeColors.SURFACE)
        
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        
        self.sub_vars = []
        self.sibling_var = None  # 用于存储同级路径的变量

        def on_checkbox_change():
            if hasattr(self, 'base_preview_listbox') and self.base_preview_listbox.winfo_exists():
                self._update_base_branch_preview()
            if hasattr(self, 'selected_listbox') and self.selected_listbox.winfo_exists():
                self._update_selected_preview()

        # 添加子模块选项
        for info in self.module_info_cache:
            sub_path = self.manager.root_dir / info.relative_path
            var = tk.BooleanVar(value=(info.name in self.config.last_selected_modules))
            item_frame = tk.Frame(scrollable_frame, bg=ThemeColors.SURFACE)
            item_frame.pack(fill=tk.X, padx=8, pady=1, anchor="w")
            cb = tk.Checkbutton(item_frame,
                              text=f"📦 {info.name}",
                              variable=var,
                              command=on_checkbox_change,
                              bg=ThemeColors.SURFACE,
                              fg=ThemeColors.ON_SURFACE,
                              font=self.button_font,
                              selectcolor=ThemeColors.SURFACE,
                              activebackground=ThemeColors.HOVER,
                              activeforeground=ThemeColors.ON_SURFACE,
                              relief='flat',
                              borderwidth=0,
                              highlightthickness=0,
                              anchor='w')
            cb.pack(side=tk.LEFT, anchor=tk.W)
            
            status_tooltip = "工作区干净" if info.status_icon == '✅' else "有未提交更改" if info.status_icon == '⚠️' else "状态未知"
            status_label = ttk.Label(item_frame, text=info.status_icon, font=self.status_font)
            status_label.pack(side=tk.RIGHT, padx=(5, 0))
            self._create_tooltip(status_label, f"{info.name}\n状态: {status_tooltip}")
            self.sub_vars.append((var, sub_path))

        # 在base_branch模式下添加toyota_目录选项（显示为Toyota_Apollo_DSP_GriffinXP）
        sibling_path = self.manager.root_dir  # 指向toyota_目录本身
        if hasattr(self, 'current_dialog_mode') and self.current_dialog_mode == "base_branch":
            # toyota_目录肯定存在且有.git目录
            if sibling_path.exists() and (sibling_path / '.git').exists():
                # 添加分隔线
                separator_frame = tk.Frame(scrollable_frame, bg=ThemeColors.SURFACE, height=2)
                separator_frame.pack(fill=tk.X, padx=8, pady=8)
                ttk.Separator(separator_frame, orient=tk.HORIZONTAL).pack(fill=tk.X)
                
                # 添加同级路径选项
                self.sibling_var = tk.BooleanVar(value=True)  # 默认选中同级路径
                sibling_frame = tk.Frame(scrollable_frame, bg=ThemeColors.SURFACE)
                sibling_frame.pack(fill=tk.X, padx=8, pady=1, anchor="w")
                sibling_cb = tk.Checkbutton(sibling_frame,
                                          text="📁 Toyota_Apollo_DSP_GriffinXP (主库)",
                                          variable=self.sibling_var,
                                          command=on_checkbox_change,
                                          bg=ThemeColors.SURFACE,
                                          fg=ThemeColors.ON_SURFACE,
                                          font=self.button_font,
                                          selectcolor=ThemeColors.SURFACE,
                                          activebackground=ThemeColors.HOVER,
                                          activeforeground=ThemeColors.ON_SURFACE,
                                          relief='flat',
                                          borderwidth=0,
                                          highlightthickness=0,
                                          anchor='w')
                sibling_cb.pack(side=tk.LEFT, anchor=tk.W)
                
                # toyota_目录不检查工作区状态，直接显示为干净
                sibling_status_icon = "✅"
                sibling_status_tooltip = "工作区干净"
                
                sibling_status_label = ttk.Label(sibling_frame, text=sibling_status_icon, font=self.status_font)
                sibling_status_label.pack(side=tk.RIGHT, padx=(5, 0))
                self._create_tooltip(sibling_status_label, f"Toyota_Apollo_DSP_GriffinXP\n状态: {sibling_status_tooltip}")

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        for widget in [canvas, scrollable_frame]:
            widget.bind("<MouseWheel>", on_mousewheel)
            for child in scrollable_frame.winfo_children():
                child.bind("<MouseWheel>", on_mousewheel)
                for grandchild in child.winfo_children():
                    grandchild.bind("<MouseWheel>", on_mousewheel)

        self.root.after(50, configure_scroll_region)
        self.root.after(100, on_checkbox_change)

    def _create_selected_preview(self, parent):
        self.selected_listbox = tk.Listbox(parent, selectmode=tk.SINGLE, exportselection=False, bg=ThemeColors.SURFACE, fg=ThemeColors.ON_SURFACE, font=self.log_font, height=20, activestyle='none')
        listbox_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.selected_listbox.yview)
        self.selected_listbox.config(yscrollcommand=listbox_scrollbar.set)
        self.selected_listbox.pack(side="left", fill="both", expand=True)
        listbox_scrollbar.pack(side="right", fill="y")
        self._update_selected_preview()
    
    def _select_all_modules(self, select: bool):
        for var, _ in self.sub_vars:
            var.set(select)
        # 处理Toyota_Apollo_DSP_GriffinXP选项（如果存在）
        if hasattr(self, 'sibling_var') and self.sibling_var is not None:
            self.sibling_var.set(select)
        if hasattr(self, 'base_preview_listbox') and self.base_preview_listbox.winfo_exists():
            self._update_base_branch_preview()
        if hasattr(self, 'selected_listbox') and self.selected_listbox.winfo_exists():
            self._update_selected_preview()
            
    def _update_selected_preview(self):
        if not hasattr(self, 'selected_listbox'): return
        self.selected_listbox.delete(0, tk.END)
        selected_modules = [sub for var, sub in self.sub_vars if var.get()]
        parent = self.selected_listbox.master
        
        if selected_modules:
            self.selected_listbox.insert(tk.END, f"📊 已选择 {len(selected_modules)} 个模块:")
            self.selected_listbox.insert(tk.END, "")
            for i, sub in enumerate(selected_modules, 1):
                self.selected_listbox.insert(tk.END, f"{i:2d}. 📦 {sub.name}")
            if hasattr(parent, 'config'): parent.config(text=f"✅ 已选模块 ({len(selected_modules)} 个)")
        else:
            self.selected_listbox.insert(tk.END, "💡 点击左侧复选框选择模块")
            if hasattr(parent, 'config'): parent.config(text="✅ 已选模块")
    
    def _create_base_branch_preview(self, parent):
        self.base_preview_listbox = tk.Listbox(parent, selectmode=tk.SINGLE, exportselection=False, bg=ThemeColors.SURFACE, fg=ThemeColors.ON_SURFACE, font=self.log_font, height=20, activestyle='none')
        listbox_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.base_preview_listbox.yview)
        self.base_preview_listbox.config(yscrollcommand=listbox_scrollbar.set)
        self.base_preview_listbox.pack(side="left", fill="both", expand=True)
        listbox_scrollbar.pack(side="right", fill="y")
        self._update_base_branch_preview()

    def _update_base_branch_preview(self):
        if not hasattr(self, 'base_preview_listbox'): return
        self.base_preview_listbox.delete(0, tk.END)
        selected_modules = [sub for var, sub in self.sub_vars if var.get()]
        
        # 检查用户是否选中toyota_目录选项
        sibling_path = self.manager.root_dir  # 指向toyota_目录本身
        include_sibling = False
        if hasattr(self, 'sibling_var') and self.sibling_var is not None:
            include_sibling = self.sibling_var.get()
        
        if selected_modules or include_sibling:
            # 计算总数
            total_count = len(selected_modules)
            has_sibling = include_sibling and sibling_path.exists() and (sibling_path / '.git').exists()
            if has_sibling:
                total_count += 1
            
            self.base_preview_listbox.insert(tk.END, f"🎯 将切换到以下Base分支 ({total_count} 个):")
            self.base_preview_listbox.insert(tk.END, "")
            
            # 显示选中的子模块
            for i, sub in enumerate(selected_modules, 1):
                base_branch = self._get_base_branch_name(sub.name)
                self.base_preview_listbox.insert(tk.END, f"{i:2d}. 📦 {sub.name}")
                self.base_preview_listbox.insert(tk.END, f"    └─ 🎯 {base_branch}")
                self.base_preview_listbox.insert(tk.END, "")
            
            # 只有当用户选中同级路径时才显示
            if has_sibling:
                self.base_preview_listbox.insert(tk.END, f"{total_count:2d}. 📁 Toyota_Apollo_DSP_GriffinXP (主库)")
                self.base_preview_listbox.insert(tk.END, f"    └─ 🎯 MisraFix/TPCY21PD-11805_MisraFixBaseBranch_Unit_1")
        else:
            self.base_preview_listbox.insert(tk.END, "✅ 选择模块后将显示对应的Base分支名")
    
    def _set_ref(self, ref_name: str):
        if hasattr(self, 'ref_entry'):
            self.ref_entry.delete(0, tk.END)
            self.ref_entry.insert(0, ref_name)
    
    def _get_base_branch_name(self, module_name: str) -> str:
        special_mappings = {"c_ap_a17_components": "a17_components", "c_ap_common_components": "common_components", "c_ap_d17_components": "d17_components"}
        suffix = special_mappings.get(module_name, module_name)
        return f"MisraFix/TPCY21PD-11805_MisraFixBaseBranch_{suffix}"
    
    def switch_refs(self):
        if self._start_operation("初始化切换") is False: return
        dialog_result = self._show_module_selection_dialog("🔄 批量切换子模块", mode="standard")
        if dialog_result is None:
            self._end_operation()
            return

        subs, ref = dialog_result
        if not subs or not ref:
            self._end_operation()
            return
            
        self._execute_switch(subs, ref)

    def switch_base_branches(self):
        if self._start_operation("初始化Base切换") is False: return
        dialog_result = self._show_module_selection_dialog("🎯 切换到Base分支", mode="base_branch")
        if not dialog_result:
            self._end_operation()
            return
        
        # 提取子模块列表和sibling选择状态
        selected_modules = dialog_result["modules"]
        include_sibling = dialog_result["include_sibling"]
        
        self._execute_base_branch_switch(selected_modules, include_sibling)

    def _execute_switch(self, subs: List[Path], ref: str):
        self.operation_name_var.set(f"切换到 {ref}")
        def task():
            res = self.manager._execute_parallel_operation(subs, self.manager.process_submodule, self.update_progress_from_thread, ref=ref)
            self._handle_operation_result(res, subs, self._execute_switch, (subs, ref))
        threading.Thread(target=task, daemon=True).start()

    def _execute_base_branch_switch(self, selected_modules: List[Path], include_sibling: bool = False):
        self.operation_name_var.set("切换到Base分支")
        
        # 添加同级路径到处理列表（根据用户选择）
        all_paths_to_process = []
        
        # 先添加选中的子模块
        all_paths_to_process.extend(selected_modules)
        
        # 根据传入的参数决定是否添加toyota_目录
        sibling_path = self.manager.root_dir  # 指向toyota_目录本身
        
        # 如果用户选中同级路径，且路径存在且是git仓库，则添加
        if include_sibling and sibling_path.exists() and (sibling_path / '.git').exists():
            all_paths_to_process.append(sibling_path)
            self.log(f"添加同级路径到Base切换列表: {sibling_path.name}", "INFO")
        
        # 如果没有任何路径需要处理，直接结束
        if not all_paths_to_process:
            self._end_operation(0, 0)
            return
        
        def single_base_switch_op(repo: Path):
            if repo == sibling_path:
                # 同级路径使用固定的base分支名
                base_branch = "MisraFix/TPCY21PD-11805_MisraFixBaseBranch_Unit_1"
            else:
                base_branch = self._get_base_branch_name(repo.name)
            return self.manager.process_submodule(repo, base_branch)
        
        def task():
            # 使用自定义的并行操作，跳过toyota_目录的工作区检查
            res = self._execute_base_parallel_operation(all_paths_to_process, single_base_switch_op, self.update_progress_from_thread, sibling_path)
            self._handle_operation_result(res, all_paths_to_process, self._execute_base_branch_switch, (selected_modules, include_sibling))
        threading.Thread(target=task, daemon=True).start()

    def _execute_base_parallel_operation(self, subs: List[Path], operation_func: Callable,
                                       progress_callback: Optional[Callable] = None,
                                       skip_dirty_check_path: Optional[Path] = None) -> Dict[str, Any]:
        """专门为Base分支切换的并行操作，可以跳过特定路径的工作区检查"""
        res = {'success': [], 'failure': [], 'dirty': []}
        total = len(subs)

        self.manager.logger.info("开始预检查工作区状态...")
        dirty_repos = []
        max_workers_check = self.manager.config.max_workers or min(total, 16)
        with ThreadPoolExecutor(max_workers=max_workers_check) as exe:
            futs = {exe.submit(self._check_working_tree_for_base, s, skip_dirty_check_path): s for s in subs}
            for i, fut in enumerate(as_completed(futs)):
                repo = futs[fut]
                if progress_callback:
                    progress = int((i + 1) / total * 20)
                    progress_callback(progress, f"检查状态: {repo.name}...")
                if not fut.result():
                    dirty_repos.append(repo)
        
        if dirty_repos:
            res['dirty'] = dirty_repos
            return res

        self.manager.logger.info("预检查通过, 开始执行核心操作...")
        max_workers_op = self.manager.config.max_workers or min(total, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=max_workers_op) as exe:
            futs = {exe.submit(operation_func, s): s for s in subs}
            completed = 0
            for fut in as_completed(futs):
                completed += 1
                repo = futs[fut]
                if progress_callback:
                    progress = 20 + int(completed / total * 80)
                    progress_callback(progress, f"处理中: {repo.name}...")
                try:
                    r = fut.result()
                    if r.success:
                        res['success'].append(r)
                        self.manager.logger.info(f"✓ {repo.name}: {r.message} ({r.duration:.1f}s)")
                    else:
                        res['failure'].append(r)
                        self.manager.logger.error(f"✗ {repo.name}: {r.message} ({r.duration:.1f}s)")
                except Exception as e:
                    error_result = OperationResult(False, f"异常: {e}", str(repo))
                    res['failure'].append(error_result)
                    self.manager.logger.error(f"✗ {repo.name}: 异常 {e}")
        return res

    def _check_working_tree_for_base(self, repo: Path, skip_path: Optional[Path] = None) -> bool:
        """为Base分支切换检查工作区状态，可以跳过特定路径"""
        if skip_path and repo == skip_path:
            # 跳过toyota_目录的工作区检查，直接返回True（干净）
            return True
        return self.manager.check_working_tree_clean(repo)
    
    def _handle_operation_result(self, res: Dict, subs: List[Path], on_retry_callable: Callable, on_retry_args: tuple):
        if 'dirty' in res and res['dirty']:
            def on_continue(action: str):
                self._perform_cleanup_and_retry(res['dirty'], action, on_retry_callable, on_retry_args)
            self.root.after(0, lambda: self._handle_dirty_repos_dialog(res['dirty'], on_continue))
        else:
            success_count = len(res.get('success', []))
            total_count = len(subs)
            out = self._format_results(res)
            
            self.root.after(0, lambda: self._show_result(out, f"{self.current_operation} - 操作结果"))
            self.root.after(0, lambda: self._end_operation(success_count, total_count))
            self.root.after(100, self.async_load_initial_data)

    def _handle_dirty_repos_dialog(self, dirty_repos: List[Path], on_continue: Callable[[str], None]):
        self.log(f"发现 {len(dirty_repos)} 个子模块有未提交更改", "WARNING")
        win = tk.Toplevel(self.root)
        win.title("⚠️ 处理未提交更改")
        win.geometry("700x500")
        win.transient(self.root)
        win.grab_set()
        win.configure(bg=ThemeColors.BACKGROUND)
        try:
            win.iconbitmap(self._resource_path("icon.ico"))
        except:
            pass
        self._center_on_parent(win)
        main_frame = ttk.Frame(win, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text="⚠️ 检测到未提交的更改", font=self.subtitle_font, foreground=ThemeColors.WARNING).pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(main_frame, text="以下子模块存在未提交的更改，需要先处理：", font=self.status_font).pack(anchor=tk.W, pady=(0, 15))
        list_frame = ttk.LabelFrame(main_frame, text="📋 有更改的模块", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        modules_text = scrolledtext.ScrolledText(list_frame, wrap=tk.WORD, font=self.log_font, height=8, bg=ThemeColors.SURFACE, fg=ThemeColors.ON_SURFACE)
        modules_text.pack(fill=tk.BOTH, expand=True)
        modules_text.insert(tk.END, "\n".join([f"📦 {repo.name}" for repo in dirty_repos]))
        modules_text.config(state=tk.DISABLED)
        options_frame = ttk.LabelFrame(main_frame, text="🛠️ 处理方式", padding=15)
        options_frame.pack(fill=tk.X, pady=(0, 20))
        dirty_action = tk.StringVar(value="stash")
        ttk.Radiobutton(options_frame, text="📦 暂存更改 (git stash) - 推荐", variable=dirty_action, value="stash", style="TRadiobutton").pack(anchor=tk.W, pady=2)
        ttk.Label(options_frame, text="   将未提交的更改暂存起来，可以随时恢复", font=self.status_font, foreground=ThemeColors.ON_SURFACE_VARIANT).pack(anchor=tk.W, padx=(20, 0))
        ttk.Radiobutton(options_frame, text="🗑️ 放弃更改 (git reset --hard) - 谨慎使用", variable=dirty_action, value="discard", style="TRadiobutton").pack(anchor=tk.W, pady=(10, 2))
        ttk.Label(options_frame, text="   永久删除所有未提交的更改，无法恢复", font=self.status_font, foreground=ThemeColors.DANGER).pack(anchor=tk.W, padx=(20, 0))
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        def on_process():
            action = dirty_action.get()
            win.destroy()
            on_continue(action)
        
        ttk.Button(btn_frame, text="🚀 继续处理", command=on_process, style="Success.TButton").pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="❌ 取消操作", command=lambda: (win.destroy(), self._end_operation(0, 0)), style="Secondary.TButton").pack(side=tk.RIGHT, padx=(0, 10))

    def _perform_cleanup_and_retry(self, dirty_repos: List[Path], action: str, retry_callable: Callable, retry_args: tuple):
        action_name = "暂存更改" if action == "stash" else "放弃更改"
        self.operation_name_var.set(f"执行 {action_name}")
        self.log(f"开始{action_name}：{len(dirty_repos)} 个子模块", "INFO")
        
        def task():
            results = []
            total = len(dirty_repos)
            op_func = self.manager.stash_changes if action == "stash" else self.manager.discard_changes
            max_workers = self.config.max_workers or min(total, 10)
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futs = {exe.submit(op_func, repo): repo for repo in dirty_repos}
                for i, fut in enumerate(as_completed(futs)):
                    self.update_progress_from_thread((i + 1) / total * 100, f"处理: {futs[fut].name}...")
                    result = fut.result()
                    results.append(result)
                    level = "SUCCESS" if result.success else "ERROR"
                    self.log(f"{futs[fut].name}: {result.message}", level)

            if all(r.success for r in results):
                self.log(f"{action_name}成功，重新执行原始操作...", "INFO")
                self.root.after(500, lambda: retry_callable(*retry_args))
            else:
                self.root.after(0, lambda: self._show_error(f"{action_name}失败，操作已中止。"))
                self.root.after(0, lambda: self._end_operation(0, 0))
        
        threading.Thread(target=task, daemon=True).start()

    def build_project(self):
        if self._start_operation("项目编译") is False: return
        
        def task():
            result = self.manager.build_project(self.update_progress_from_thread)
            if result.success:
                # 编译成功 - 强制弹到最前面
                self.root.after(0, lambda: self._show_build_success(result.message))
                self.root.after(0, lambda: self._end_operation(1, 1))
            else:
                # 编译失败 - 强制弹到最前面
                self.root.after(0, lambda: self._show_build_error(result.message))
                self.root.after(0, lambda: self._end_operation(0, 1))
        
        threading.Thread(target=task, daemon=True).start()
    
    def _show_help(self):
        help_text = """
🚀 Harman Git 子模块管理工具 v5.0 

【主要功能】
• 🔄 批量切换子模块分支/标签
• 🎯 一键切换选中模块到对应的Base分支
• 🔨 执行项目编译脚本

【快捷键】
• Ctrl+S: 切换分支/标签
• Ctrl+D: 一键切换Base分支
• Ctrl+B: 执行编译
• Ctrl+Q: 退出应用
• F1:     显示帮助
• F5:     刷新模块列表

【使用说明】
1. 程序启动后会自动加载所有子模块的状态。
2. 点击操作按钮（如"切换分支/标签"）。
3. 在弹出的对话框中，勾选要操作的子模块。
4. 输入目标信息（如分支名），点击确认。
5. 在"进度监控"标签页查看实时进度。
6. 操作完成后，会弹出详细的结果报告。
7. 切换到"子模块信息"标签页会自动刷新模块列表。


【注意事项】
• 切换操作前会检查工作区是否有未提交的更改。
• 如果检测到未提交更改，会提示您"暂存"或"放弃"。
• 所有耗时操作都在后台执行，不会冻结界面。
• 网络连接问题可能导致Git操作失败。
        """
        self._show_result(help_text.strip(), "帮助信息")
    
    def _on_closing(self):
        try:
            self.config.window_geometry = self.root.geometry()
            self.config.window_state = self.root.state()
            self.config.save(self.config_file)
            
            if self.is_operation_running:
                if messagebox.askokcancel("确认退出", "有操作正在进行中，确定要退出吗？"):
                    self.root.quit()
                    self.root.destroy()
            else:
                self.root.quit()
                self.root.destroy()
        except Exception as e:
            print(f"关闭时发生错误: {e}")
            self.root.quit()
            self.root.destroy()
    
    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log("=== 日志已清空 ===")
    
    def _save_log(self):
        content = self.log_text.get(1.0, tk.END)
        if not content.strip():
            self._show_warning("日志为空，无需保存")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("日志文件", "*.log"), ("文本文件", "*.txt"), ("所有文件", "*.*")],
            title="保存日志",
            initialfile=f"git_submodule_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                self._show_info(f"日志已保存到: {filename}")
            except Exception as e:
                self._show_error(f"保存日志失败: {e}")
    
    def _copy_log(self):
        try:
            content = self.log_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._show_info("日志已复制到剪贴板")
        except Exception as e:
            self._show_error(f"复制日志失败: {e}")
    
    def _center_on_parent(self, window):
        window.update_idletasks()
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()
        window_width = window.winfo_width()
        window_height = window.winfo_height()
        x = parent_x + (parent_width - window_width) // 2
        y = parent_y + (parent_height - window_height) // 2
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def _create_tooltip(self, widget, text):
        tooltip = None
        def on_enter(event):
            nonlocal tooltip
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            label = tk.Label(tooltip, text=text, bg=ThemeColors.PRIMARY, fg=ThemeColors.SURFACE, font=self.status_font, padx=8, pady=4)
            label.pack()
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            tooltip.geometry(f"+{x}+{y}")
        
        def on_leave(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
    
    def _on_tree_motion(self, event):
        try:
            item_iid = self.modules_tree.identify_row(event.y)
            if not item_iid:
                self._hide_tree_tooltip()
                return

            info = next((i for i in self.module_info_cache if i.name == item_iid), None)
            if info:
                status_text = '工作区干净' if info.status_icon == '✅' else ('有未提交更改' if info.status_icon == '⚠️' else '状态异常')
                tooltip_text = (f"模块: {info.name}\n"
                              f"路径: {info.relative_path}\n"
                              f"状态: {status_text}\n"
                              f"当前: {info.ref_display}\n\n"
                              f"💡 双击可查看详细信息")
                self._show_tree_tooltip(event.x_root, event.y_root, tooltip_text)
            else:
                self._hide_tree_tooltip()
        except Exception:
            self._hide_tree_tooltip()
    
    def _on_tree_leave(self, event):
        self._hide_tree_tooltip()
    
    def _show_tree_tooltip(self, x, y, text):
        self._hide_tree_tooltip()
        self.tree_tooltip = tk.Toplevel(self.root)
        self.tree_tooltip.wm_overrideredirect(True)
        self.tree_tooltip.wm_geometry(f"+{x+10}+{y+15}")
        label = tk.Label(self.tree_tooltip, text=text, background=ThemeColors.SURFACE_VARIANT, foreground=ThemeColors.ON_SURFACE, relief="solid", borderwidth=1, font=self.status_font, justify=tk.LEFT, padx=8, pady=6)
        label.pack()
    
    def _hide_tree_tooltip(self):
        if self.tree_tooltip:
            self.tree_tooltip.destroy()
            self.tree_tooltip = None
    
    def _on_module_double_click(self, event):
        selected_iids = self.modules_tree.selection()
        if not selected_iids: return
        module_name = selected_iids[0]

        info = next((i for i in self.module_info_cache if i.name == module_name), None)
        if not info: return

        def worker() -> str:
            repo_path = self.manager.root_dir / info.relative_path
            
            try:
                branches_cmd = self.manager._run_git_command(repo_path, ['branch', '-a'])
                all_branches = branches_cmd.stdout.strip().splitlines() if branches_cmd.returncode == 0 else []

                tags_cmd = self.manager._run_git_command(repo_path, ['tag', '--list'])
                all_tags = tags_cmd.stdout.strip().splitlines() if tags_cmd.returncode == 0 else []
                
                found_active_branch = False
                for i, branch in enumerate(all_branches):
                    if branch.strip().startswith('*'):
                        all_branches[i] = f"-> {branch.lstrip('* ')}"
                        found_active_branch = True
                        break
                
                if not found_active_branch and info.current_ref.ref_type == RefType.TAG:
                     for i, tag in enumerate(all_tags):
                        if tag == info.current_ref.name:
                            all_tags[i] = f"-> {tag}"
                            break
                
                error_msg = None
            except GitOperationError as e:
                all_branches, all_tags = [], []
                error_msg = f"获取分支/标签列表时出错: {e}"

            status_map = {'✅': '工作区干净', '⚠️': '有未提交更改', '❌': '错误'}
            status_desc = status_map.get(info.status_icon, '未知')

            details_text = [
                f"模块详细信息: {info.name}",
                f"{'='*40}",
                f"  - 路径: {info.relative_path}",
                f"  - 状态: {status_desc}",
                f"  - 当前引用: {info.ref_display}",
                f"  - 引用类型: {info.current_ref.ref_type.value}",
                f"  - 引用名称: {info.current_ref.name}",
                f"  - Commit HASH: {info.commit_hash}",
            ]

            if info.error_message:
                details_text.append(f"  - 加载错误: {info.error_message}")

            details_text.append("\n" + "="*40 + "\n")

            if error_msg:
                details_text.append(error_msg)
            else:
                details_text.append("🌿 所有分支 (-> 表示当前):")
                if all_branches:
                    details_text.extend([f"  {b.strip()}" for b in all_branches])
                else:
                    details_text.append("  (无或获取失败)")

                details_text.append("\n🏷️ 所有标签 (-> 表示当前):")
                if all_tags:
                    details_text.extend([f"  {t.strip()}" for t in all_tags])
                else:
                    details_text.append("  (无或获取失败)")
            
            return "\n".join(details_text)

        def on_done(result_text: str):
            self._show_result(result_text, f"模块详情 - {info.name}")

        self.log(f"正在为 {module_name} 加载详细信息...", "INFO")
        self._run_in_thread(worker, on_done)

    def _show_main_repo_details(self, event=None):
        """显示主存储库的详细信息"""
        self.log(f"正在为主库 {self.manager.root_dir.name} 加载详细信息...", "INFO")
        
        def on_done(result_text: str):
            self._show_result(result_text, f"主库详情 - {self.manager.root_dir.name}")

        self._run_in_thread(self._get_main_repo_details_worker, on_done)

    def _get_main_repo_details_worker(self) -> str:
        """[工作线程] 获取主存储库的详细Git信息"""
        repo_path = self.manager.root_dir
        
        try:
            current_ref = self.manager.get_current_ref(repo_path)
            is_clean = self.manager.check_working_tree_clean(repo_path)
            status_icon = "✅" if is_clean else "⚠️"
            
            hash_cmd = self.manager._run_git_command(repo_path, ['rev-parse', 'HEAD'], timeout=5)
            commit_hash = hash_cmd.stdout.strip() if hash_cmd.returncode == 0 else "N/A"

            if current_ref.ref_type == RefType.BRANCH:
                ref_display = f"🌿 {current_ref.name}"
            elif current_ref.ref_type == RefType.TAG:
                ref_display = f"🏷️ {current_ref.name}"
            else:
                ref_display = f"🔗 {current_ref.name}"

            branches_cmd = self.manager._run_git_command(repo_path, ['branch', '-a'])
            all_branches = branches_cmd.stdout.strip().splitlines() if branches_cmd.returncode == 0 else []

            tags_cmd = self.manager._run_git_command(repo_path, ['tag', '--list'])
            all_tags = tags_cmd.stdout.strip().splitlines() if tags_cmd.returncode == 0 else []
            
            found_active_branch = False
            for i, branch in enumerate(all_branches):
                if branch.strip().startswith('*'):
                    all_branches[i] = f"-> {branch.lstrip('* ')}"
                    found_active_branch = True
                    break
            
            if not found_active_branch and current_ref.ref_type == RefType.TAG:
                 for i, tag in enumerate(all_tags):
                    if tag == current_ref.name:
                        all_tags[i] = f"-> {tag}"
                        break
            
            error_msg = None
        except GitOperationError as e:
            current_ref = GitRef("Error", RefType.DETACHED)
            ref_display = "获取失败"
            commit_hash = "N/A"
            status_icon = "❌"
            all_branches, all_tags = [], []
            error_msg = f"获取分支/标签列表时出错: {e}"

        status_map = {'✅': '工作区干净', '⚠️': '有未提交更改', '❌': '错误'}
        status_desc = status_map.get(status_icon, '未知')

        details_text = [
            f"主库详细信息: {repo_path.name}",
            f"{'='*40}",
            f"  - 路径: {repo_path}",
            f"  - 状态: {status_desc}",
            f"  - 当前引用: {ref_display}",
            f"  - 引用类型: {current_ref.ref_type.value}",
            f"  - 引用名称: {current_ref.name}",
            f"  - Commit HASH: {commit_hash}",
        ]

        details_text.append("\n" + "="*40 + "\n")

        if error_msg:
            details_text.append(error_msg)
        else:
            details_text.append("🌿 所有分支 (-> 表示当前):")
            if all_branches:
                details_text.extend([f"  {b.strip()}" for b in all_branches])
            else:
                details_text.append("  (无或获取失败)")

            details_text.append("\n🏷️ 所有标签 (-> 表示当前):")
            if all_tags:
                details_text.extend([f"  {t.strip()}" for t in all_tags])
            else:
                details_text.append("  (无或获取失败)")
        
        return "\n".join(details_text)

    def _on_window_configure(self, event):
        if event.widget == self.root and hasattr(self, 'config'):
            self.config.window_geometry = self.root.geometry()

    def log(self, message: str, level: str = "INFO"):
        if not self.root.winfo_exists():
            return
        
        def update_ui():
            self.log_text.config(state=tk.NORMAL)
            timestamp = time.strftime('%H:%M:%S')
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", (level, "TIMESTAMP"))
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            self.status_left_var.set(message[:120] + "..." if len(message) > 120 else message)
        
        self.root.after(0, update_ui)

    def update_progress_from_thread(self, value: int, message: str, operation: str = ""):
        self.root.after(0, self.update_progress, value, message, operation)

    def update_progress(self, value: int, message: str, operation: str = ""):
        if not self.root.winfo_exists(): return
        self.progress_var.set(value)
        self.progress_percent_var.set(f"{int(value)}%")
        self.status_var.set(message)
        if operation:
            self.operation_name_var.set(operation)
        self.root.update_idletasks()

    def _start_operation(self, operation_name: str) -> bool:
        if self.is_operation_running:
            self._show_warning("已有操作正在进行中，请等待其完成后再开始新操作。")
            return False
        
        self.is_operation_running = True
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        self.operation_name_var.set(f"🔄 {operation_name}")
        self.progress_var.set(0)
        self.progress_percent_var.set("0%")
        self._reset_stats()
        self.log(f"=== 开始 {operation_name} ===")
        self.notebook.select(0)
        return True
    
    def _end_operation(self, success_count: int = 0, total_count: int = 0):
        if not self.is_operation_running: return # Avoid ending an already ended operation

        self.is_operation_running = False
        duration = time.time() - self.operation_start_time if self.operation_start_time else 0
        
        self.operation_name_var.set("✅ 操作完成")
        self.progress_var.set(100)
        self.progress_percent_var.set("100%")
        self.status_var.set(f"操作完成，耗时 {duration:.1f}s")
        
        if total_count > 0:
            self.stats['total'].set(str(total_count))
            self.stats['success'].set(str(success_count))
            self.stats['failed'].set(str(total_count - success_count))
        self.stats['time'].set(f"{duration:.1f}s")
        
        self.log(f"=== {self.current_operation} 完成，耗时 {duration:.1f}s ===")
        self.current_operation = None
        self.operation_start_time = None

def main():
    try:
        try:
            # For Python 3.9+
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log', filemode='w', encoding='utf-8')
        except ValueError:
            # For older Python (like 3.7) that doesn't support encoding in basicConfig
            handler = logging.FileHandler('app.log', 'w', 'utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
        mgr = GitSubmoduleManager()
        ModernGitSubmoduleGUI(mgr)
    except Exception as e:
        logging.error(f"应用启动失败: {e}", exc_info=True)
        messagebox.showerror("严重错误", f"应用启动失败，请查看控制台日志和 app.log。\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()