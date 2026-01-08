# -*- coding: utf-8 -*-
"""
backend/core/job_manager.py
学習ジョブのプロセス管理、ログ監視、排他制御を行う共通基盤モジュール。
Text/Image/Audio などの各エンジンから利用されます。
"""
import sys
import os
import json
import time
import subprocess
import threading
import queue
import psutil
import uuid
from pathlib import Path
from lora_config import settings
from backend.core.env_snapshot import snapshot_env
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from lora_config import settings

class JobManager:
    """
    サブプロセスの実行、監視、ログ収集、強制停止を一元管理するクラス。
    シングルトンとして利用することを想定。
    """
    def __init__(self):
        # ジョブ状態の保持
        self.current_job = {
            "proc": None,       # subprocess.Popen
            "log_queue": None,  # queue.Queue
            "status": "idle",   # idle, running, completed, failed, stopped
            "job_id": None,
            "logs": [],
            "stderr_logs": [],
            "run_dir": None,
            "params": {},       # 実行時のパラメータ（参照用）
            "log_file": None    # Path
        }
        
        # 排他制御用ロックファイル
        self.pid_lock_file = settings.base_dir / "active_job.json"
        
        # 起動時に残留ロックファイルをチェックしてクリーンアップ
        self._check_and_clean_pid_file()

    def _check_and_clean_pid_file(self) -> bool:
        """
        起動時にPIDファイルをチェックし、死んでいるプロセスのロックなら解除する。
        実行中なら True, それ以外は False を返す。
        """
        if self.pid_lock_file.exists():
            try:
                data = json.loads(self.pid_lock_file.read_text(encoding="utf-8"))
                pid = data.get("pid")
                if pid and psutil.pid_exists(pid):
                    try:
                        p = psutil.Process(pid)
                        # プロセス名に python が含まれていれば自身のジョブとみなす（簡易判定）
                        if "python" in p.name().lower():
                            print(f"[JobManager] 既存の学習プロセス(PID: {pid})を検出しました。")
                            # 状態を復元（完全に復元はできないが、Runningとしては認識させる）
                            self.current_job["status"] = "running"
                            self.current_job["job_id"] = data.get("job_id", "unknown")
                            return True 
                    except:
                        pass
                
                print("[JobManager] 不正終了したジョブのロックファイルを削除します。")
                self.pid_lock_file.unlink()
            except Exception as e:
                print(f"[JobManager] PIDファイル読み込みエラー (削除します): {e}")
                self.pid_lock_file.unlink(missing_ok=True)
        return False

    def is_active(self) -> bool:
        """現在ジョブが実行中かどうかを判定"""
        if self.current_job["status"] == "running":
            return True
        if self.pid_lock_file.exists():
            return self._check_and_clean_pid_file()
        return False

    def start_job(
        self, 
        cmd: List[str], 
        params: Dict[str, Any], 
        cwd: Path,
        env: Dict[str, str] = None,
        log_prefix: str = "train"
    ) -> Dict[str, Any]:
        """
        新しいジョブ（サブプロセス）を開始する。
        
        Args:
            cmd: 実行するコマンドライン引数のリスト
            params: フロントエンド表示用のパラメータ情報
            cwd: カレントディレクトリ
            env: 環境変数（Noneの場合は os.environ.copy() を使用）
            log_prefix: ログファイル名の接頭辞
            
        Returns:
            開始したジョブの情報辞書
        """
        if self.is_active():
            raise RuntimeError("ジョブが既に実行中です。")

        # 状態初期化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # runs/<job_id>/ に再現性用のスナップショットを保存
        try:
            run_dir = (settings.runs_root / job_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            # config.json: 実行コマンドや表示パラメータ
            (run_dir / "config.json").write_text(
                json.dumps({
                    "job_id": job_id,
                    "created_at": timestamp,
                    "cwd": str(cwd),
                    "cmd": cmd,
                    "params": params,
                    "log_prefix": log_prefix,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            # env.json: 実行環境スナップショット（packages等）
            (run_dir / "env.json").write_text(
                json.dumps(snapshot_env(), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            # 保存失敗でもジョブは継続（運用A: 失敗で壊さない）
            pass

        self.current_job = {
            "proc": None,
            "log_queue": queue.Queue(),
            "status": "running",
            "job_id": job_id,
            "logs": [],
            "stderr_logs": [],
            "params": params,
            "log_file": settings.logs_dir / f"{log_prefix}_{timestamp}.log"
        }

        # 環境変数の準備
        if env is None:
            env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        print(f"[JobManager] Running command: {' '.join(cmd)}")

        try:
            # プロセス起動
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, # 標準エラーは分離（UIでstdout/stderr表示）
                text=True,
                bufsize=1,
                encoding='utf-8',
                cwd=str(cwd),
                env=env
            )
            self.current_job["proc"] = proc
            
            # ロックファイル作成
            try:
                lock_info = {
                    "pid": proc.pid,
                    "job_id": job_id,
                    "start_time": timestamp,
                    "log_file": str(self.current_job["log_file"])
                }
                self.pid_lock_file.write_text(json.dumps(lock_info), encoding="utf-8")
            except Exception as e:
                print(f"[JobManager] PID Lock作成失敗: {e}")

            # 監視スレッド開始
            threading.Thread(target=self._monitor_process, args=(proc,), daemon=True).start()
            
            return {
                "status": "started",
                "job_id": job_id,
                "log_file": str(self.current_job["log_file"])
            }

        except Exception as e:
            self.current_job["status"] = "failed"
            self.current_job["logs"].append(f"プロセスの起動に失敗しました: {e}")
            print(f"[JobManager] Failed to start process: {e}")
            raise e

    def stop_job(self, job_id: str | None = None):
        """実行中のジョブを強制停止する"""
        if job_id and self.current_job.get("job_id") and job_id != self.current_job.get("job_id"):
            # 指定ジョブが現行ジョブでない場合は no-op
            return
        if self.current_job["proc"] and self.current_job["status"] == "running":
            self.current_job["status"] = "stopped"
            try:
                self.current_job["proc"].terminate()
            except Exception:
                pass
            self.current_job["logs"].append("ユーザーによってジョブが停止されました。")
            return

        # プロセスオブジェクトが無いがロックファイルがある場合の強制クリーニング
        if self.pid_lock_file.exists():
            try:
                data = json.loads(self.pid_lock_file.read_text(encoding="utf-8"))
                pid = data.get("pid")
                if pid and psutil.pid_exists(pid):
                    p = psutil.Process(pid)
                    p.terminate()
                    print(f"[JobManager] PID {pid} を強制停止しました。")
                self.pid_lock_file.unlink()
            except Exception as e:
                print(f"[JobManager] 強制停止エラー: {e}")

    def get_status(self, job_id: str | None = None) -> Dict[str, Any]:
        """現在のジョブステータスと最新のログを取得"""
        # キューからログを取り出してリストに移す
        if self.current_job["log_queue"]:
            while not self.current_job["log_queue"].empty():
                try:
                    line = self.current_job["log_queue"].get_nowait()
                    # キューから取り出す処理は _monitor_process 側でもリスト追加しているが、
                    # リアルタイム性を高めるためにUI側ポーリング時に吸い出す設計も可。
                    # 今回は _monitor_process が logs に append しているので、
                    # log_queue は「UIへの差分通知用」あるいは「使わなくても良い」が、
                    # 既存設計を踏襲し、ここでは単に空にするだけにしておく（logsリストを参照させる）。
                    pass 
                except queue.Empty:
                    break

        return {
            "job_id": self.current_job["job_id"],
            "status": self.current_job["status"],
            "logs": self.current_job["logs"][-100:] # 最新100行を返す
        }


    def _monitor_process(self, proc):
        """
        バックグラウンドで実行されるプロセス監視メソッド。
        stdout / stderr を読み取り、ファイルとメモリに書き込む。
        """
        log_path = self.current_job.get("log_file")

        f_log = None
        try:
            if log_path:
                f_log = open(log_path, "a", encoding="utf-8")
        except Exception as e:
            print(f"[JobManager] Failed to open log file: {e}")

        import threading

        def _read_stream(stream, tag: str, store_key: str):
            try:
                if stream is None:
                    return
                for line in iter(stream.readline, ""):
                    if not line:
                        continue
                    line_str = line.rstrip("\n")
                    line_out = f"{tag}{line_str}" if tag else line_str

                    try:
                        self.current_job["log_queue"].put(line_out)
                    except Exception:
                        pass

                    try:
                        self.current_job.setdefault(store_key, []).append(line_out)
                    except Exception:
                        pass

                    # 統合ログ（従来互換）
                    try:
                        self.current_job.setdefault("logs", []).append(line_out)
                    except Exception:
                        pass

                    if f_log:
                        try:
                            f_log.write(line_out + "\n")
                            f_log.flush()
                        except Exception:
                            pass

                    # サーバーコンソールにも表示
                    try:
                        print(f"[SubProcess] {line_out}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"[JobManager] Error reading {store_key}: {e}")

        t_out = threading.Thread(target=_read_stream, args=(proc.stdout, "", "logs"), daemon=True)
        t_err = threading.Thread(target=_read_stream, args=(proc.stderr, "[stderr] ", "stderr_logs"), daemon=True)

        try:
            t_out.start()
            t_err.start()
            t_out.join()
            t_err.join()
        finally:
            # 終了処理
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

            return_code = proc.wait()

            if f_log:
                try:
                    f_log.close()
                except Exception:
                    pass

            # ログ終了メッセージ
            if return_code == 0 and self.current_job["status"] == "running":
                self.current_job["status"] = "completed"
            elif self.current_job["status"] == "running":
                self.current_job["status"] = "failed"
                self.current_job["logs"].append(f"プロセスがエラー終了しました（code={return_code}）")



# -----------------------------
# シングルトン（既存ルート互換）
# -----------------------------
# 既存実装は `from backend.core.job_manager import job_manager` を前提にしているため
# インスタンスを公開して互換性を保つ。
job_manager = JobManager()


# -----------------------------
# シングルトン（既存ルート互換）
# -----------------------------
# 既存実装は `from backend.core.job_manager import job_manager` を前提にしているため
# インスタンスを公開して互換性を保つ。
job_manager = JobManager()
