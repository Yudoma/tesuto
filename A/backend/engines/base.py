# -*- coding: utf-8 -*-
"""
backend/engines/base.py
各モダリティ（Text, Image, Audio）のエンジンが実装すべき基底クラス。
共通のインターフェースを定義することで、APIルーターからの呼び出しを統一します。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseEngine(ABC):
    """
    LoRA Factoryの各モダリティエンジンの基底クラス。
    学習ジョブの制御、推論モデルの管理などの共通インターフェースを定義します。
    """

    @abstractmethod
    def start_training(self, base_model: str, dataset: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        学習ジョブを開始する。
        backend/core/job_manager.py を使用してサブプロセスを起動することを推奨。
        
        Args:
            base_model (str): ベースモデル名（またはパス）
            dataset (str): データセット名（またはパス）
            params (Dict[str, Any]): 学習パラメータ全般

        Returns:
            Dict[str, Any]: 開始したジョブの情報（status, job_id等）
        """
        pass

    @abstractmethod
    def stop_training(self) -> Dict[str, str]:
        """
        現在実行中の学習ジョブを停止する。

        Returns:
            Dict[str, str]: 停止要求の結果ステータス
        """
        pass

    @abstractmethod
    def get_training_status(self) -> Dict[str, Any]:
        """
        現在の学習ジョブのステータスとログを取得する。

        Returns:
            Dict[str, Any]: { "status": str, "logs": List[str], "job_id": str }
        """
        pass

    @abstractmethod
    def get_training_history(self) -> Dict[str, Any]:
        """
        過去の学習履歴を取得する。

        Returns:
            Dict[str, Any]: { "history": List[Dict] }
        """
        pass

    @abstractmethod
    def load_inference_model(self, base_model: str, adapter_path: Optional[str] = None) -> Dict[str, Any]:
        """
        検証/推論用のモデルをメモリにロードする。
        
        Args:
            base_model (str): ベースモデル名
            adapter_path (Optional[str]): LoRAアダプタのパス

        Returns:
            Dict[str, Any]: ロード結果ステータス
        """
        pass

    @abstractmethod
    def unload_inference_model(self) -> Dict[str, str]:
        """
        検証用モデルをアンロードし、VRAMを解放する。

        Returns:
            Dict[str, str]: アンロード結果ステータス
        """
        pass

    @abstractmethod
    def is_inference_model_loaded(self) -> bool:
        """
        検証用モデルがロードされているか判定する。

        Returns:
            bool: ロード済みならTrue
        """
        pass