"""
配置管理模組

定義專案的路徑、預設超參數和環境變數配置
"""

import os
from pathlib import Path
from typing import Dict, Any

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ===== 路徑配置 =====

# 資料目錄
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DB_PATH = DATA_DIR / "models.db"
DATASETS_DB_PATH = DATA_DIR / "datasets.db"
INFERENCE_LOGS_DB_PATH = DATA_DIR / "inference_logs.db"
TRAINING_JOBS_DB_PATH = DATA_DIR / "training_jobs.db"

# 快取目錄
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_CACHE_DIR = CACHE_DIR / "models"
DATASETS_CACHE_DIR = CACHE_DIR / "datasets"

# 輸出目錄
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Hugging Face 預設快取路徑
HF_HOME = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# ===== 硬體配置 =====

# GPU 和 VRAM 限制
REQUIRED_VRAM_GB = 12.0  # 強制要求的最小 VRAM
MAX_VRAM_GB = 12.0  # 最大允許的 VRAM 使用量
MIN_SYSTEM_RAM_GB = 16.0  # 建議的最小系統記憶體

# CUDA 配置
CUDA_DEVICE = int(os.getenv("CUDA_VISIBLE_DEVICES", "0"))

# ===== 預設模型配置 =====

# 預設支援的基底模型列表（嚴格遵守 CLAUDE.md 規定）
# 必須使用以下 Thinking/Reasoning 蒸餾模型
DEFAULT_BASE_MODELS = [
    {
        "model_id": "open-r1/OpenR1-Distill-7B",
        "display_name": "OpenR1 Distill 7B (推薦)",
        "architecture": "Llama",
        "parameters": 7_000_000_000,
        "model_size_gb": 14.0,
        "estimated_vram_gb": 3.5,  # 4-bit 量化後
        "description": "OpenAI o1 蒸餾模型，專注推理能力",
        "recommended": True,
    },
    {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "display_name": "DeepSeek-R1 Distill Llama 8B",
        "architecture": "Llama",
        "parameters": 8_000_000_000,
        "model_size_gb": 16.0,
        "estimated_vram_gb": 4.0,
        "description": "DeepSeek-R1 蒸餾至 Llama，強推理能力",
        "recommended": True,
    },
    {
        "model_id": "Qwen/Qwen2.5-8B-Instruct",
        "display_name": "Qwen 2.5 8B Instruct",
        "architecture": "Qwen2.5",
        "parameters": 8_000_000_000,
        "model_size_gb": 16.0,
        "estimated_vram_gb": 4.0,
        "description": "通用指令模型，可用於 reasoning 微調",
        "recommended": True,
    },
]

# ===== 預設資料集配置 =====

# 推薦的 Thinking/Reasoning 訓練資料集（嚴格遵守 CLAUDE.md 規定）
DEFAULT_DATASETS = [
    {
        "dataset_id": "gsm8k",
        "display_name": "GSM8K (數學推理 - 測試用)",
        "dataset_url": "https://huggingface.co/datasets/gsm8k",
        "required_fields": ["question", "answer"],
        "field_mapping": {"input": "question", "output": "answer"},
        "split": "train",
        "description": "數學推理資料集，適合初期測試",
    },
    {
        "dataset_id": "kaist-ai/CoT-Collection",
        "display_name": "ThoughtSource / CoT-Collection",
        "dataset_url": "https://huggingface.co/datasets/kaist-ai/CoT-Collection",
        "required_fields": ["source", "rationale", "target"],
        "field_mapping": {"input": "source", "thinking": "rationale", "output": "target"},
        "split": "train",
        "description": "多樣化思考過程資料集，多領域推理能力",
    },
    {
        "dataset_id": "argilla/distilabel-math-preference-dpo",
        "display_name": "AM-DeepSeek-R1 相關資料",
        "dataset_url": "https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo",
        "required_fields": ["chosen", "rejected"],
        "field_mapping": {"input": "prompt", "output": "chosen"},
        "split": "train",
        "description": "DeepSeek-R1 蒸餾相關資料（數學偏好）",
    },
    # 注意：以下資料集ID可能需要確認或替代
    # {
    #     "dataset_id": "Open-R1/Mixture-of-Thoughts",  # 待確認
    #     "display_name": "Open-R1 Mixture-of-Thoughts (優先推薦)",
    #     "description": "推理鏈混合資料，包含完整思考鏈",
    # },
    # {
    #     "dataset_id": "LongTalk-CoT",  # 待確認或使用 GSM8K 替代
    #     "display_name": "LongTalk-CoT",
    #     "description": "長對話形式的 Chain-of-Thought",
    # },
]

# ===== 訓練超參數預設值 =====

DEFAULT_TRAINING_HYPERPARAMETERS: Dict[str, Any] = {
    # ===== 嚴格遵守 CLAUDE.md 規定的超參數 =====

    # 訓練配置
    "epochs": 3,
    "learning_rate": 2e-4,  # 初始值，若 loss 不收斂可降至 1e-4 或 5e-5
    "per_device_train_batch_size": 1,  # 12GB VRAM 限制，必須為 1
    "gradient_accumulation_steps": 8,  # 範圍 8-16，有效 batch size = 8-16
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "max_grad_norm": 0.3,  # 梯度裁剪

    # LoRA 配置（不可更改）
    "lora_rank": 16,  # 固定值
    "lora_alpha": 16,  # 固定值，與 rank 相同（不是 32！）
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # 必須包含這三個！
    ],
    "lora_bias": "none",  # Unsloth 優化
    "use_rslora": False,
    "use_gradient_checkpointing": "unsloth",  # "unsloth" 使用 30% 更少 VRAM

    # 早停配置
    "early_stopping_patience": 2,

    # 量化配置（必須）
    "load_in_4bit": True,  # 必須為 True
    "bnb_4bit_compute_dtype": "bfloat16",  # 優先使用 bfloat16，若不支援則用 float16
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,

    # 序列與記憶體優化
    "max_seq_length": 2048,  # 起始值，可依任務調整
    "gradient_checkpointing": True,  # 訓練時必開

    # 優化器配置
    "optim": "paged_adamw_8bit",  # 記憶體優化
    "lr_scheduler_type": "cosine",

    # 訓練策略
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",  # 新版 transformers 使用 eval_strategy
    "fp16": False,  # 使用 bfloat16 時設為 False
    "bf16": True,   # 啟用 bfloat16（若 GPU 支援）
    "report_to": "none",
}

# 超參數範圍限制
HYPERPARAMETER_RANGES = {
    "epochs": {"min": 1, "max": 10},
    "learning_rate": {"min": 1e-5, "max": 1e-3},
    "lora_rank": {"min": 8, "max": 64},
    "lora_alpha": {"min": 16, "max": 128},
    "lora_dropout": {"min": 0.0, "max": 0.3},
    "per_device_train_batch_size": {"min": 1, "max": 8},  # 嚴格遵守：必須為 1
    "batch_size": {"min": 1, "max": 8},  # 向後兼容（已棄用，請使用 per_device_train_batch_size）
    "gradient_accumulation_steps": {"min": 1, "max": 16},
    "early_stopping_patience": {"min": 1, "max": 5},
}

# ===== 推論參數預設值 =====

DEFAULT_INFERENCE_PARAMS: Dict[str, Any] = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# 推論參數範圍限制
INFERENCE_PARAM_RANGES = {
    "max_new_tokens": {"min": 1, "max": 2048},
    "temperature": {"min": 0.0, "max": 2.0},
    "top_p": {"min": 0.0, "max": 1.0},
    "top_k": {"min": 0, "max": 100},
    "repetition_penalty": {"min": 1.0, "max": 2.0},
}

# ===== 效能配置 =====

# 下載配置
DOWNLOAD_TIMEOUT_SEC = 600  # 10 分鐘
DOWNLOAD_RETRY_ATTEMPTS = 3
DOWNLOAD_CHUNK_SIZE = 8192  # 8KB

# 推論配置
INFERENCE_TIMEOUT_SEC = 300  # 5 分鐘
INFERENCE_MAX_BATCH_SIZE = 1  # 單一使用者模式

# 訓練配置
TRAINING_LOG_INTERVAL_STEPS = 10  # 每 10 步記錄一次
TRAINING_CHECKPOINT_INTERVAL_EPOCHS = 1  # 每個 epoch 保存 checkpoint
TRAINING_VRAM_WARNING_THRESHOLD_GB = 11.0  # VRAM 警告閾值

# ===== UI 配置 =====

# Gradio 配置
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# UI 更新頻率
UI_PROGRESS_UPDATE_INTERVAL_SEC = 1.0  # 進度條更新間隔
UI_LOG_UPDATE_INTERVAL_SEC = 0.5  # 日誌更新間隔

# ===== 資料庫配置 =====

# SQLite 配置
SQLITE_TIMEOUT = 30  # 秒
SQLITE_CHECK_SAME_THREAD = False

# ===== 日誌配置 =====

# 日誌級別
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ===== 環境變數 =====


# Hugging Face Endpoint (可設定鏡像)
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")

# ===== 驗證規則 =====

# 模型驗證
MODEL_MAX_SIZE_GB = 48.0  # 最大模型大小（原始）
MODEL_MAX_VRAM_GB = MAX_VRAM_GB  # 4-bit 量化後不能超過 12GB

# 資料集驗證
DATASET_MIN_TRAIN_SAMPLES = 100
DATASET_MIN_VAL_SAMPLES = 20
DATASET_VAL_SPLIT_RATIO = 0.2  # 20% 驗證集

# 提示詞驗證
PROMPT_MAX_LENGTH_TOKENS = 2048
PROMPT_WARNING_LENGTH_TOKENS = 1024

# ===== 錯誤碼 =====

ERROR_CODES = {
    "E001": "模型 ID 格式無效",
    "E002": "找不到模型",
    "E003": "模型超出 VRAM 限制",
    "E004": "網路連線失敗",
    "E005": "下載中斷",
    "E006": "磁碟空間不足",
    "E007": "檔案損壞",
    "E008": "資料集格式不相容",
    "E009": "推論 OOM",
    "E010": "訓練 OOM",
    "E011": "已有訓練進行中",
    "E012": "Checkpoint 不存在或損壞",
    "E013": "Adapter 檔案遺失",
    "E014": "資料庫操作失敗",
    "E015": "參數驗證失敗",
}


def get_error_message(error_code: str) -> str:
    """取得錯誤訊息"""
    return ERROR_CODES.get(error_code, "未知錯誤")


def validate_hyperparameters(hyperparams: Dict[str, Any]) -> tuple[bool, str]:
    """
    驗證訓練超參數是否在合法範圍內

    Returns:
        (is_valid, error_message)
    """
    for param, value in hyperparams.items():
        if param in HYPERPARAMETER_RANGES:
            range_config = HYPERPARAMETER_RANGES[param]
            if value < range_config["min"] or value > range_config["max"]:
                return False, f"{param} 必須在 {range_config['min']} 到 {range_config['max']} 範圍內"
    return True, ""


def validate_inference_params(params: Dict[str, Any]) -> tuple[bool, str]:
    """
    驗證推論參數是否在合法範圍內

    Returns:
        (is_valid, error_message)
    """
    for param, value in params.items():
        if param in INFERENCE_PARAM_RANGES:
            range_config = INFERENCE_PARAM_RANGES[param]
            if value < range_config["min"] or value > range_config["max"]:
                return False, f"{param} 必須在 {range_config['min']} 到 {range_config['max']} 範圍內"
    return True, ""

# ===== 002：推薦、訓練筆數與微調模型管理 =====

# 推薦篩選：預估訓練 VRAM 不得超過可用 VRAM 的比例（保留餘裕避免臨界 OOM）
RECOMMEND_VRAM_RATIO = 0.9

# 訓練使用筆數（FR-009）
SAMPLE_COUNT_DEFAULT = 2000
SAMPLE_COUNT_MIN = 100

# 微調模型成品位置（與 OUTPUTS_DIR 相同，刪除時須驗證路徑位於其下）
ADAPTER_OUTPUT_DIR = OUTPUTS_DIR

# 內建評測題庫（FR-016a）
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "src" / "resources" / "eval_questions.json"
EVAL_MIN_QUESTIONS = 10
