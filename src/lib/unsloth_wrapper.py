"""
Unsloth API 封裝層

封裝 FastLanguageModel 載入與 PEFT 配置
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 重要：Unsloth 要求必須先導入 unsloth 模組本身（早於 transformers／peft／trl）。
# 沒有 NVIDIA GPU 的開發機無法安裝 Unsloth，因此匯入失敗時不中斷，
# 改為在實際載入模型（load_model_4bit 等）時才報錯，讓單元／契約測試可在無 GPU 環境執行。
try:
    import unsloth  # noqa: F401

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning("未安裝 Unsloth，僅能執行不需 GPU 的功能與測試")


def load_model_4bit(
    model_id: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    dtype: Optional[str] = None,
):
    """
    載入 4-bit 量化模型（嚴格遵守 CLAUDE.md 規定）

    Args:
        model_id: Hugging Face model ID
        max_seq_length: 最大序列長度
        load_in_4bit: 是否使用 4-bit 量化（必須為 True）
        dtype: 資料型別，None 為自動偵測（優先 bfloat16）

    Returns:
        (model, tokenizer)
    """
    try:
        import gc

        import torch
        from unsloth import FastLanguageModel

        # 清理 GPU 快取，確保有足夠記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(
                "GPU 記憶體已清理，總量: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB"
            )

        # 優先使用 bfloat16（若 GPU 支援），否則 float16
        if dtype is None:
            # 檢查 GPU 是否支援 bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = None  # Unsloth 會自動選擇 bfloat16
                logger.info("GPU 支援 bfloat16，使用 bfloat16 進行訓練")
            else:
                dtype = None  # Unsloth 會自動選擇 float16
                logger.info("GPU 不支援 bfloat16，使用 float16 進行訓練")

        logger.info(f"開始載入模型: {model_id}")
        logger.info(f"配置: max_seq_length={max_seq_length}, load_in_4bit={load_in_4bit}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # 本應用程式僅支援公開模型，不傳入存取權杖（FR-017a）
        )

        logger.info(f"成功載入模型: {model_id} (4-bit 量化)")

        # 顯示模型所在設備
        if hasattr(model, "device"):
            logger.info(f"模型設備: {model.device}")

        return model, tokenizer

    except ImportError:
        logger.error("Unsloth 未安裝！必須安裝 Unsloth 才能繼續。")
        logger.error(
            '請執行: uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
        )
        raise ImportError("Unsloth 是此專案的必要依賴。請依照 CLAUDE.md 安裝 Unsloth。")


def setup_lora_config(
    model,
    lora_rank: int = 16,
    lora_alpha: int = 16,  # 必須與 rank 相同！
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    use_gradient_checkpointing: str = "unsloth",
    max_seq_length: int = 2048,
):
    """
    設定 LoRA 配置（嚴格遵守 CLAUDE.md 規定）

    Args:
        model: 基底模型
        lora_rank: LoRA rank（固定值 16）
        lora_alpha: LoRA alpha（固定值 16，與 rank 相同）
        lora_dropout: LoRA dropout（固定值 0.05）
        target_modules: 目標模組列表（預設包含完整 6 個模組）
        use_gradient_checkpointing: "unsloth" 使用 30% 更少 VRAM
        max_seq_length: 最大序列長度

    Returns:
        配置好 LoRA 的模型
    """
    try:
        from unsloth import FastLanguageModel

        # 嚴格遵守規定：必須包含完整 6 個目標模組
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",  # 這三個必須包含！
            ]

        logger.info(f"配置 LoRA: r={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"目標模組: {target_modules}")

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",  # Unsloth 優化
            use_gradient_checkpointing=use_gradient_checkpointing,  # "unsloth" 或 True
            random_state=3407,
            max_seq_length=max_seq_length,
            use_rslora=False,  # 不使用 RSLoRA
            loftq_config=None,  # 不使用 LoftQ
        )

        logger.info(
            f"LoRA 配置完成: r={lora_rank}, alpha={lora_alpha}, targets={len(target_modules)}"
        )
        return model

    except ImportError:
        logger.error("Unsloth 未安裝！必須安裝 Unsloth 才能繼續。")
        raise ImportError("Unsloth 是此專案的必要依賴。請依照 CLAUDE.md 安裝 Unsloth。")


def enable_inference_mode(model):
    """啟用推論模式 (faster inference)"""
    try:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
        logger.info("已啟用 Unsloth 快速推論模式")
    except Exception:
        model.eval()
        logger.info("已啟用標準推論模式")

    return model


def load_adapter_4bit(base_model_id: str, adapter_path: str, max_seq_length: int = 2048):
    """
    載入 4-bit 基底模型並掛上訓練好的 LoRA adapter（推論用）

    每次都重新載入一份基底模型再掛 adapter，不會改動已快取的純基底模型，
    因此基底與微調模型的推論結果可以放心對照。

    Args:
        base_model_id: 基底模型 Hugging Face ID（須已下載）
        adapter_path: 訓練產出的 adapter 資料夾
        max_seq_length: 最大序列長度

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = load_model_4bit(
        model_id=base_model_id, max_seq_length=max_seq_length, load_in_4bit=True
    )
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("PEFT 是此專案的必要依賴，請依照 README 重新安裝相依套件。")

    logger.info(f"掛載 LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return model, tokenizer
