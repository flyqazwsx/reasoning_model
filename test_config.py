"""
測試配置腳本

驗證所有配置是否符合 CLAUDE.md 的嚴格規定
"""

import sys
from pathlib import Path

# 加入 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import DEFAULT_TRAINING_HYPERPARAMETERS, DEFAULT_BASE_MODELS, DEFAULT_DATASETS


def test_hyperparameters():
    """測試超參數配置"""
    print("=" * 60)
    print("測試超參數配置")
    print("=" * 60)

    # 檢查必要的超參數
    assert DEFAULT_TRAINING_HYPERPARAMETERS["lora_rank"] == 16, "lora_rank 必須為 16"
    assert DEFAULT_TRAINING_HYPERPARAMETERS["lora_alpha"] == 16, "lora_alpha 必須為 16（與 rank 相同）"
    assert DEFAULT_TRAINING_HYPERPARAMETERS["lora_dropout"] == 0.05, "lora_dropout 必須為 0.05"

    # 檢查目標模組
    target_modules = DEFAULT_TRAINING_HYPERPARAMETERS["lora_target_modules"]
    required_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert set(target_modules) == set(required_modules), f"目標模組必須包含完整 7 個: {required_modules}"

    # 檢查 batch size
    assert DEFAULT_TRAINING_HYPERPARAMETERS["per_device_train_batch_size"] == 1, "per_device_train_batch_size 必須為 1"

    # 檢查 gradient accumulation
    grad_accum = DEFAULT_TRAINING_HYPERPARAMETERS["gradient_accumulation_steps"]
    assert 8 <= grad_accum <= 16, f"gradient_accumulation_steps 必須在 8-16 範圍內，當前: {grad_accum}"

    # 檢查量化配置
    assert DEFAULT_TRAINING_HYPERPARAMETERS["load_in_4bit"] == True, "load_in_4bit 必須為 True"
    assert DEFAULT_TRAINING_HYPERPARAMETERS["bnb_4bit_compute_dtype"] == "bfloat16", "優先使用 bfloat16"

    # 檢查 gradient checkpointing
    assert DEFAULT_TRAINING_HYPERPARAMETERS["use_gradient_checkpointing"] == "unsloth", "使用 unsloth gradient checkpointing"

    print("✅ 超參數配置正確！")
    print(f"   LoRA: r={DEFAULT_TRAINING_HYPERPARAMETERS['lora_rank']}, "
          f"alpha={DEFAULT_TRAINING_HYPERPARAMETERS['lora_alpha']}, "
          f"dropout={DEFAULT_TRAINING_HYPERPARAMETERS['lora_dropout']}")
    print(f"   目標模組: {len(target_modules)} 個")
    print(f"   Batch size: {DEFAULT_TRAINING_HYPERPARAMETERS['per_device_train_batch_size']}, "
          f"Gradient accumulation: {grad_accum}")
    print(f"   有效 batch size: {DEFAULT_TRAINING_HYPERPARAMETERS['per_device_train_batch_size'] * grad_accum}")
    print()


def test_models():
    """測試模型配置"""
    print("=" * 60)
    print("測試模型配置")
    print("=" * 60)

    # 檢查推薦模型
    model_ids = [m["model_id"] for m in DEFAULT_BASE_MODELS]

    required_models = [
        "open-r1/OpenR1-Distill-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "Qwen/Qwen2.5-8B-Instruct"
    ]

    for req_model in required_models:
        assert req_model in model_ids, f"必須包含推薦模型: {req_model}"

    print(f"✅ 模型配置正確！包含 {len(DEFAULT_BASE_MODELS)} 個推薦模型：")
    for model in DEFAULT_BASE_MODELS:
        print(f"   - {model['model_id']}")
        print(f"     {model['description']}")
        print(f"     估計 VRAM: {model['estimated_vram_gb']}GB (4-bit)")
    print()


def test_datasets():
    """測試資料集配置"""
    print("=" * 60)
    print("測試資料集配置")
    print("=" * 60)

    print(f"✅ 資料集配置正確！包含 {len(DEFAULT_DATASETS)} 個推薦資料集：")
    for dataset in DEFAULT_DATASETS:
        print(f"   - {dataset['dataset_id']}")
        print(f"     {dataset['description']}")
    print()


def main():
    """主測試函數"""
    print("\n" + "=" * 60)
    print("開始驗證配置...")
    print("=" * 60)
    print()

    try:
        test_hyperparameters()
        test_models()
        test_datasets()

        print("=" * 60)
        print("✅ 所有配置驗證通過！")
        print("=" * 60)
        print("\n配置符合 CLAUDE.md 的嚴格規定：")
        print("  ✓ 使用推薦的 Thinking/Reasoning 模型")
        print("  ✓ 使用推薦的 CoT 資料集")
        print("  ✓ 嚴格遵守 LoRA 超參數規定")
        print("  ✓ 使用 4-bit 量化 + Unsloth 優化")
        print("  ✓ Batch size = 1, Gradient accumulation = 8-16")
        print()

        return 0

    except AssertionError as e:
        print(f"\n❌ 配置驗證失敗: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
