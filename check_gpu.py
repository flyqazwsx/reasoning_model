"""檢查 GPU 記憶體使用狀況"""
import torch
import gc

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"🎮 GPU 資訊:")
    print(f"  名稱: {props.name}")
    print(f"  總記憶體: {props.total_memory / 1024**3:.2f} GB")

    # 清理快取
    torch.cuda.empty_cache()
    gc.collect()

    # 檢查已分配記憶體
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    free = (props.total_memory - torch.cuda.memory_reserved(device)) / 1024**3

    print(f"\n💾 記憶體使用狀況:")
    print(f"  已分配: {allocated:.2f} GB")
    print(f"  已保留: {reserved:.2f} GB")
    print(f"  可用: {free:.2f} GB")

    if allocated > 0.5:
        print(f"\n⚠️  警告: GPU 已有 {allocated:.2f} GB 被占用")
        print("   建議重新啟動應用程式或執行以下命令清理:")
        print("   python -c \"import torch; torch.cuda.empty_cache()\"")
else:
    print("❌ 沒有可用的 CUDA GPU")
