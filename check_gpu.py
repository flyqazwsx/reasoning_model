"""檢查 GPU 狀態：名稱、總／可用 VRAM（quickstart 驗證步驟使用）"""

import gc

import torch


def main() -> None:
    """輸出 CUDA 可用性與 VRAM 資訊。"""
    print(f"torch {torch.__version__} / CUDA 可用: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[X] 沒有可用的 CUDA GPU（可執行單元／契約測試，但無法訓練）")
        return

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    torch.cuda.empty_cache()
    gc.collect()
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    print(f"[GPU] 名稱: {props.name}")
    print(f"[GPU] 總 VRAM: {total_bytes / 1024**3:.2f} GB")
    print(f"[GPU] 可用 VRAM: {free_bytes / 1024**3:.2f} GB")
    if (total_bytes - free_bytes) / 1024**3 > 0.5:
        print("[!] 已有其他程式占用 VRAM，建議先關閉再訓練")


if __name__ == "__main__":
    main()
