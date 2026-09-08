import os
import shutil
import subprocess
import json

def prepare_and_push():
    print("Preparing Kaggle Kernel Package...")
    os.makedirs("kaggle_runner", exist_ok=True)
    os.environ["PYTHONUTF8"] = "1"
    
    # 1. Update kernel-metadata.json for standalone script execution
    metadata = {
        "id": "dauletanekesh/kazakh-gpu-runner-nb",
        "title": "kazakh-gpu-runner-nb",
        "code_file": "diagnostic_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "accelerator": "gpu_t4_x2",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    with open("kaggle_runner/kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 2. Copy seed human data into kaggle_runner directory so it is uploaded with the script
    if os.path.exists("data/seed_human_reviews_1k.json"):
        shutil.copy("data/seed_human_reviews_1k.json", "kaggle_runner/seed_human_reviews_1k.json")
        print("Copied seed_human_reviews_1k.json to kaggle_runner/")

    # 3. Check kaggle CLI
    print("\nPushing kernel to Kaggle via CLI: 'kaggle kernels push -p kaggle_runner'")
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", "kaggle_runner"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        print("Output:", res.stdout)
        print("Kaggle push SUCCESSFUL!")
    except subprocess.CalledProcessError as e:
        print("Kaggle push failed:", e.stderr)
        print("Standard output:", e.stdout)

if __name__ == "__main__":
    prepare_and_push()
