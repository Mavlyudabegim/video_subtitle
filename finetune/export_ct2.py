"""Convert the fine-tuned model to CTranslate2 format, which faster-whisper (and the app) needs,
and optionally upload it to the Hugging Face Hub.

    python finetune/export_ct2.py --model /kaggle/working/whisper-small-uz \
        --out /kaggle/working/whisper-small-uz-ct2 --upload <your-username>/whisper-small-uz-ct2
"""
import argparse
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned Transformers model folder or Hub id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upload", default=None, metavar="REPO_ID")
    args = ap.parse_args()

    copy_files = [f for f in ["tokenizer.json", "preprocessor_config.json"]
                  if (Path(args.model) / f).exists()]
    cmd = ["ct2-transformers-converter", "--model", args.model, "--output_dir", args.out,
           "--quantization", "float16", "--force"]
    if copy_files:
        cmd += ["--copy_files", *copy_files]
    subprocess.run(cmd, check=True)
    print(f"CTranslate2 model written to {args.out}")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.upload)
        print(f"Uploaded to https://huggingface.co/{args.upload}")


if __name__ == "__main__":
    main()