from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    local_dir="/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    local_dir_use_symlinks=False,
)