from huggingface_hub import snapshot_download

# Download all files from a specific dataset
local_dir = snapshot_download(
    local_dir="/data3/silvio/sisap_dataset",
    repo_id="SISAP-Challenges/SISAP2026",
    repo_type="dataset",
    allow_patterns="fiqa-dev/*"
)