from huggingface_hub import login, hf_hub_download

# --- CONFIG ---
REPO_ID = "unsloth/Llama-3.2-1B-Instruct-GGUF"
FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
LOCAL_DIR = "./downloaded_resources"

# --- DOWNLOAD ---
path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=LOCAL_DIR,
)

print("Download complete.\nFile saved to:", path)
