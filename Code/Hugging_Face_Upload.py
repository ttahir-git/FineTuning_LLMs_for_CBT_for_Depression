from huggingface_hub import HfApi, create_repo, whoami
import os
import time

# Initialize the Hugging Face API (no need to pass token here)
api_token = "***"
api = HfApi()

# Set the repository name (ensure it follows the format "username/repo_name")
username = whoami(api_token)["name"]
repo_name = f"{username}/Qwen_7b_CBT_Depression_Finetune_Nov_15"
local_folder_path = "/Users/TalhaTahir/Desktop/Qwen_CBT_Nov_15"  

# Check if the repository exists, create it if it doesn't
def ensure_repo_exists(api, repo_name, token):
    try:
        # Try to get repository info
        repo_info = api.repo_info(repo_id=repo_name, repo_type="model", token=token)
        print(f"Repository '{repo_name}' already exists.")
    except Exception as e:
        # If the repository doesn't exist, create it
        print(f"Repository not found. Creating new repository '{repo_name}'.")
        create_repo(repo_name, repo_type="model", token=token)
        print(f"Repository '{repo_name}' created successfully.")
        # Adding delay to ensure the repository is available
        time.sleep(30)

# Ensure repository exists before uploading files
ensure_repo_exists(api, repo_name, api_token)

# Function to upload a single file with retry logic
def upload_file(local_file_path, path_in_repo):
    max_retries = 5
    retry_delay = 60 

    for attempt in range(max_retries):
        try:
            print(f"Uploading {local_file_path} - Attempt {attempt + 1} of {max_retries}")
            url = api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_name,
                repo_type="model",
                token=api_token  
            )
            print(f"File {local_file_path} uploaded successfully. URL: {url}")
            return True
        except Exception as e:
            print(f"Upload failed for {local_file_path}: {e}")
            if "404" in str(e):
                # Recheck repository existence before retrying
                print("Rechecking if the repository exists...")
                ensure_repo_exists(api, repo_name, api_token)
            elif "403" in str(e):
                print(f"Forbidden error (403). Check if the token has sufficient permissions.")
                return False
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Upload failed for {local_file_path}.")
                return False

# Iterate through all files in the folder and upload them
for root, dirs, files in os.walk(local_folder_path):
    for file in files:
        local_file_path = os.path.join(root, file)
        path_in_repo = os.path.relpath(local_file_path, local_folder_path)
        
        if upload_file(local_file_path, path_in_repo):
            print(f"Successfully uploaded {local_file_path}")
        else:
            print(f"Failed to upload {local_file_path}")

print("Upload process completed.")
