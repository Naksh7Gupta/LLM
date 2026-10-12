from huggingface_hub import login, upload_folder


login()


upload_folder(folder_path="data\\json", repo_id="ThinkNet/HQ-Chat-1K", repo_type="dataset")
