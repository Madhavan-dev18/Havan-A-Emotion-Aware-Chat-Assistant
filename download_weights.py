import os
import shutil

src_dir = r"m:\Personal\Workspace\havan-vision\Emotion-Aware-AI-Chat-Bot\frontend\node_modules\@vladmandic\face-api\model"
dest_dir = r"m:\Personal\Workspace\havan-vision\Emotion-Aware-AI-Chat-Bot\frontend\public\models"

os.makedirs(dest_dir, exist_ok=True)

print("Copying local weights from node_modules...")
for item in os.listdir(src_dir):
    src_file = os.path.join(src_dir, item)
    dest_file = os.path.join(dest_dir, item)
    if os.path.isfile(src_file):
        shutil.copy2(src_file, dest_file)
        print(f"Copied {item}")

print("Local copy sequence completed successfully.")
