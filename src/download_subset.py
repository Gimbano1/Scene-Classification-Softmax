from collections import defaultdict
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download, list_repo_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "intel_subset"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def main(per_class=80):
    """Download a small balanced image subset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = list_repo_files("sfarrukhm/intel-image-classification", repo_type="dataset")

    by_class = defaultdict(list)
    for file in files:
        for class_name in CLASSES:
            prefix = f"data/seg_train/{class_name}/"
            if file.startswith(prefix) and file.lower().endswith(".jpg"):
                by_class[class_name].append(file)

    for class_name in CLASSES:
        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        selected = sorted(by_class[class_name])[:per_class]

        for file in selected:
            target = class_dir / Path(file).name
            if target.exists():
                continue

            downloaded = hf_hub_download(
                repo_id="sfarrukhm/intel-image-classification",
                repo_type="dataset",
                filename=file,
                local_dir=PROJECT_ROOT / ".cache" / "hf_files",
            )
            shutil.copy2(downloaded, target)

        print(f"{class_name}: {len(list(class_dir.glob('*.jpg')))} images")


if __name__ == "__main__":
    main()
