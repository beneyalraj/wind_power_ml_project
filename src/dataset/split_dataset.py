from pathlib import Path
import shutil
import random

DATA_PATH = Path("data/interim/scenario_dataset_v1")
OUTPUT_PATH = Path("data/interim/splits")

TRAIN_DIR = OUTPUT_PATH / "train"
VAL_DIR = OUTPUT_PATH / "validation"
TEST_DIR = OUTPUT_PATH / "test"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


def get_layout_files():
    return list(DATA_PATH.rglob("*.parquet"))


def split_layouts(layout_files):

    random.seed(42)
    random.shuffle(layout_files)

    total = len(layout_files)

    train_split = int(total * 0.7)
    val_split = int(total * 0.85)

    train_files = layout_files[:train_split]
    val_files = layout_files[train_split:val_split]
    test_files = layout_files[val_split:]

    return train_files, val_files, test_files


def copy_files(files, destination):

    for file in files:
        # Get the parent folder name (e.g., 'layout_0') 
        # to prevent overwriting files named 'part-00000.parquet'
        parent_name = file.parent.name 
        new_name = f"{parent_name}_{file.name}"
        
        dst = destination / new_name
        shutil.copy(file, dst)
        print(f"  → Copied {file.name} as {new_name}")


def main():

    print("Finding parquet layouts...")

    layout_files = get_layout_files()

    print(f"Total layouts found: {len(layout_files)}")

    train_files, val_files, test_files = split_layouts(layout_files)

    print(f"Layouts: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    print("Copying train layouts...")
    copy_files(train_files, TRAIN_DIR)

    print("Copying validation layouts...")
    copy_files(val_files, VAL_DIR)

    print("Copying test layouts...")
    copy_files(test_files, TEST_DIR)

    print("Dataset split completed")


if __name__ == "__main__":
    main()