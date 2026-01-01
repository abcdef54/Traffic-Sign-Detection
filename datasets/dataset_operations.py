import os
import shutil
from tqdm import tqdm
from collections import Counter, defaultdict
import glob
import cv2
from transaction import DatasetTransaction
from PIL import Image
import imagehash
import yaml
import matplotlib.pyplot as plt
import random

def merge_all_splits(workdir):
    """
    Merges train/val/test splits into a single 'all_data' folder
    WITHIN the transactional directory (workdir).
    """
    # 1. Define Destinations relative to the transaction folder
    # We will create a new folder 'all_data' inside the temp dir
    dest_root = os.path.join(workdir, "all")
    img_dst = os.path.join(dest_root, "images")
    lbl_dst = os.path.join(dest_root, "labels")
    
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    subsets = ['train', 'val', 'valid', 'test']
    count = 0

    # 2. Iterate through splits inside the temp dir
    for split in subsets:
        split_dir = os.path.join(workdir, split)
        img_src_dir = os.path.join(split_dir, 'images')
        lbl_src_dir = os.path.join(split_dir, 'labels')

        # Skip if this split doesn't exist in the temp dir
        if not os.path.exists(img_src_dir):
            print(f'{split} does not exist in temp dir, skipping....')
            continue

        files = os.listdir(img_src_dir)

        for f in tqdm(files, desc=f"Merging {split}"):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            src_img = os.path.join(img_src_dir, f)

            # ---- Handle Name Collision Safely ----
            # Prefix filename with split name (e.g., train_img01.jpg)
            base, ext = os.path.splitext(f)
            new_name = f"{split}_{base}{ext}"
            dst_img_path = os.path.join(img_dst, new_name)

            # Move Image
            shutil.move(src_img, dst_img_path)

            # ---- Move Label if exists ----
            label_file = base + '.txt'
            src_lbl = os.path.join(lbl_src_dir, label_file)
            if os.path.exists(src_lbl):
                dst_lbl_path = os.path.join(lbl_dst, f"{split}_{label_file}")
                shutil.move(src_lbl, dst_lbl_path)

            count += 1

        # 3. Remove the empty split folder from the temp dir
        # This is safe because we are in a transaction!
        shutil.rmtree(split_dir, ignore_errors=True)

    print(f"✅ Merged {count} images into {dest_root}")



def remove_class(dataset_all_dir, classes: set[int]|int):
    if isinstance(classes, int): classes = [classes]
    print(f"🗑️ Removing classes {classes}...")

    if not isinstance(classes, set): classes = set(classes)
    files = glob.glob(os.path.join(dataset_all_dir, 'labels', '*.txt'))
    modified = 0
    removed_images = 0

    for label_files in files:
        with open(label_files, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if int(parts[0]) not in classes:
                new_lines.append(line)

        if len(new_lines) != len(lines):
            if len(new_lines) == 0:
                base = os.path.splitext(os.path.basename(label_files))[0]
                img_removed = False

                for ext in ['.jpg', '.png', '.jpeg']:
                    img_path = os.path.join(dataset_all_dir, 'images', base + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        img_removed = True
                        break
                
                os.remove(label_files)
                if img_removed:
                    removed_images += 1

            else:
                with open(label_files, 'w') as f:
                    f.writelines(new_lines)
                modified += 1   
    
    print(f"✅ Cleaned {modified} label files.")


def flip_only_single_class(
    split_dir: str,
    target_class: int,
    new_class: int,
    amount: int = -1
):
    """
    Flip images that contain ONLY target_class.
    target_class → new_class
    """

    image_dir = os.path.join(split_dir, "images")
    label_dir = os.path.join(split_dir, "labels")

    assert os.path.exists(image_dir)
    assert os.path.exists(label_dir)

    candidates = []

    # --- 1. Collect valid candidates ---
    for lbl_file in os.listdir(label_dir):
        if not lbl_file.endswith(".txt"):
            continue

        lbl_path = os.path.join(label_dir, lbl_file)

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        classes_in_image = {
            int(line.split()[0])
            for line in lines
            if len(line.split()) >= 5
        }

        # must contain ONLY the target class
        if classes_in_image == {target_class}:
            candidates.append(lbl_file)
        else: print(f"{lbl_path} contains other signs, skipping...")

    if amount != -1:
        candidates = candidates[:amount]

    print(f"🔄 Flipping {len(candidates)} images: class {target_class} → {new_class}")

    generated = 0

    # --- 2. Flip & remap ---
    for lbl_file in tqdm(candidates):
        base = os.path.splitext(lbl_file)[0]

        # locate image
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(image_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_flipped = cv2.flip(img, 1)

        new_label_lines = []

        with open(os.path.join(label_dir, lbl_file), "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, x, y, w, h = map(float, parts)
                x = 1.0 - x
                new_label_lines.append(
                    f"{new_class} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                )

        new_base = f"{base}_flip_{target_class}to{new_class}"

        cv2.imwrite(os.path.join(image_dir, new_base + ".jpg"), img_flipped)
        with open(os.path.join(label_dir, new_base + ".txt"), "w") as f:
            f.writelines(new_label_lines)

        generated += 1

    print(f"✅ Created {generated} flipped images.")



def merge_class(workdir: str, classes_to_merge: set[int]|int, destination_class: int):
    """
    Replaces occurrences of classes_to_merge with destination_class.
    Example: merge_class([38, 39], 37) -> Turns IDs 38 and 39 into 37.
    """
    assert os.path.exists(workdir)
    
    if isinstance(classes_to_merge, int): classes_to_merge = {classes_to_merge}

    label_dir = os.path.join(workdir, 'labels')
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    modified = 0

    for file in label_files:
        full_path = os.path.join(label_dir, file)
        with open(full_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        file_modified = False
        for line in lines:
            if len(line.split()) == 5 and int(line.split()[0]) in classes_to_merge:
                c, x, y, w, h = line.split()
                new_lines.append(f'{destination_class} {x} {y} {w} {h}\n')
                file_modified = True
            else: new_lines.append(line)
        
        if not file_modified: continue

        with open(full_path, 'w') as f:
            f.writelines(new_lines)
        modified += 1
    
    print(f'Applied change to {modified} files')
        


def deduplicate_all_vs_all(workdir, backup_dir, similarity_thresh):
    # 1. Setup
    assert os.path.exists(workdir), "Workdir does not exist"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    img_dir = os.path.join(workdir, "images")
    label_dir = os.path.join(workdir, "labels")

    assert os.path.exists(img_dir), "images/ directory not found"
    assert os.path.exists(label_dir), "labels/ directory not found"

    os.makedirs(os.path.join(backup_dir, "images"))
    os.makedirs(os.path.join(backup_dir, "labels"))


    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        print("❌ No images found.")
        return

    # 2. Pre-calculate Hashes (This is the slow part, do it once)
    print(f"1️⃣  Computing hashes for {len(files)} images...")
    image_hashes = []
    
    for f in tqdm(files):
        path = os.path.join(img_dir, f)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                # Store (filename, hash_object)
                h = imagehash.phash(img)
                image_hashes.append((f, h))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # 3. All-vs-All Comparison
    print(f"2️⃣  Comparing all images against each other...")
    to_remove = set()
    
    # We use a nested loop, but skip indices we've already marked for removal
    for i in tqdm(range(len(image_hashes))):
        name_a, hash_a = image_hashes[i]
        
        # If A is already marked for deletion, it can't be a "parent" to check others
        if name_a in to_remove:
            continue
            
        # Check against all subsequent images
        for j in range(i + 1, len(image_hashes)):
            name_b, hash_b = image_hashes[j]
            
            if name_b in to_remove:
                continue

            # Calculate Distance
            if (hash_a - hash_b) <= similarity_thresh:
                # Mark B for removal (Keep A as the "original")
                to_remove.add(name_b)

    # 4. Move Files
    print(f"3️⃣  Moving {len(to_remove)} duplicates to backup...")
    count = 0
    for fname in to_remove:
        src_img = os.path.join(img_dir, fname)
        src_lbl = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
        
        dst_img = os.path.join(backup_dir, "images", fname)
        dst_lbl = os.path.join(backup_dir, "labels", os.path.basename(src_lbl))
        
        try:
            shutil.move(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            count += 1
        except Exception as e:
            print(f"Error moving {fname}: {e}")

    print("\n" + "="*40)
    print(f"✅ DONE!")
    print(f"   - Original Count: {len(files)}")
    print(f"   - Removed: {count}")
    print(f"   - Remaining: {len(files) - count}")
    print("="*40)

def copy_paste_dynamic(workdir: str, target_count: int = 100):
    img_dir = os.path.join(workdir, "images")
    lbl_dir = os.path.join(workdir, "labels")

    assert os.path.exists(img_dir), "Train images not found!"
    assert os.path.exists(lbl_dir), "Train labels not found!"

    # --- Scan Counts ---
    cls_map = defaultdict(list)
    cls_instance_count = defaultdict(int)
    
    files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    print(f"📊 Scanning {len(files)} files to find under-represented classes...")
    
    for f in files:
        with open(f) as txt:
            lines = [l.split() for l in txt if len(l.split()) == 5]
        
        present = set()
        for parts in lines:
            try:
                c = int(parts[0])
                present.add(c)
                cls_instance_count[c] += 1
            except ValueError:
                continue
        for c in present:
            cls_map[c].append(f)

    # --- Identify Classes that need a boost ---
    # We only care about classes that exist (>0) but are below target (<100)
    needs_boost = {c for c, count in cls_instance_count.items() if 0 < count < target_count}
    
    if not needs_boost:
        print("✅ All classes are already above the target! Nothing to do.")
        return

    print(f"🚀 Found {len(needs_boost)} classes below {target_count}. Starting Top-Up...")
    
    total_gen = 0
    
    for cls in sorted(needs_boost):
        current = cls_instance_count[cls]
        print(f"   Class {cls}: {current} → {target_count}")

        sources = cls_map.get(cls, [])
        if not sources:
            continue

        i = 0
        # Loop until we hit target
        while cls_instance_count[cls] < target_count:
            src_lbl = sources[i % len(sources)]
            base = os.path.splitext(os.path.basename(src_lbl))[0]

            # Find image
            src_img = None
            src_ext = ""
            for ext in (".jpg", ".png", ".jpeg"):
                p = os.path.join(img_dir, base + ext)
                if os.path.exists(p):
                    src_img = p
                    src_ext = ext
                    break

            if not src_img:
                i += 1
                continue

            # Unique Name: "topup" prefix to distinguish this run
            new_base = f"topup_{cls}_{total_gen}_{base}"
            
            shutil.copy(src_img, os.path.join(img_dir, new_base + src_ext))
            shutil.copy(src_lbl, os.path.join(lbl_dir, new_base + ".txt"))

            # Update count
            added_count = 0
            with open(src_lbl) as f:
                for l in f:
                    if int(l.split()[0]) == cls:
                        added_count += 1
            
            cls_instance_count[cls] += added_count
            total_gen += 1
            i += 1

    print(f"✅ Top-Up Complete. Added {total_gen} extra images.")

def split_all_to_train_val(workdir: str, train_ratio: float = 0.9, seed: int = 42):
    """
    Splits workdir/all -> workdir/train + workdir/val
    """
    assert 0.0 < train_ratio < 1.0
    assert os.path.exists(workdir)

    src_root = os.path.join(workdir, "all")
    img_src = os.path.join(src_root, "images")
    lbl_src = os.path.join(src_root, "labels")

    assert os.path.exists(img_src) and os.path.exists(lbl_src)

    # Create destination folders
    train_img = os.path.join(workdir, "train/images")
    train_lbl = os.path.join(workdir, "train/labels")
    val_img = os.path.join(workdir, "val/images")
    val_lbl = os.path.join(workdir, "val/labels")

    for p in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(p, exist_ok=True)

    # Collect images
    images = [
        f for f in os.listdir(img_src)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not images:
        raise RuntimeError("❌ No images found in all/images")

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_set = images[:split_idx]
    val_set = images[split_idx:]

    # Move files
    def move_pair(img_name, img_dst, lbl_dst):
        base = os.path.splitext(img_name)[0]

        shutil.move(
            os.path.join(img_src, img_name),
            os.path.join(img_dst, img_name)
        )

        lbl_path = os.path.join(lbl_src, base + ".txt")
        if os.path.exists(lbl_path):
            shutil.move(
                lbl_path,
                os.path.join(lbl_dst, base + ".txt")
            )

    print(f"🚚 Splitting dataset ({len(images)} images)")
    print(f"   ➜ Train: {len(train_set)}")
    print(f"   ➜ Val  : {len(val_set)}")

    for img in tqdm(train_set, desc="Train split"):
        move_pair(img, train_img, train_lbl)

    for img in tqdm(val_set, desc="Val split"):
        move_pair(img, val_img, val_lbl)

    # Cleanup
    shutil.rmtree(src_root)

    print("✅ Dataset split complete")


def load_class_names(data_yaml: str):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    return names


def count_classes(label_dir: str):
    counter = Counter()
    total_labels = 0
    total_label_files = 0

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        total_label_files += 1
        with open(os.path.join(label_dir, file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                counter[cls] += 1
                total_labels += 1

    return counter, total_label_files, total_labels


def main(data_yaml = "VNTS/data.yaml", label_dir = "VNTS/all/labels", show = False):
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    class_names = load_class_names(data_yaml)
    counts, num_label_files, num_labels = count_classes(label_dir)
    num_classes = len(class_names)

    print("\n📊 DATASET CLASS BALANCE (ALL)")
    print("=" * 50)
    print(f"Label files  : {num_label_files}")
    print(f"Total boxes  : {num_labels}")
    print(f"Classes      : {num_classes}")
    print("-" * 50)

    for i in range(num_classes):
        name = class_names[i] if i < len(class_names) else "UNKNOWN"
        c = counts.get(i, 0)
        pct = (c / num_labels * 100) if num_labels else 0
        print(f"[{i:02d}] {name:<25} : {c:5d} ({pct:5.2f}%)")

    missing = [i for i in range(num_classes) if counts.get(i, 0) == 0]
    if missing:
        print("\n⚠️ Classes with ZERO instances:")
        for i in missing:
            print(f" - [{i}] {class_names[i]}")

    if show:
        try:
            ids = list(range(num_classes))
            values = [counts.get(i, 0) for i in ids]

            plt.figure(figsize=(14, 5))
            plt.bar(ids, values)
            plt.xticks(ids, class_names, rotation=90)
            plt.ylabel("Instances")
            plt.title("Class Distribution (ALL)")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("(matplotlib not installed, skipping plot)")

if __name__ == "__main__":
    DATASET_PATH = 'datasets/VNTS'

    main(os.path.join(DATASET_PATH, 'data.yaml'), os.path.join(DATASET_PATH, 'train', 'labels'))

    # with DatasetTransaction(DATASET_PATH) as workdir:
    #     all_dir = os.path.join(workdir, "all")

        # merge_all_splits(workdir) # Done

        # 2. Remove Noise / Irrelevant Classes
        # [25] P.245a (Noise), [34] Hospital (Irrelevant)
        # remove_class(all_dir, {25, 34})  # Done

        # --- Symmetry Pairs (Flip Left <-> Right) ---
        # Dangerous Curve: 37 <-> 38
        # flip_only_single_class(all_dir, 37, 38)
        # flip_only_single_class(all_dir, 38, 37)
        
        # # Zigzag Road: 39 <-> 40
        # flip_only_single_class(all_dir, 39, 40)
        # flip_only_single_class(all_dir, 40, 39)

        # # Narrow Road: 41 <-> 42
        # flip_only_single_class(all_dir, 41, 42)
        # flip_only_single_class(all_dir, 42, 41)
        
        # # Intersections (Non-Prio): 47 <-> 48         # ALL FLIPING DONE
        # flip_only_single_class(all_dir, 47, 48)
        # flip_only_single_class(all_dir, 48, 47)
        
        # # No Turn Car: 5 <-> 6
        # flip_only_single_class(all_dir, 5, 6)
        # flip_only_single_class(all_dir, 6, 5)

        # # --- One-Way Boosts (Rare -> Common) ---
        # # Mandatory Turn: 29 (Right Only) -> 30 (Left Only)
        # flip_only_single_class(all_dir, 29, 30)

        # --- New Class Creation ---
        # # P.124c (18: No Left/U) -> P.124d (56: No Right/U)
        # flip_only_single_class(all_dir, target_class=18, new_class=56) # DONE CREATE NEW CLASS
        
        # # 4. MERGE CLASSES (Simplify Dataset)
        
        # # R.301: Merge 'Left Turn' variant (28) into Main 'Left Turn' (26)
        # merge_class(all_dir, classes_to_merge=28, destination_class=26)
        
        # # Curves: Merge Right (38) -> Left (37)
        # merge_class(all_dir, classes_to_merge=38, destination_class=37)
        
        # # Zigzag: Merge Right (40) -> Left (39)
        # merge_class(all_dir, classes_to_merge=40, destination_class=39)
        
        # # Narrow: Merge Right (42) -> Left (41)
        # merge_class(all_dir, classes_to_merge=42, destination_class=41)  # ALL MERGING DONE
        
        # # T-Intersection: Merge 45 -> 44
        # merge_class(all_dir, classes_to_merge=45, destination_class=44)
        
        # # Non-Prio Intersection: Merge 47 & 48 -> 46
        # merge_class(all_dir, classes_to_merge={47, 48}, destination_class=46)

        
        # # 5. DEDUPLICATE (Remove exact duplicates created by flipping)
        # dedup_backup = os.path.join(workdir, "duplicates_removed_2")
        # deduplicate_all_vs_all(all_dir, dedup_backup, similarity_thresh=5) # DONE REMOVE DUPS
        
        # 6. SMART SPLIT (Preserves Video Sequences)
        # split_all_to_train_val(workdir, train_ratio=0.85)
        
        # # # 7. OVERSAMPLE (Train Only)
        # train_dir = os.path.join(workdir, "train")
        
        # # # List of classes to boost (Rare + New Class 56)
        # rare_classes = {
        #     0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 14, 16, 17, 18, 19, 
        #     21, 24, 26, 27, 30, 31, 33, 35, 36, 49, 50, 51, 54, 56
        # }
        
        # copy_paste_dynamic(train_dir, target_count=100)
        
        # 8. FINAL STATS
        # print("\n🏁 FINAL DATASET STATISTICS:")
        # # Try to use data.yaml from the source if available
        # yaml_path = os.path.join(DATASET_PATH, "data.yaml")
        # if os.path.exists(yaml_path):
        #     main(os.path.join(train_dir, "labels"), yaml_path)
        # else:
        #     print("⚠️ data.yaml not found, skipping stats print.")


    
        