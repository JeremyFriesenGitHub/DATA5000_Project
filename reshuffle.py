# reshuffling for eval_model test data

import shutil
import random
from pathlib import Path
import tempfile

def reshuffle_splits(dataset_dir, train_ratio=0.7, val_ratio=0.15):
    dataset_dir = Path(dataset_dir)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        all_pairs = []
        
        for split in ["train", "val", "test"]:
            img_dir = dataset_dir / split / "images"
            mask_dir = dataset_dir / split / "masks"
            if img_dir.exists():
                for img in sorted(img_dir.glob("*.jpg")):
                    name = img.stem
                    mask = mask_dir / f"{name}_m.png"
                    if mask.exists():
                        # Copy to temp
                        tmp_img = tmp / img.name
                        tmp_mask = tmp / mask.name
                        shutil.copy(img, tmp_img)
                        shutil.copy(mask, tmp_mask)
                        all_pairs.append((tmp_img, tmp_mask))
        
        print(f"Total samples: {len(all_pairs)}")
        
        random.seed(42)
        random.shuffle(all_pairs)
        
        n = len(all_pairs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            "train": all_pairs[:n_train],
            "val": all_pairs[n_train:n_train + n_val],
            "test": all_pairs[n_train + n_val:],
        }
        
        # Clear and rebuild
        for split, pairs in splits.items():
            img_dir = dataset_dir / split / "images"
            mask_dir = dataset_dir / split / "masks"
            
            if img_dir.exists():
                shutil.rmtree(img_dir)
            if mask_dir.exists():
                shutil.rmtree(mask_dir)
            
            img_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            
            for img, mask in pairs:
                shutil.copy(img, img_dir / img.name)
                shutil.copy(mask, mask_dir / mask.name)
            
            print(f"  {split}: {len(pairs)} samples")

reshuffle_splits("dataset")