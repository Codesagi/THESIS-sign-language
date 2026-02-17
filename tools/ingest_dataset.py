# ============================================================
# ingest_dataset.py — JPG Dataset Ingestion Script
# ============================================================

import os
import sys
import argparse
import time
import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

from core.db import get_client, get_or_create_sign, insert_landmark_sample, _flatten_landmarks
import sys; sys.path.insert(0, ".."); import config

# Batch settings to avoid overwhelming Supabase
BATCH_SIZE  = 20
BATCH_PAUSE = 1.0   # seconds

# MediaPipe (static mode for images)
mp_hands = mp.solutions.hands
hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_landmarks(image_path: str):
    """
    Run MediaPipe on one image file.
    Returns (landmarks_list, hand_side, confidence) or (None, None, None) if no hand found.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    # Try original orientation
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands_static.process(rgb)

    # If no hand found, try flipped
    if not result.multi_hand_landmarks:
        flipped = cv2.flip(img, 1)
        rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        result = hands_static.process(rgb)

    if not result.multi_hand_landmarks:
        return None, None, None

    hand_landmarks = result.multi_hand_landmarks[0]
    lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]

    hand_side = "Unknown"
    confidence = 0.0
    if result.multi_handedness:
        hand_side  = result.multi_handedness[0].classification[0].label
        confidence = result.multi_handedness[0].classification[0].score

    return lm_list, hand_side, confidence


def ingest_dataset(dataset_path: str, language: str, dry_run: bool = False):
    """Main ingestion loop."""
    root = Path(dataset_path).resolve()
    
    print(f"\n[DEBUG] Resolved path : {root}")
    
    if not root.exists():
        print(f"[ERROR] Dataset path not found: {root}")
        sys.exit(1)

    # Show what's inside for debugging
    all_children = sorted(root.iterdir())
    print(f"[DEBUG] Contents ({len(all_children)} items):")
    for child in all_children:
        kind = "DIR " if child.is_dir() else "FILE"
        print(f"         {kind}  {child.name}")

    # Collect all label folders
    label_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    # Handle wrapped folders (e.g. ./asl_dataset/asl_dataset/A/)
    if len(label_dirs) == 1:
        candidate = label_dirs[0]
        inner_dirs = [d for d in candidate.iterdir() if d.is_dir()]
        if inner_dirs:
            print(f"\n[INFO] Only one subfolder found: '{candidate.name}'")
            print(f"[INFO] It contains {len(inner_dirs)} subdirs — treating it as the real root.")
            root = candidate
            label_dirs = sorted(inner_dirs)

    if not label_dirs:
        print(f"\n[ERROR] No label subfolders found in {root}")
        print("        Make sure the structure is:  root/A/img.jpg  root/B/img.jpg  etc.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Ingesting {language} dataset from: {root}")
    print(f"  Labels found: {len(label_dirs)}")
    print(f"  Dry run: {dry_run}")
    print(f"{'='*55}\n")

    if not dry_run:
        client = get_client()

    total_images = 0
    total_saved  = 0
    total_skipped = 0

    for label_dir in label_dirs:
        try:
            label = label_dir.name.upper()
            image_files = [
                f for f in label_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTS
            ]

            if not image_files:
                print(f"  [{label}] No images found, skipping.")
                continue

            # Get/create sign in Supabase
            sign_id = None
            if not dry_run:
                try:
                    sign_id = get_or_create_sign(client, language, label)
                except Exception as e:
                    print(f"  [{label}] ERROR creating sign: {e}")
                    continue

            saved   = 0
            skipped = 0

            for img_idx, img_path in enumerate(tqdm(image_files, desc=f"  [{label:>4}]", leave=False)):
                total_images += 1

                lm_list, hand_side, confidence = extract_landmarks(str(img_path))

                if lm_list is None:
                    skipped += 1
                    total_skipped += 1
                    continue

                if not dry_run:
                    retries = 3
                    for attempt in range(retries):
                        try:
                            insert_landmark_sample(
                                client,
                                sign_id=sign_id,
                                language=language,
                                label=label,
                                landmarks=lm_list,
                                source="jpg_dataset",
                                hand_side=hand_side,
                                confidence=confidence,
                                source_file=str(img_path),
                            )
                            break  # success
                        except Exception as e:
                            if attempt < retries - 1:
                                time.sleep(2 ** attempt)
                            else:
                                print(f"  [WARN] Failed after {retries} attempts: {img_path.name} - {e}")

                    # Pause between batches
                    if (img_idx + 1) % BATCH_SIZE == 0:
                        time.sleep(BATCH_PAUSE)

                saved += 1
                total_saved += 1

            status = "[DRY]" if dry_run else "✓"
            print(f"  {status} [{label}]: {saved} saved, {skipped} skipped (no hand detected)")

        except Exception as e:
            print(f"  [ERROR] [{label_dir.name}] Folder processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*55}")
    print(f"  COMPLETE")
    print(f"  Total images processed : {total_images}")
    print(f"  Landmarks saved        : {total_saved}")
    print(f"  Skipped (no hand)      : {total_skipped}")
    print(f"  Detection rate         : {100*total_saved/max(total_images,1):.1f}%")
    print(f"{'='*55}\n")

    if dry_run:
        print("  ↑ This was a DRY RUN — nothing was written to Supabase.\n"
              "  Re-run without --dry-run to commit.\n")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a JPG sign language dataset into Supabase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_dataset.py --path ./asl_dataset --language ASL
  python ingest_dataset.py --path ./fsl_dataset --language FSL
  python ingest_dataset.py --path ./asl_dataset --language ASL --dry-run
        """
    )
    parser.add_argument("--path",     required=True,  help="Root folder of the dataset")
    parser.add_argument("--language", required=True,  choices=["ASL", "FSL"], help="ASL or FSL")
    parser.add_argument("--dry-run",  action="store_true", help="Scan without writing to Supabase")

    args = parser.parse_args()
    ingest_dataset(args.path, args.language, dry_run=args.dry_run)

    hands_static.close()