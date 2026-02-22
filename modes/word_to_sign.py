# ============================================================
# modes/word_to_sign.py — Learning Mode (Word → Sign)
# ============================================================

import threading
from tkinter import messagebox

from ui.word_picker import show_word_picker


def run_word_to_sign_mode(db_client, language: str):
    """Word picker → mode selection → viewer loop."""
    
    # Load reference samples
    reference_samples = {}
    
    def load_samples():
        nonlocal reference_samples
        print(f"[INFO] Loading {language} reference samples...")
        try:
            all_rows = []
            page_size = 1000
            offset = 0
            
            while True:
                resp = (
                    db_client.table("landmark_samples")
                    .select("label, landmarks_json")
                    .eq("language", language)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                rows = resp.data or []
                if not rows:
                    break
                all_rows.extend(rows)
                print(f"      Loaded {len(all_rows)} samples so far...")
                if len(rows) < page_size:
                    break
                offset += page_size
            
            if not all_rows:
                print(f"[ERROR] No samples found for {language}")
                messagebox.showerror("No Data", 
                    f"No landmark samples found for {language}.\n\n"
                    f"Please ingest dataset first:\n"
                    f"python -m tools.ingest_dataset --path ./dataset --language {language}")
                return
            
            from collections import defaultdict
            samples_by_label = defaultdict(list)
            for row in all_rows:
                label = row["label"]
                lm_json = row.get("landmarks_json")
                if lm_json:
                    samples_by_label[label].append(lm_json)
            
            reference_samples = {label: samples[0] for label, samples in samples_by_label.items()}
            print(f"[INFO] ✓ Loaded reference samples for {len(reference_samples)} signs")
        except Exception as e:
            print(f"[ERROR] Failed to load samples: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load samples:\n{e}")
    
    t = threading.Thread(target=load_samples, daemon=True)
    t.start()
    t.join()
    
    if not reference_samples:
        print(f"[ERROR] No reference samples available.")
        return
    
    # ── Mode selection ─────────────────────────────────────────
    from tkinter import Tk, Button, Label
    
    mode_choice = [None]
    
    def _choose_2d_ar():
        mode_choice[0] = "2d_ar"
        root.destroy()
    
    def _choose_3d_ar():
        mode_choice[0] = "3d_ar"
        root.destroy()
    
    def _choose_aruco_ar():
        mode_choice[0] = "aruco_ar"
        root.destroy()
    
    def _choose_viewer():
        mode_choice[0] = "viewer"
        root.destroy()
    
    root = Tk()
    root.title(f"Word-to-Sign: {language}")
    root.geometry("600x280")
    root.resizable(False, False)
    
    Label(root, text=f"Choose Learning Mode", font=("Arial", 14, "bold")).pack(pady=15)
    
    Button(
        root, text="2D Hand AR (skeleton floats above hand, fast)",
        font=("Arial", 11), width=55, height=2,
        bg="#4CAF50", fg="white", command=_choose_2d_ar
    ).pack(pady=3)
    
    Button(
        root, text="3D Mesh AR (full 3D mesh floats, phone anchor)",
        font=("Arial", 11), width=55, height=2,
        bg="#9C27B0", fg="white", command=_choose_3d_ar
    ).pack(pady=3)
    
    Button(
        root, text="ARuco AR (marker-based, precise) [EXPERIMENTAL]",
        font=("Arial", 11), width=55, height=2,
        bg="#FF5722", fg="white", command=_choose_aruco_ar
    ).pack(pady=3)
    
    Button(
        root, text="3D Viewer (separate rotatable window)",
        font=("Arial", 11), width=55, height=2,
        bg="#2196F3", fg="white", command=_choose_viewer
    ).pack(pady=3)
    
    root.mainloop()
    
    if mode_choice[0] is None:
        return
    
    chosen_mode = mode_choice[0]
    print(f"[INFO] Word-to-Sign mode: {language} ({chosen_mode.upper()})")
    
    # Main loop
    while True:
        label = show_word_picker(db_client, language)
        if label is None:
            break
        
        if label not in reference_samples:
            messagebox.showwarning("No Data", f"No reference sample found for '{label}'")
            continue
        
        # ── Launch chosen mode ─────────────────────────────────
        try:
            if chosen_mode == "2d_ar":
                print("[INFO] Starting 2D Hand AR")
                from modes.word_to_sign_hand_2d import run_hand_2d_ar
                run_hand_2d_ar(reference_samples, language, label)
            
            elif chosen_mode == "3d_ar":
                print("[INFO] Starting 3D Mesh AR (phone-anchored)")
                from modes.word_to_sign_hand_3d_final import run_hand_3d_ar_final
                run_hand_3d_ar_final(reference_samples, language, label)
            
            elif chosen_mode == "aruco_ar":
                print("[INFO] Starting ARuco Marker AR (experimental)")
                from modes.word_to_sign_aruco import run_aruco_ar
                run_aruco_ar(reference_samples, language, label)
            
            else:  # viewer
                from visualization.hand_3d_combined import show_combined_3d_camera_view
                show_combined_3d_camera_view(reference_samples, language, label)
        
        except Exception as e:
            print(f"[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", 
                f"Failed to show {chosen_mode.upper()}:\n{e}\n\n"
                f"Check dependencies: pip install mediapipe opencv-python pyvista")