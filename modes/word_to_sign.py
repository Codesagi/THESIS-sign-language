# ============================================================
# modes/word_to_sign.py — Learning Mode (Word → Sign)
# ============================================================

import threading
from tkinter import messagebox

from ui.word_picker import show_word_picker
from ui.overlays import set_status
from visualization.hand_3d_combined import show_combined_3d_camera_view


def run_word_to_sign_mode(db_client, language: str):
    """Word picker → mode selection → viewer loop."""
    
    # Load reference samples for visualization
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
            
            # Pick first sample per label
            from collections import defaultdict
            samples_by_label = defaultdict(list)
            for row in all_rows:
                label = row["label"]
                lm_json = row.get("landmarks_json")
                if lm_json:
                    samples_by_label[label].append(lm_json)
            
            reference_samples = {label: samples[0] for label, samples in samples_by_label.items()}
            print(f"[INFO] ✓ Loaded reference samples for {len(reference_samples)} signs: {sorted(reference_samples.keys())}")
        except Exception as e:
            print(f"[ERROR] Failed to load samples: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load samples:\n{e}")
    
    t = threading.Thread(target=load_samples, daemon=True)
    t.start()
    t.join()  # Wait for samples to load
    
    if not reference_samples:
        print(f"[ERROR] No reference samples available. Exiting Word-to-Sign mode.")
        return
    
    # ── Ask user to choose mode ───────────────────────────────
    from tkinter import Tk, Button, Label
    
    mode_choice = [None]  # closure workaround
    
    def _choose_ar():
        mode_choice[0] = "ar"
        root.destroy()
    
    def _choose_3d():
        mode_choice[0] = "3d"
        root.destroy()
    
    root = Tk()
    root.title(f"Word-to-Sign: {language}")
    root.geometry("400x200")
    root.resizable(False, False)
    
    Label(root, text=f"Choose Learning Mode", font=("Arial", 14, "bold")).pack(pady=20)
    
    Button(
        root, text="AR Overlay Mode (2D skeleton + real-time feedback)",
        font=("Arial", 11), width=40, height=2,
        bg="#4CAF50", fg="white", command=_choose_ar
    ).pack(pady=5)
    
    Button(
        root, text="3D Viewer Mode (rotate 3D hand model)",
        font=("Arial", 11), width=40, height=2,
        bg="#2196F3", fg="white", command=_choose_3d
    ).pack(pady=5)
    
    root.mainloop()
    
    if mode_choice[0] is None:
        return  # user closed window
    
    chosen_mode = mode_choice[0]
    print(f"[INFO] Word-to-Sign mode: {language} ({chosen_mode.upper()}). Pick words to learn.")
    
    # Main loop
    while True:
        label = show_word_picker(db_client, language)
        if label is None:
            break  # User closed picker
        
        if label not in reference_samples:
            messagebox.showwarning("No Data", f"No reference sample found for '{label}'")
            continue
        
        # ── Launch chosen viewer ──────────────────────────────
        try:
            if chosen_mode == "ar":
                from modes.word_to_sign_ar import run_ar_word_to_sign
                run_ar_word_to_sign(reference_samples, language, label)
            else:  # 3d
                # Lazy import 3D viewer (defers pyvista/vtk load)
                from visualization.hand_3d_combined import show_combined_3d_camera_view
                show_combined_3d_camera_view(reference_samples, language, label)
        except Exception as e:
            print(f"[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            err_msg = f"Failed to show {chosen_mode.upper()} visualization:\n{e}"
            if chosen_mode == "3d":
                err_msg += "\n\nMake sure PyVista is installed:\npip install pyvista vtk"
            messagebox.showerror("Error", err_msg)