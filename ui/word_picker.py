# ui/word_picker.py — Word Grid Picker
import tkinter as tk
from tkinter import messagebox
import config


def show_word_picker(db_client, language: str) -> str:
    """Show grid of signs. Returns selected label or None."""
    from core.db import get_all_signs
    
    chosen = {"label": None}
    
    root = tk.Tk()
    root.title(f"{language} — Select Word/Letter to Learn")
    root.configure(bg="#1e1e2e")
    
    try:
        signs = get_all_signs(db_client, language)
        labels = sorted([s["label"] for s in signs])
    except:
        labels = []
    
    if not labels:
        messagebox.showerror("No Data", f"No signs found for {language}.")
        root.destroy()
        return None
    
    # Title
    tk.Label(root, text=f"{language} — Click a sign to view",
             font=("Segoe UI", 14, "bold"), fg="white", bg="#1e1e2e").pack(pady=12)
    
    # Scrollable grid
    canvas = tk.Canvas(root, bg="#1e1e2e", highlightthickness=0, width=700, height=500)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#1e1e2e")
    
    scrollable_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    cols = 8
    lang_color = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL
    color_hex = f"#{lang_color[2]:02x}{lang_color[1]:02x}{lang_color[0]:02x}"
    
    def pick(label):
        chosen["label"] = label
        root.destroy()
    
    for i, label in enumerate(labels):
        btn = tk.Button(scrollable_frame, text=label, font=("Segoe UI", 13, "bold"),
                       bg=color_hex, fg="white", relief="flat", cursor="hand2",
                       width=6, height=2, command=lambda l=label: pick(l))
        btn.grid(row=i // cols, column=i % cols, padx=4, pady=4)
    
    canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scrollbar.pack(side="right", fill="y")
    
    root.update_idletasks()
    w, h = 720, 540
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")
    
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    
    return chosen["label"]