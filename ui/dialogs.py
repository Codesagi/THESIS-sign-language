# ============================================================
# ui/dialogs.py — Selection Dialogs
# ============================================================

import tkinter as tk
from tkinter import simpledialog, messagebox
from typing import Optional


def show_language_dialog() -> Optional[str]:
    """Select ASL or FSL. Returns 'ASL', 'FSL', or None."""
    chosen = {"value": None}

    root = tk.Tk()
    root.title("Sign Language Recognizer")
    root.resizable(False, False)
    root.geometry("380x220")
    root.configure(bg="#1e1e2e")

    root.update_idletasks()
    x = (root.winfo_screenwidth()  - 380) // 2
    y = (root.winfo_screenheight() - 220) // 2
    root.geometry(f"+{x}+{y}")

    tk.Label(
        root, text="Select Sign Language",
        font=("Segoe UI", 16, "bold"),
        fg="white", bg="#1e1e2e"
    ).pack(pady=(24, 4))

    tk.Label(
        root, text="Choose the language for this session:",
        font=("Segoe UI", 10),
        fg="#aaa", bg="#1e1e2e"
    ).pack(pady=(0, 20))

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack()

    def pick(lang):
        chosen["value"] = lang
        root.destroy()

    asl_btn = tk.Button(
        btn_frame, text="🤟  ASL",
        font=("Segoe UI", 13, "bold"),
        bg="#f5a623", fg="white", relief="flat",
        activebackground="#e09510", cursor="hand2",
        width=10, command=lambda: pick("ASL")
    )
    asl_btn.grid(row=0, column=0, padx=12)

    fsl_btn = tk.Button(
        btn_frame, text="🖐  FSL",
        font=("Segoe UI", 13, "bold"),
        bg="#4a9eff", fg="white", relief="flat",
        activebackground="#2a7ee0", cursor="hand2",
        width=10, command=lambda: pick("FSL")
    )
    fsl_btn.grid(row=0, column=1, padx=12)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    return chosen["value"]


def show_mode_dialog() -> Optional[str]:
    """Select Sign-to-Word or Word-to-Sign mode. Returns 'sign-to-word', 'word-to-sign', or None."""
    chosen = {"value": None}

    root = tk.Tk()
    root.title("Translation Mode")
    root.resizable(False, False)
    root.geometry("450x250")
    root.configure(bg="#1e1e2e")

    root.update_idletasks()
    x = (root.winfo_screenwidth()  - 450) // 2
    y = (root.winfo_screenheight() - 250) // 2
    root.geometry(f"+{x}+{y}")

    tk.Label(
        root, text="Choose Translation Mode",
        font=("Segoe UI", 16, "bold"),
        fg="white", bg="#1e1e2e"
    ).pack(pady=(24, 4))

    tk.Label(
        root, text="How do you want to use the system?",
        font=("Segoe UI", 10),
        fg="#aaa", bg="#1e1e2e"
    ).pack(pady=(0, 20))

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack()

    def pick(mode):
        chosen["value"] = mode
        root.destroy()

    s2w_btn = tk.Button(
        btn_frame, text="📷 Sign → Word\n(Recognize my signs)",
        font=("Segoe UI", 11, "bold"),
        bg="#00c853", fg="white", relief="flat",
        activebackground="#00a043", cursor="hand2",
        width=18, height=3, command=lambda: pick("sign-to-word")
    )
    s2w_btn.grid(row=0, column=0, padx=12)

    w2s_btn = tk.Button(
        btn_frame, text="📖 Word → Sign\n(Show me how to sign)",
        font=("Segoe UI", 11, "bold"),
        bg="#ff6f00", fg="white", relief="flat",
        activebackground="#e65100", cursor="hand2",
        width=18, height=3, command=lambda: pick("word-to-sign")
    )
    w2s_btn.grid(row=0, column=1, padx=12)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    return chosen["value"]


def ask_recording_label(language: str) -> Optional[str]:
    """Prompt for recording label. Returns label or None."""
    root = tk.Tk()
    root.withdraw()
    label = simpledialog.askstring(
        "Record Sign",
        f"[{language}] Enter label for this recording\n(e.g. A, B, Hello):",
        parent=root
    )
    root.destroy()
    return label.strip().upper() if label else None