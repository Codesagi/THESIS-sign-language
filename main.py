#!/usr/bin/env python3
# ============================================================
# main.py — Sign Language Recognition System (Entry Point)
# ============================================================

from tkinter import messagebox
from core.db import get_client
from ui.dialogs import show_language_dialog, show_mode_dialog
from modes.sign_to_word import run_sign_to_word_mode
from modes.word_to_sign import run_word_to_sign_mode


def main():
    """Main entry point."""
    # Step 1: Language selection
    language = show_language_dialog()
    if not language:
        print("No language selected. Exiting.")
        return

    # Step 2: Mode selection
    mode = show_mode_dialog()
    if not mode:
        print("No mode selected. Exiting.")
        return

    # Step 3: Connect to database
    try:
        db_client = get_client()
    except Exception as e:
        print(f"[ERROR] Cannot connect to Supabase: {e}")
        messagebox.showerror("DB Error", f"Cannot connect to Supabase:\n{e}")
        return

    # Step 4: Run selected mode
    if mode == "sign-to-word":
        run_sign_to_word_mode(db_client, language)
    else:  # word-to-sign
        run_word_to_sign_mode(db_client, language)

    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()