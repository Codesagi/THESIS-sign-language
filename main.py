#!/usr/bin/env python3
# ============================================================
# main.py — Sign Language Recognition System (Entry Point)
#
# Navigation flow with back buttons:
# Start → Language → Mode → Activity → (loop back or exit)
# ============================================================

from tkinter import messagebox


def main():
    """Main entry point with navigation loop."""
    # Lazy import for faster startup
    from core.db import get_client
    from ui.dialogs import show_language_dialog, show_mode_dialog
    
    print("[INFO] Starting Sign Language Recognition System...")
    
    # Connect to database once (reuse connection)
    try:
        db_client = get_client()
        print("[INFO] ✓ Connected to Supabase")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Supabase: {e}")
        messagebox.showerror("DB Error", 
            f"Cannot connect to Supabase:\n{e}\n\n"
            f"Check your .env or config.py settings.")
        return
    
    # Navigation loop
    while True:
        # Step 1: Language selection (with exit option)
        language = show_language_dialog()
        if not language:
            print("[INFO] User exited. Goodbye!")
            break
        
        # Language-specific loop
        while True:
            # Step 2: Mode selection (with back button)
            mode = show_mode_dialog()
            
            if not mode:
                # Back to language selection
                print(f"[INFO] Returning to language selection...")
                break
            
            # Step 3: Run selected mode
            if mode == "sign-to-word":
                # Lazy import (only when needed)
                from modes.sign_to_word import run_sign_to_word_mode
                run_sign_to_word_mode(db_client, language)
            else:  # word-to-sign
                # Lazy import (only when needed)
                from modes.word_to_sign import run_word_to_sign_mode
                run_word_to_sign_mode(db_client, language)
            
            # After mode ends, return to mode selection
            print(f"[INFO] Returning to mode selection ({language})...")
    
    print("[INFO] Session ended. Thank you!")


if __name__ == "__main__":
    main()