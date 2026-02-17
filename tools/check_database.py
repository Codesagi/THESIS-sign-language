# ============================================================
# check_database.py — See what's in your Supabase database
# ============================================================

from core.db import get_client

client = get_client()

print("\n" + "="*60)
print("DATABASE CONTENTS CHECK")
print("="*60)

# Check signs table
print("\n📋 SIGNS TABLE:")
signs_resp = client.table("signs").select("*").order("language, label").execute()
signs = signs_resp.data or []

if not signs:
    print("  ⚠️  No signs found in database!")
else:
    print(f"  Total signs: {len(signs)}")
    for lang in ['ASL', 'FSL']:
        lang_signs = [s for s in signs if s['language'] == lang]
        print(f"  {lang}: {len(lang_signs)} signs")

# Check landmark_samples table
print("\n📊 LANDMARK SAMPLES:")

# Fetch ALL samples with pagination (Supabase default limit is 1000)
all_samples = []
page_size = 1000
offset = 0

while True:
    samples_resp = (
        client.table("landmark_samples")
        .select("language, label, source")
        .range(offset, offset + page_size - 1)
        .execute()
    )
    rows = samples_resp.data or []
    if not rows:
        break
    all_samples.extend(rows)
    if len(rows) < page_size:
        break
    offset += page_size

samples = all_samples

if not samples:
    print("  ⚠️  No landmark samples found!")
else:
    print(f"  Total samples: {len(samples)}")
    
    # Count by language
    for lang in ['ASL', 'FSL']:
        lang_samples = [s for s in samples if s['language'] == lang]
        print(f"\n  {lang} samples: {len(lang_samples)}")
        
        # Count per label
        from collections import Counter
        label_counts = Counter(s['label'] for s in lang_samples)
        
        if label_counts:
            print(f"  Sample count per {lang} label:")
            for label, count in sorted(label_counts.items()):
                source_breakdown = Counter(s['source'] for s in lang_samples if s['label'] == label)
                sources = ", ".join(f"{src}:{cnt}" for src, cnt in source_breakdown.items())
                print(f"    {label}: {count:3d}  ({sources})")
        else:
            print(f"    No samples yet for {lang}")

print("\n" + "="*60)
print("✓ Check complete")
print("="*60 + "\n")