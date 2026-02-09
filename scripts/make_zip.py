import os
import zipfile

root = r"C:\hoang-credit-fairness"
zip_path = os.path.join(root, "hoang-credit-fairness-export.zip")
exclude_dirs = {"venv", ".git", "__pycache__"}

# Remove existing small/empty zip if present
try:
    if os.path.exists(zip_path):
        os.remove(zip_path)
except Exception as e:
    print('WARN: could not remove existing zip:', e)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if rel == '.':
            rel_parts = []
        else:
            rel_parts = rel.split(os.sep)
        if len(rel_parts) > 0 and rel_parts[0] in exclude_dirs:
            continue
        for fname in filenames:
            if fname == os.path.basename(zip_path):
                continue
            full = os.path.join(dirpath, fname)
            try:
                arcname = os.path.relpath(full, root)
                zf.write(full, arcname)
            except Exception as e:
                print('WARN: failed to add', full, '->', e)

print('ZIP_CREATED:' + zip_path)
