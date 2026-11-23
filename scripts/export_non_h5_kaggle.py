from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from IPython.display import FileLink, display

# Run in a Kaggle notebook cell after your experiment.
root = Path("/kaggle/working")
destination = root / "LHC_All_Except_H5.zip"

# Never include raw training/signal/blackbox datasets.
skip_extensions = {".h5", ".hdf5", ".hdf"}
skip_dirs = {"__pycache__", ".ipynb_checkpoints", ".git"}
skip_other = {".zip", ".part"}  # Avoid recursive ZIP inclusion and incomplete downloads.

if not root.is_dir():
    raise FileNotFoundError(f"Missing Kaggle output folder: {root}")

candidates = []
for p in root.rglob("*"):
    if not p.is_file() or p.resolve() == destination.resolve():
        continue
    if skip_dirs.intersection(p.relative_to(root).parts):
        continue
    if p.suffix.lower() in (skip_extensions | skip_other):
        print(f"Skipped: {p.relative_to(root)}")
        continue
    candidates.append(p)

if not candidates:
    raise RuntimeError("No exportable files found in /kaggle/working.")

with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=6) as zf:
    for p in sorted(candidates):
        arcname = p.relative_to(root).as_posix()
        zf.write(p, arcname=arcname)
        print(f"Added: {arcname} ({p.stat().st_size / (1024**2):.2f} MB)")

# Verification: ZIP must not contain any HDF5 dataset.
with ZipFile(destination) as zf:
    names = zf.namelist()
    assert all(Path(n).suffix.lower() not in skip_extensions for n in names)
    assert zf.testzip() is None, "ZIP integrity check failed"

print(f"\nCreated: {destination}")
print(f"Files: {len(names)}; size: {destination.stat().st_size / (1024**2):.2f} MB")
print("NOTE: This ZIP includes .keras weights and .npz scalers if they are in /kaggle/working.")
print("For GitHub, avoid committing old results and excluded model files by default.")
display(FileLink(str(destination), result_html_prefix="Download ZIP: "))
