#!/usr/bin/env python3
"""Safe file sync utility.

Writes a set of source files into a destination tree only if the
contents differ. Creates a timestamped backup of replaced files and
verifies Python syntax before committing the overwrite.

Usage:
    python sync_files_safe.py --src-dir . --dst-dir /content/thinknet-transformer
"""
import argparse
import os
import shutil
import time
import filecmp
import py_compile


DEFAULT_FILES = [
    "data.py",
    "configs/model_config.py",
    "model/block.py",
    "train.py",
    "training/trainer.py",
]


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def backup_path(p: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{p}.bak.{ts}"


def safe_copy(src, dst):
    ensure_dir(dst)

    if not os.path.exists(src):
        print(f"SKIP (missing source): {src}")
        return False

    if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
        print(f"UNCHANGED: {dst}")
        return False

    # backup existing dst
    if os.path.exists(dst):
        b = backup_path(dst)
        shutil.copy2(dst, b)
        print(f"BACKUP: {dst} -> {b}")

    shutil.copy2(src, dst)

    # verify python syntax where applicable
    if dst.endswith('.py'):
        try:
            py_compile.compile(dst, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"SYNTAX ERROR: {dst}: {e}")
            # restore backup if present
            if os.path.exists(b):
                shutil.copy2(b, dst)
                print(f"RESTORED: {dst} from {b}")
            return False

    print(f"UPDATED: {dst}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src-dir', default='.', help='Source root directory')
    p.add_argument('--dst-dir', default='.', help='Destination root directory')
    p.add_argument('--files', nargs='*', help='Specific relative files to sync')
    args = p.parse_args()

    files = args.files if args.files else DEFAULT_FILES

    src_root = os.path.abspath(args.src_dir)
    dst_root = os.path.abspath(args.dst_dir)

    changed = 0
    for rel in files:
        src = os.path.join(src_root, rel)
        dst = os.path.join(dst_root, rel)
        try:
            if safe_copy(src, dst):
                changed += 1
        except Exception as e:
            print(f"ERROR syncing {rel}: {e}")

    print(f"Done. Files updated: {changed}")


if __name__ == '__main__':
    main()
