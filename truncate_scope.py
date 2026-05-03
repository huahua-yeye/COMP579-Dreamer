"""
Truncate all .float files in a scope directory to a given maximum step.
Creates a new directory (non-destructive) or overwrites in-place with --inplace.

Usage:
  python truncate_scope.py <scope_dir> <max_step> [--outdir <out>] [--inplace]

Examples:
  # Create truncated copy of tx scope up to step 44255
  python truncate_scope.py logdir/dreamer/crafter/crafter_size12m_tx/scope 44255

  # Overwrite in-place
  python truncate_scope.py logdir/dreamer/crafter/crafter_size12m_tx/scope 44255 --inplace
"""

import argparse
import pathlib
import shutil
import struct


def truncate_float_file(src: pathlib.Path, dst: pathlib.Path, max_step: int):
    data = src.read_bytes()
    record_size = struct.calcsize('>qd')
    n = len(data) // record_size
    rows = list(struct.iter_unpack('>qd', data))
    kept = [r for r in rows if r[0] <= max_step]
    print(f"  {src.name}: {n} -> {len(kept)} records (step <= {max_step:,})")
    out = bytearray(len(kept) * record_size)
    for i, row in enumerate(kept):
        struct.pack_into('>qd', out, i * record_size, *row)
    dst.write_bytes(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scope_dir', type=pathlib.Path)
    parser.add_argument('max_step', type=int)
    parser.add_argument('--outdir', type=pathlib.Path, default=None)
    parser.add_argument('--inplace', action='store_true')
    args = parser.parse_args()

    src_dir = args.scope_dir.resolve()
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    if args.inplace:
        dst_dir = src_dir
    else:
        if args.outdir:
            dst_dir = args.outdir.resolve()
        else:
            dst_dir = src_dir.parent / (src_dir.name + f'_trun{args.max_step}')
        if dst_dir != src_dir:
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

    print(f"Truncating .float files in: {dst_dir}")
    for f in sorted(dst_dir.glob('*.float')):
        src_f = src_dir / f.name
        truncate_float_file(src_f, f, args.max_step)
    print("Done.")


if __name__ == '__main__':
    main()
