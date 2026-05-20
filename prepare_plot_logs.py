"""
Copy training scores.jsonl files into the directory layout expected by plot.py.

plot.py expects:
  <outdir>/<task>/<method>/<seed>/scores.jsonl

Task, seed, model size, and RSSM core are read from each run's config.yaml.

Usage:
  python prepare_plot_logs.py
  python prepare_plot_logs.py --srcdir logdir/dreamer --outdir logdir_plot
  python prepare_plot_logs.py --srcdir logdir/dreamer --outdir logdir_plot --dry-run
  python prepare_plot_logs.py --symlink
  python prepare_plot_logs.py --clean   # remove outdir before copying
"""

import argparse
import pathlib
import re
import shutil

import ruamel.yaml as yaml


SIZE_PRESETS = {
    (512, 64, 4): 'size1m',
    (2048, 256, 16): 'size12m',
    (3072, 384, 24): 'size25m',
    (4096, 512, 32): 'size50m',
    (6144, 768, 48): 'size100m',
    (8192, 1024, 64): 'size200m',
    (12288, 1536, 96): 'size400m',
}

CORE_ALIASES = {
    'gru': 'gru',
    'transformer': 'tx',
    'trm': 'trm',
}


def load_config(run_dir: pathlib.Path) -> dict:
  config_path = run_dir / 'config.yaml'
  if not config_path.exists():
    raise FileNotFoundError(f'missing config.yaml in {run_dir}')
  return yaml.YAML(typ='safe').load(config_path.read_text())


def infer_size(rssm: dict) -> str:
  key = (rssm['deter'], rssm['hidden'], rssm['classes'])
  if key in SIZE_PRESETS:
    return SIZE_PRESETS[key]
  return f"deter{rssm['deter']}"


def infer_core(rssm: dict, run_dir: pathlib.Path) -> str:
  core = rssm.get('core')
  if core:
    if core not in CORE_ALIASES:
      raise ValueError(f'unknown core {core!r} in {run_dir}')
    return CORE_ALIASES[core]
  match = re.search(r'_(gru|tx|trm|transformer)(?:_|$)', run_dir.name)
  if match:
    name = match.group(1)
    return CORE_ALIASES.get(name, name)
  return 'default'


def infer_method(config: dict, run_dir: pathlib.Path) -> str:
  rssm = config['agent']['dyn']['rssm']
  return f"{infer_size(rssm)}_{infer_core(rssm, run_dir)}"


def infer_from_folder(run_dir: pathlib.Path) -> tuple[str, str, str]:
  """Fallback when config.yaml is missing: parse names like crafter_size12m_gru."""
  match = re.match(
      r'(?:(.+)_)?(size\d+m|deter\d+)_(gru|tx|trm|transformer)$',
      run_dir.name,
  )
  if not match:
    raise ValueError(
        f'cannot infer task/method/seed from folder name {run_dir.name!r}')
  task_hint, size, core = match.groups()
  task = task_hint or 'unknown'
  method = f"{size}_{CORE_ALIASES.get(core, core)}"
  return task, method, '0'


def layout_for_run(run_dir: pathlib.Path) -> tuple[str, str, str]:
  config = load_config(run_dir)
  task = config['task']
  seed = str(config.get('seed', 0))
  method = infer_method(config, run_dir)
  return task, method, seed


def find_runs(srcdir: pathlib.Path) -> list[pathlib.Path]:
  runs = []
  for scores in sorted(srcdir.rglob('scores.jsonl')):
    run_dir = scores.parent
    if (run_dir / 'config.yaml').exists() or re.search(
        r'size\d+m|deter\d+', run_dir.name):
      runs.append(run_dir)
  return runs


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--srcdir', type=pathlib.Path, default=pathlib.Path('logdir/dreamer'),
      help='Root directory containing training runs')
  parser.add_argument(
      '--outdir', type=pathlib.Path, default=pathlib.Path('logdir_plot'),
      help='Output root for plot.py layout')
  parser.add_argument(
      '--dry-run', action='store_true',
      help='Print planned copies without writing files')
  parser.add_argument(
      '--symlink', action='store_true',
      help='Create symlinks instead of copying scores.jsonl')
  parser.add_argument(
      '--on-collision', choices=('replace', 'skip', 'error'), default='replace',
      help='What to do when the destination scores.jsonl already exists')
  parser.add_argument(
      '--clean', action='store_true',
      help='Delete outdir before writing (avoids stale runs from old layouts)')
  args = parser.parse_args()

  srcdir = args.srcdir.resolve()
  outdir = args.outdir.resolve()
  if not srcdir.exists():
    raise FileNotFoundError(srcdir)

  runs = find_runs(srcdir)
  if not runs:
    raise SystemExit(f'No runs with scores.jsonl found under {srcdir}')

  planned = []
  for run_dir in runs:
    scores_src = run_dir / 'scores.jsonl'
    try:
      task, method, seed = layout_for_run(run_dir)
    except FileNotFoundError:
      task, method, seed = infer_from_folder(run_dir)
    dest = outdir / task / method / seed / 'scores.jsonl'
    planned.append((run_dir, scores_src, dest))

  print(f'Found {len(planned)} run(s) under {srcdir}')
  for run_dir, scores_src, dest in planned:
    print(f'  {run_dir.name}')
    print(f'    -> {dest.relative_to(outdir)}')

  if args.dry_run:
    print('Dry run: no files written.')
    return

  if args.clean and outdir.exists():
    shutil.rmtree(outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  for run_dir, scores_src, dest in planned:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
      if args.on_collision == 'skip':
        print(f'Skip (exists): {dest}')
        continue
      if args.on_collision == 'error':
        raise FileExistsError(dest)
      if dest.is_symlink() or dest.is_file():
        dest.unlink()
    if args.symlink:
      dest.symlink_to(scores_src.resolve())
    else:
      shutil.copy2(scores_src, dest)
  print(f'Done. Plot with: python plot.py --indirs {outdir.name}')


if __name__ == '__main__':
  main()
