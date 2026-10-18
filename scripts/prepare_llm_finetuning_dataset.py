"""Explicit curation/draft packaging; never scrapes private chat history."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from stock_multi_agent.services.finetuning_dataset import make_synthetic_drafts, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Reviewed JSONL. No directory scanning.')
    parser.add_argument('--output', type=Path, default=Path('.runtime/llm_training/v1'))
    parser.add_argument('--generate-drafts', type=int, metavar='COUNT', help='Create authored unreviewed examples; never gold')
    parser.add_argument('--allow-synthetic-demo', action='store_true')
    parser.add_argument('--cloud-upload-opt-in', action='store_true', help='Attest manual rights/privacy preview before creating a notebook-upload zip')
    args = parser.parse_args()
    if bool(args.input) == bool(args.generate_drafts):
        parser.error('Choose --input OR --generate-drafts COUNT')
    if args.generate_drafts:
        examples = make_synthetic_drafts(args.generate_drafts)
        args.output.mkdir(parents=True, exist_ok=True)
        draft_path = args.output / 'curation-drafts.jsonl'
        if draft_path.exists():
            parser.error('Draft file already exists; use a new versioned output folder')
        draft_path.write_text(''.join(item.model_dump_json() + '\n' for item in examples), encoding='utf-8', newline='\n')
        print(f'Unreviewed drafts: {draft_path.resolve()}')
        if not args.allow_synthetic_demo:
            print('DRAFT_ONLY: review labels/rights or explicitly select --allow-synthetic-demo; no train/test package created.')
            return
    else:
        examples = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    manifest = prepare_dataset(examples, args.output, cloud_upload_opt_in=args.cloud_upload_opt_in,
        allow_synthetic_demo=args.allow_synthetic_demo)
    print(manifest.model_dump_json(indent=2))
    if args.cloud_upload_opt_in:
        archive_path = args.output / 'notebook-upload.zip'
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ['manifest.json', 'output-schema.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl']:
                archive.write(args.output / name, name)
        print(f'Explicit upload package (not uploaded automatically): {archive_path.resolve()}')


if __name__ == '__main__':
    main()
