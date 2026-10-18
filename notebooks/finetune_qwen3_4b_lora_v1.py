"""Colab-only Qwen3 QLoRA experiment. No model or training dependency loads on import.

Upload notebook-upload.zip prepared locally. Install the pinned cloud stack in
the notebook, select a T4, then run main with explicit rights/upload flags.
Completing training never approves an adapter for the stock application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

BASE_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
BASE_REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
LICENSE_SOURCE = 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/' + BASE_REVISION + '/LICENSE'
PINS = {'transformers': '4.55.4', 'peft': '0.17.1', 'accelerate': '1.10.1', 'bitsandbytes': '0.48.2', 'jsonschema': '4.25.1'}


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_extract(archive_path, destination):
    """Never extract arbitrary paths, executable files or a zip bomb from an upload."""
    destination = Path(destination).resolve()
    allowed = {'manifest.json', 'output-schema.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl'}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if sum(item.file_size for item in members) > 100_000_000:
            raise ValueError('Dataset package exceeds 100MB pilot bound')
        names = [item.filename for item in members]
        if set(names) != allowed or len(names) != len(set(names)):
            raise ValueError('Package must contain only the five expected root-level dataset files')
        payloads = {}
        for item in members:
            if item.filename not in allowed or item.is_dir():
                raise ValueError('Unsafe archive member')
            payloads[item.filename] = archive.read(item)
        # Validate the complete archive and all existing targets before any write.
        for name, payload in payloads.items():
            path = destination / name
            if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != payload)):
                raise ValueError('Existing extracted dataset differs; use a new output directory')
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            if not (destination / name).exists():
                (destination / name).write_bytes(payload)
    return destination


def native_quantization_smoke(torch, bnb):
    """Exercise actual NF4 and 8-bit optimizer kernels, not only import/version.

    Tiny GPU tensors only; no pretrained weights, source data or network I/O.
    This catches a CPU-only stub/missing CUDA shared library before model load.
    """
    tensor = torch.randn(64, 64, device='cuda', dtype=torch.float16)
    quantized, state = bnb.functional.quantize_4bit(tensor, quant_type='nf4')
    restored = bnb.functional.dequantize_4bit(quantized, quant_state=state)
    if restored.shape != tensor.shape or not torch.isfinite(restored).all().item():
        raise RuntimeError('NF4 native kernel produced invalid values')
    parameter = torch.nn.Parameter(torch.randn(8192, device='cuda', dtype=torch.float32))
    parameter.grad = torch.ones_like(parameter)
    optimizer = bnb.optim.PagedAdamW8bit([parameter], lr=1e-4)
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(parameter).all().item():
        raise RuntimeError('8-bit optimizer native kernel produced invalid values')
    return {'nf4_quantize_dequantize': True, 'paged_adamw_8bit_step': True}


def cloud_runtime_smoke():
    """Colab-only CUDA/native readiness; no model weights or web request."""
    import importlib.metadata
    import sys
    import torch
    if os.name == 'nt' or not Path('/content').exists():
        raise RuntimeError('Run cloud readiness in Google Colab, not the app environment')
    for library, expected in PINS.items():
        if importlib.metadata.version(library) != expected:
            raise RuntimeError(f'{library} must be {expected}; restart runtime after install')
    if not torch.cuda.is_available():
        raise RuntimeError('Select Runtime > Change runtime type > T4 GPU')
    capability = torch.cuda.get_device_capability(0)
    if capability < (6, 0):
        raise RuntimeError('NF4/FP16 requires a supported NVIDIA CUDA GPU')
    if os.getenv('BNB_CUDA_VERSION', '').strip():
        raise RuntimeError('Remove the manual BNB_CUDA_VERSION override and restart session; this notebook verifies the PyTorch-linked CUDA runtime.')
    if not torch.version.cuda:
        raise RuntimeError('PyTorch is not a CUDA build; choose a fresh GPU Colab runtime')
    # The PyTorch-linked runtime matters, NOT nvidia-smi's maximum driver CUDA.
    import bitsandbytes as bnb
    cuda_parts = torch.version.cuda.split('.')
    cuda_tag = ''.join(cuda_parts[:2])
    native_path = Path(bnb.__file__).resolve().parent / f'libbitsandbytes_cuda{cuda_tag}.so'
    if not native_path.is_file():
        raise RuntimeError(f'bitsandbytes {PINS["bitsandbytes"]} lacks {native_path.name} for torch CUDA {torch.version.cuda}. Install the updated notebook dependencies, restart session and rerun definitions. Do not rename native libraries or set a blind CUDA override.')
    try:
        native_checks = native_quantization_smoke(torch, bnb)
    except Exception as exc:
        raise RuntimeError(f'Native NF4/8-bit preflight failed with torch={torch.__version__}, CUDA={torch.version.cuda}, bitsandbytes={PINS["bitsandbytes"]}. Restart after installation; run python -m bitsandbytes for diagnostics. Details: {type(exc).__name__}: {exc}') from exc
    return {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'bitsandbytes': importlib.metadata.version('bitsandbytes'), 'native_checks': native_checks,
        'gpu': torch.cuda.get_device_name(0), 'compute_capability': list(capability),
        'gpu_memory_gib': torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        'status': 'CUDA_NF4_OPTIMIZER_READY_NOT_MODEL_TRAINING_VERIFIED'}


def validate_dataset(folder, *, allow_synthetic_demo=False):
    """Recheck rights, review, schema, split locks and connected-group leakage."""
    from jsonschema import Draft202012Validator
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    schema = json.loads((folder / 'output-schema.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 'llm-training-dataset-v1' or not manifest.get('cloud_upload_authorized'):
        raise ValueError('Dataset has not passed explicit cloud-upload opt-in packaging')
    if manifest['status'] == 'SYNTHETIC_DEMO_ONLY' and not allow_synthetic_demo:
        raise ValueError('Synthetic drafts require ALLOW_SYNTHETIC_DEMO=True; they cannot promote an adapter')
    if manifest['status'] not in {'SYNTHETIC_DEMO_ONLY', 'REVIEWED_TRAINING_SET'}:
        raise ValueError('Unknown dataset status')
    if digest(schema) != manifest['schema_hash']:
        raise ValueError('Output schema hash mismatch')
    from jsonschema import FormatChecker
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.check_schema(schema)
    splits, seen_ids, seen_groups, prompts = {}, set(), {}, set()
    privacy = re.compile(r'(?i)(?:bearer\s+[\w.\-]{8,}|(?:api[_-]?key|password|access[_-]?token|secret)\s*[=:]\s*[^\s,}]{4,}|[A-Z]:[\\/]|[\w.+-]+@[\w.-]+\.[a-z]{2,}|[?&](?:token|key|signature)=[^&#\s]+)')
    for split in ['train', 'validation', 'test']:
        data = (folder / f'{split}.jsonl').read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['split_hashes'][split]:
            raise ValueError('Frozen split checksum mismatch: ' + split)
        rows = read_jsonl(folder / f'{split}.jsonl')
        if not rows or len(rows) != manifest['split_counts'][split]:
            raise ValueError('Empty/count-mismatched split')
        for row in rows:
            if row['example_id'] in seen_ids:
                raise ValueError('Duplicate ID across splits')
            seen_ids.add(row['example_id'])
            if (row.get('usage_rights') not in {'AUTHORED', 'PERMISSION_GRANTED', 'PUBLIC_DOMAIN'}
                or not row.get('consent_to_training') or not row.get('cloud_upload_allowed')
                or row.get('contains_personal_portfolio') or privacy.search(stable_json(row))):
                raise ValueError('Rights/privacy check failed; re-review locally before uploading')
            reviewed = row.get('approval_status') == 'APPROVED' and row.get('label_origin') == 'human_reviewed' and row.get('reviewer') and row.get('reviewed_at')
            if not reviewed and not (allow_synthetic_demo and row.get('approval_status') == 'DRAFT' and row.get('label_origin') == 'synthetic_draft' and manifest['status'] == 'SYNTHETIC_DEMO_ONLY'):
                raise ValueError('Missing human-reviewed label or explicitly allowed demo draft')
            if row.get('task') != manifest['task'] or digest(row['output_schema']) != manifest['schema_hash']:
                raise ValueError('Mixed task or schema')
            validator.validate(row['approved_output'])
            if not row['messages'] or row['messages'][-1]['role'] != 'user':
                raise ValueError('Messages must end at user, with output held separately')
            for message in row['messages']:
                if message['role'] == 'system':
                    prompts.add(message['content'])
            keys = [(name, row.get(name)) for name in ['template_group', 'document_family', 'conversation_group'] if row.get(name)]
            keys += [('source_checksum', value) for value in row.get('source_checksums', [])]
            normalized_input = [{'role': message['role'], 'content': re.sub(r'\s+', ' ', message['content']).strip().casefold()} for message in row['messages']]
            keys.append(('normalized_input', digest({'task': row['task'], 'messages': normalized_input})))
            for key in keys:
                if key in seen_groups and seen_groups[key] != split:
                    raise ValueError('Connected group/input leakage across splits')
                seen_groups[key] = split
        splits[split] = rows
    if digest(sorted(prompts)) != manifest['prompt_hash'] or digest(manifest['split_hashes']) != manifest['dataset_digest']:
        raise ValueError('Prompt/dataset lock mismatch')
    return manifest, schema, splits


def make_cloud_demo_package(destination):
    """Optional independent authored pilot. No private chat or market reports are read."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    schema = {'type': 'object', 'properties': {'intent': {'type': 'string', 'enum': ['STOCK_ANALYSIS', 'CHART', 'STOCK_RECOMMENDATION']}, 'tickers': {'type': 'array', 'items': {'type': 'string'}}, 'count': {'type': ['integer', 'null']}}, 'required': ['intent', 'tickers', 'count'], 'additionalProperties': False}
    system = 'Phân loại yêu cầu nghiên cứu chứng khoán. Chỉ trả JSON. Không tự đặt lệnh. Schema: ' + stable_json(schema)
    templates = [('Phân tích {t}', 'STOCK_ANALYSIS'), ('Xem triển vọng {t}', 'STOCK_ANALYSIS'), ('Đánh giá cổ phiếu {t}', 'STOCK_ANALYSIS'), ('Cho nhận xét về {t}', 'STOCK_ANALYSIS'), ('Nghiên cứu {t} giúp tôi', 'STOCK_ANALYSIS'), ('Tìm hiểu mã {t}', 'STOCK_ANALYSIS'), ('Vẽ RSI {t}', 'CHART'), ('Cho biểu đồ giá {t}', 'CHART'), ('Xem chart MACD {t}', 'CHART'), ('Đồ thị doanh thu {t}', 'CHART'), ('Vẽ lợi nhuận {t}', 'CHART'), ('Hiện biểu đồ {t}', 'CHART'), ('Tìm {n} mã đáng chú ý', 'STOCK_RECOMMENDATION'), ('Lọc đúng {n} cổ phiếu', 'STOCK_RECOMMENDATION'), ('Gợi ý {n} mã để nghiên cứu', 'STOCK_RECOMMENDATION'), ('Cho danh sách {n} cổ phiếu', 'STOCK_RECOMMENDATION'), ('Chọn {n} mã tham khảo', 'STOCK_RECOMMENDATION'), ('Tôi cần {n} mã để theo dõi', 'STOCK_RECOMMENDATION')]
    tickers = ['FPT', 'VNM', 'HPG', 'MWG', 'VCB', 'SSI', 'TCB', 'MBB', 'ACB', 'VPB', 'CTG', 'VHM', 'VIC', 'GAS', 'MSN', 'GMD']
    group_order = list(range(len(templates)))
    random.Random(42).shuffle(group_order)
    group_splits = {index: 'train' if i < 12 else 'validation' if i < 15 else 'test' for i, index in enumerate(group_order)}
    splits = {name: [] for name in ['train', 'validation', 'test']}
    for index, (template, intent) in enumerate(templates):
        for i, ticker in enumerate(tickers):
            n = i + 1
            splits[group_splits[index]].append({'example_id': f'cloud-draft-{index:02d}-{i:02d}', 'task': 'nlu_routing', 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': template.format(t=ticker, n=n)}], 'approved_output': {'intent': intent, 'tickers': [ticker] if '{t}' in template else [], 'count': n if '{n}' in template else None}, 'output_schema': schema, 'schema_version': 'cloud-pilot-only-v1', 'prompt_version': 'cloud-pilot-only-v1', 'template_group': f'cloud-authored-template-{index:02d}', 'issuer_group': ticker if '{t}' in template else '', 'document_family': '', 'conversation_group': '', 'source_checksums': [], 'usage_rights': 'AUTHORED', 'consent_to_training': True, 'cloud_upload_allowed': True, 'contains_personal_portfolio': False, 'label_origin': 'synthetic_draft', 'approval_status': 'DRAFT', 'reviewer': '', 'reviewed_at': None})
    hashes = {}
    for name, rows in splits.items():
        path = destination / f'{name}.jsonl'
        path.write_text(''.join(stable_json(row) + '\n' for row in rows), encoding='utf-8', newline='\n')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'schema_version': 'llm-training-dataset-v1', 'status': 'SYNTHETIC_DEMO_ONLY', 'task': 'nlu_routing', 'split_hashes': hashes, 'split_counts': {name: len(rows) for name, rows in splits.items()}, 'schema_hash': digest(schema), 'prompt_hash': digest([system]), 'dataset_digest': digest(hashes), 'cloud_upload_authorized': True, 'promotion_eligible': False}
    write_json(destination / 'manifest.json', manifest)
    write_json(destination / 'output-schema.json', schema)
    return destination


def run_experiment(args):
    if os.name == 'nt' or not Path('/content').exists():
        raise RuntimeError('Run this training script in a Google Colab GPU notebook, not in the app venv')
    if not args.upload_confirmed or not args.license_reviewed:
        raise ValueError('Explicit --upload-confirmed and --license-reviewed are required')
    os.environ.setdefault('HF_HOME', '/content/.cache/huggingface')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import importlib.metadata
    import torch
    from jsonschema import Draft202012Validator
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    runtime_info = cloud_runtime_smoke()
    print('Runtime readiness:', runtime_info)
    set_seed(42)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'experiment-summary.json').exists():
        raise ValueError('Completed experiment exists; use another output directory instead of retuning its holdout')
    dataset_dir = Path(args.dataset)
    if dataset_dir.suffix == '.zip':
        dataset_dir = safe_extract(dataset_dir, output / 'dataset')
    manifest, schema, splits = validate_dataset(dataset_dir, allow_synthetic_demo=args.allow_synthetic_demo)
    # Preserve the exact frozen dataset for reproducible offline A/B after the
    # Colab session ends. This is a private result archive, not publication.
    saved_dataset = output / 'dataset'
    if saved_dataset.resolve() != dataset_dir.resolve():
        saved_dataset.mkdir(exist_ok=True)
        for name in ['manifest.json', 'output-schema.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl']:
            payload = (dataset_dir / name).read_bytes()
            target = saved_dataset / name
            if target.is_symlink() or (target.exists() and target.read_bytes() != payload):
                raise ValueError('Saved dataset differs from this run; use a new output directory')
            if not target.exists():
                target.write_bytes(payload)
    experiment_lock = {'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
        'dataset_digest': manifest['dataset_digest'], 'sequence_length': args.max_sequence_length,
        'train_full': args.train_full, 'pilot_steps': args.pilot_steps,
        'seed': 42, 'learning_rate': 1e-4, 'epochs': 2, 'micro_batch': 1,
        'gradient_accumulation': 8, 'lora_r': 8, 'lora_alpha': 16, 'library_pins': PINS,
        'runtime_stack': {name: runtime_info[name] for name in ('python', 'torch', 'cuda', 'bitsandbytes')}}
    lock_path = output / 'experiment-lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding='utf-8')) != experiment_lock:
            raise ValueError('Resume protocol/dataset changed; use a new output folder')
        if not args.resume:
            raise ValueError('Existing unfinished run: explicitly use --resume auto or a checkpoint path')
    else:
        if args.resume:
            raise ValueError('Resume requires the original experiment-lock.json alongside checkpoints')
        write_json(lock_path, experiment_lock)
    validator = Draft202012Validator(schema)
    print('GPU:', torch.cuda.get_device_name(0), '| dataset:', manifest['status'], '| counts:', manifest['split_counts'])
    print('License:', LICENSE_SOURCE, '| base revision:', BASE_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    def load_base():
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, quantization_config=quantization, torch_dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa', trust_remote_code=False)
    model = prepare_model_for_kbit_training(load_base(), use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.print_trainable_parameters()
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()

    def tokenize_row(row):
        prefix = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(row['messages'] + [{'role': 'assistant', 'content': stable_json(row['approved_output'])}], tokenize=True, add_generation_prompt=False)
        if full[:len(prefix)] != prefix:
            raise ValueError('Chat template does not have a stable assistant-answer prefix')
        if len(full) > args.max_sequence_length:
            return None  # counted below; never truncate the supervised answer
        labels = [-100] * len(prefix) + full[len(prefix):]
        if not any(label != -100 for label in labels):
            raise ValueError('No answer tokens remain for answer-only loss')
        return {'input_ids': full, 'attention_mask': [1] * len(full), 'labels': labels}

    encoded, dropped = {}, {}
    for name in ['train', 'validation']:
        encoded[name] = [tokenize_row(row) for row in splits[name]]
        dropped[name] = sum(row is None for row in encoded[name])
        encoded[name] = [row for row in encoded[name] if row is not None]
        if not encoded[name]:
            raise ValueError(f'No {name} rows fit sequence length; review dataset, do not truncate labels')
    for row in splits['test']:
        if len(tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)) > args.max_sequence_length:
            raise ValueError('A test prompt exceeds the registered context bound; register a new protocol before proceeding')

    class AnswerOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]

    def collate(rows):
        length = ((max(len(row['input_ids']) for row in rows) + 7) // 8) * 8
        return {name: torch.tensor([row[name] + [(-100 if name == 'labels' else tokenizer.pad_token_id if name == 'input_ids' else 0)] * (length - len(row[name])) for row in rows], dtype=torch.long) for name in ['input_ids', 'attention_mask', 'labels']}

    decoding = {'do_sample': False, 'max_new_tokens': 256, 'repetition_penalty': 1.0}
    def predict(rows, *, base=False):
        records = []
        model.eval()
        old_cache = model.config.use_cache
        model.config.use_cache = True
        try:
            with model.disable_adapter() if base else nullcontext():
                for row in rows:
                    inputs = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **decoding, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    raw = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True)
                    records.append({'example_id': row['example_id'], 'raw_output': raw, 'used_fallback': False, 'elapsed_seconds': time.perf_counter() - started, 'output_tokens': int(generated.shape[1] - inputs.shape[1])})
        finally:
            model.config.use_cache = old_cache
        return records

    def metrics(rows, predictions):
        valid, exact = 0, 0
        for row, record in zip(rows, predictions):
            try:
                parsed = json.loads(record['raw_output'])
                validator.validate(parsed)
                valid += 1
                exact += parsed == row['approved_output']
            except Exception:
                # jsonschema validation errors and malformed output are raw-model failures.
                pass
        return {'samples': len(rows), 'raw_schema_validity': valid / len(rows), 'exact_task_accuracy': exact / len(rows)}

    baseline_validation = metrics(splits['validation'], predict(splits['validation'], base=True))
    write_json(output / 'baseline-validation.json', baseline_validation)
    if baseline_validation['exact_task_accuracy'] >= .98 and manifest['status'] != 'SYNTHETIC_DEMO_ONLY' and not args.training_justification:
        write_json(output / 'experiment-summary.json', {'status': 'NO_TRAINING_JUSTIFIED', 'baseline_validation': baseline_validation})
        print('NO_TRAINING_JUSTIFIED: baseline is near ceiling. Record a measured error analysis before training.')
        return
    train_args = TrainingArguments(output_dir=str(output / 'checkpoints'), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=8, learning_rate=1e-4,
        num_train_epochs=2, max_steps=-1 if args.train_full else args.pilot_steps,
        fp16=True, bf16=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        warmup_ratio=.05, lr_scheduler_type='cosine', optim='paged_adamw_8bit', max_grad_norm=.3,
        logging_steps=5, eval_strategy='steps', eval_steps=25, save_strategy='steps', save_steps=25,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, report_to='none', remove_unused_columns=False,
        dataloader_num_workers=0, seed=42, data_seed=42)
    trainer = Trainer(model=model, args=train_args, train_dataset=AnswerOnlyDataset(encoded['train']),
        eval_dataset=AnswerOnlyDataset(encoded['validation']), data_collator=collate)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=True if args.resume == 'auto' else args.resume or None)
    elapsed = time.perf_counter() - started
    adapter_dir = output / 'adapter'
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    model.config.use_cache = True
    # Pilot never measures test. Final experiment consumes frozen test once;
    # choosing another configuration after that makes this test development data.
    evaluation_split = 'test' if args.train_full else 'validation'
    evaluation_rows = splits[evaluation_split]
    base_predictions = predict(evaluation_rows, base=True)
    adapted_predictions = predict(evaluation_rows)
    suffix = '' if args.train_full else '-validation'
    for filename, predictions in [('baseline' + suffix + '-predictions.jsonl', base_predictions), ('candidate' + suffix + '-predictions.jsonl', adapted_predictions)]:
        (output / filename).write_text(''.join(stable_json(row) + '\n' for row in predictions), encoding='utf-8', newline='\n')
    adapter_id = 'qwen3-4b-lora-v1-' + manifest['dataset_digest'][:12]
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in adapter_dir.iterdir() if path.is_file() and path.name in {'adapter_config.json', 'adapter_model.safetensors'}}
    identity = {'base_model': BASE_MODEL, 'base_revision': BASE_REVISION, 'tokenizer_revision': BASE_REVISION,
        'quantization': 'nf4-fp16', 'decoding_hash': digest(decoding), 'schema_hash': manifest['schema_hash'],
        'prompt_hash': manifest['prompt_hash'], 'chat_template_hash': template_hash}
    identities = {'baseline': identity, 'candidate': {**identity, 'adapter_id': adapter_id, 'adapter_files_digest': digest(files)}}
    write_json(output / 'identities.json', identities)
    summary = {'status': 'EXPERIMENTAL_UNREVIEWED', 'dataset_status': manifest['status'], 'baseline_validation': baseline_validation,
        'evaluation_split': evaluation_split,
        'baseline_' + evaluation_split: metrics(evaluation_rows, base_predictions),
        'candidate_' + evaluation_split: metrics(evaluation_rows, adapted_predictions),
        'training_seconds': elapsed, 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
        'sequence_drop_counts': dropped, 'training_steps': trainer.state.global_step,
        'full_training': args.train_full, 'training_justification': args.training_justification,
        'runtime': runtime_info,
        'limitations': ['No promotion: run frozen paired/grouped evaluation locally and obtain human review.', 'Pilot synthetic labels are not independent quality evidence.', 'Training completion/loss alone does not establish an improvement.']}
    write_json(output / 'experiment-summary.json', summary)
    adapter_manifest = {'schema_version': 'llm-adapter-v1', 'adapter_id': adapter_id, 'base_model': BASE_MODEL,
        'base_revision': BASE_REVISION, 'tokenizer_revision': BASE_REVISION, 'chat_template_hash': template_hash,
        'prompt_hash': manifest['prompt_hash'], 'output_schema_hash': manifest['schema_hash'], 'dataset_digest': manifest['dataset_digest'],
        'dataset_status': manifest['status'], 'task': manifest['task'], 'adapter_files': files, 'quantization': 'nf4-fp16',
        'library_versions': {name: importlib.metadata.version(name) for name in [*PINS, 'torch']},
        'training_config': {'sequence_length': args.max_sequence_length, 'seed': 42, 'lora_r': 8, 'alpha': 16, 'answer_only_loss': True},
        'license': 'apache-2.0', 'license_reviewed': args.license_reviewed,
        'evaluation_status': 'PENDING', 'evaluation_report': '', 'evaluation_sha256': '',
        'approval_status': 'PENDING', 'reviewer': '', 'reviewed_at': None}
    write_json(adapter_dir / 'adapter-manifest.json', adapter_manifest)
    # Reload real saved weights rather than pretending the in-memory model proves export works.
    del trainer, model
    import gc
    gc.collect(); torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(load_base(), adapter_dir, is_trainable=False)
    model.eval()
    smoke = predict(splits['validation'][:1])
    write_json(output / 'reload-smoke.json', {'status': 'GENERATED_NOT_APPROVED', 'records': smoke})
    archive_path = shutil.make_archive(str(output.parent / (output.name + '-results')), 'zip', output.parent, base_dir=output.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Adapter:', adapter_dir, '| local evaluation inputs:', output, '| download archive:', archive_path)
    print('No upload/publication/serving was performed. Download files using the notebook download cell.')


def run_saved_inference(adapter_directory, messages, *, max_new_tokens=256):
    """Explicit notebook-only reload for a custom prompt; never approves serving."""
    if os.name == 'nt' or not Path('/content').exists():
        raise RuntimeError('Run interactive model inference inside Google Colab')
    cloud_runtime_smoke()
    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU runtime is required')
    if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'user':
        raise ValueError('Provide the same training-task messages ending with user')
    if not 1 <= max_new_tokens <= 512:
        raise ValueError('Output bound is 1..512 tokens')
    folder = Path(adapter_directory).resolve()
    manifest = json.loads((folder / 'adapter-manifest.json').read_text(encoding='utf-8'))
    if (manifest['base_model'] != BASE_MODEL or manifest['base_revision'] != BASE_REVISION
        or manifest.get('tokenizer_revision') != BASE_REVISION or manifest.get('quantization') != 'nf4-fp16'):
        raise ValueError('Saved adapter base/revision mismatch')
    expected_files = manifest.get('adapter_files', {})
    if set(expected_files) != {'adapter_config.json', 'adapter_model.safetensors'}:
        raise ValueError('Reload requires exactly the hashed PEFT config and safe weights')
    for name, expected in expected_files.items():
        unresolved = folder / name
        path = unresolved.resolve()
        if unresolved.is_symlink() or not path.is_relative_to(folder) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Adapter checksum/path mismatch')
    if json.loads((folder / 'adapter_config.json').read_text(encoding='utf-8')).get('base_model_name_or_path') != BASE_MODEL:
        raise ValueError('PEFT config base model mismatch')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != manifest['chat_template_hash']:
        raise ValueError('Tokenizer chat template mismatch')
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    if ids.shape[1] > manifest['training_config']['sequence_length']:
        raise ValueError('Prompt exceeds registered training context; shorten it explicitly')
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    gc.collect(); torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
        quantization_config=quantization, torch_dtype=torch.float16, device_map={'': 0},
        attn_implementation='sdpa', trust_remote_code=False)
    model = PeftModel.from_pretrained(base, folder, is_trainable=False).eval()
    ids = ids.to('cuda')
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        return {'status': 'EXPERIMENTAL_NOT_APPROVED',
            'raw_output': tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)}
    finally:
        del model, base
        gc.collect(); torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', default='/content/qwen3_lora_v1')
    parser.add_argument('--allow-synthetic-demo', action='store_true')
    parser.add_argument('--upload-confirmed', action='store_true')
    parser.add_argument('--license-reviewed', action='store_true')
    parser.add_argument('--train-full', action='store_true', help='Two epochs instead of the initial fixed-step pilot')
    parser.add_argument('--pilot-steps', type=int, default=50)
    parser.add_argument('--max-sequence-length', type=int, choices=[512, 1024], default=512)
    parser.add_argument('--resume', default='', help='auto or an existing checkpoint path')
    parser.add_argument('--training-justification', default='', help='Measured baseline failure/error analysis, not a desire to force a pass')
    args = parser.parse_args(argv)
    if args.pilot_steps < 25 or args.pilot_steps > 100:
        parser.error('Pilot is bounded to 25..100 steps')
    run_experiment(args)


if __name__ == '__main__':
    main()
