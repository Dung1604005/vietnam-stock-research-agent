"""Kaggle input packaging tests; no GPU, downloads or investment-quality claims."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/train_lstm_kaggle_v3_ready.ipynb'


def notebook_helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    namespace = {'Path': Path}
    exec(compile(''.join(notebook['cells'][2]['source']), str(NOTEBOOK), 'exec'), namespace)
    return namespace


def dataset_fixture(tmp_path, *, bin_suffix=True):
    source = tmp_path / 'input' / 'dataset'
    source.mkdir(parents=True)
    csv = source / 'vci_ohlcv_lstm_v3.csv'
    csv.write_bytes(b'ticker,date,close\nDEMO,2026-01-02,100\n')
    name = 'ml_v3_runtime.test.zip'
    archive = source / (name + '.bin' if bin_suffix else name)
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('stock_multi_agent/__init__.py', '# Synthetic packaging fixture only\n')
    manifest = {'runtime_bundle': {'filename': name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()},
        'files': {'dataset': {'filename': csv.name, 'sha256': hashlib.sha256(csv.read_bytes()).hexdigest()}},
        'retained_rows': 1, 'tickers': 1, 'training_endpoints': 0}
    (source / 'dataset.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (source / 'protocol.json').write_text(json.dumps({'window': 30, 'horizons': [1, 5]}), encoding='utf-8')
    (source / 'train_lstm_kaggle_v3.py').write_text('# Synthetic script fixture, never executed\n', encoding='utf-8')
    return source, archive, manifest


def test_ready_notebook_all_python_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source)
            compile(source, f'{NOTEBOOK}:cell-{index}', 'exec')
            assert cell['outputs'] == [] and cell['execution_count'] is None


@pytest.mark.parametrize('bin_suffix', [True, False])
def test_stage_verified_zip_or_bin_and_preserve_input(tmp_path, monkeypatch, bin_suffix):
    source, archive, manifest = dataset_fixture(tmp_path, bin_suffix=bin_suffix)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    # Linux Kaggle links the CSV; Windows CI may not grant symlink privileges.
    monkeypatch.setattr(Path, 'symlink_to', lambda target, original: shutil.copy2(original, target))
    stage = tmp_path / 'working' / 'input-run'
    csv, script = notebook_helpers()['prepare_v3_input'](source.parent, stage)
    assert csv.read_bytes() == (source / csv.name).read_bytes()
    assert script.name == 'train_lstm_kaggle_v3.py'
    assert (stage / manifest['runtime_bundle']['filename']).read_bytes() == archive.read_bytes()
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    with pytest.raises(FileExistsError):
        notebook_helpers()['prepare_v3_input'](source.parent, stage)


def test_extracted_directory_is_not_silently_trusted(tmp_path):
    source, archive, _ = dataset_fixture(tmp_path)
    archive.unlink()
    (source / 'ml_v3_runtime.test').mkdir()
    with pytest.raises(FileNotFoundError, match='zip.bin'):
        notebook_helpers()['prepare_v3_input'](source.parent, tmp_path / 'stage')
    assert not (tmp_path / 'stage').exists()


@pytest.mark.parametrize('corrupt', ['archive', 'csv'])
def test_corrupt_inputs_rejected_before_staging(tmp_path, corrupt):
    source, archive, _ = dataset_fixture(tmp_path)
    target = archive if corrupt == 'archive' else source / 'vci_ohlcv_lstm_v3.csv'
    with target.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='checksum|manifest'):
        notebook_helpers()['prepare_v3_input'](source.parent, tmp_path / 'stage')
    assert not (tmp_path / 'stage').exists()


@pytest.mark.parametrize('name', ['../runtime.zip', 'folder\\runtime.zip'])
def test_unsafe_runtime_filename_rejected(tmp_path, name):
    source, _, manifest = dataset_fixture(tmp_path)
    manifest['runtime_bundle']['filename'] = name
    (source / 'dataset.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='an toàn'):
        notebook_helpers()['prepare_v3_input'](source.parent, tmp_path / 'stage')


def test_ambiguous_datasets_require_explicit_csv(tmp_path, monkeypatch):
    source, _, _ = dataset_fixture(tmp_path)
    other = source.parent / 'other'
    shutil.copytree(source, other)
    prepare = notebook_helpers()['prepare_v3_input']
    with pytest.raises(FileNotFoundError, match='2 CSV'):
        prepare(source.parent, tmp_path / 'stage')
    monkeypatch.setattr(Path, 'symlink_to', lambda target, original: shutil.copy2(original, target))
    csv, _ = prepare(source.parent, tmp_path / 'stage', str(source / 'vci_ohlcv_lstm_v3.csv'))
    assert csv.is_file()
