"""Fetch the two exact released checkpoints; never overwrite an existing file."""
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
if __name__ == '__main__':
    checkpoints = json.loads((HERE / 'formal-freeze.json').read_text())['checkpoints']
    for name, info in checkpoints.items():
        destination = HERE / (name + '.ckpt')
        assert not destination.exists(), destination
        partial = destination.with_suffix('.ckpt.part')
        assert not partial.exists(), partial
        urllib.request.urlretrieve(info['url'], partial)
        digest = hashlib.sha256()
        with partial.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        assert partial.stat().st_size == info['bytes'] and digest.hexdigest() == info['sha256'], name
        partial.rename(destination)
        print(name, 'verified')
