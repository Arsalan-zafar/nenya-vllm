# SPDX-License-Identifier: Apache-2.0
"""Checks a checkpoint before `vllm serve` starts (serve/serve.sh runs it; NENYA_PREFLIGHT=0 skips it).

    python -m nenya_vllm.preflight MODEL_DIR [--generation-config auto|vllm|DIR]

  * stop ids: vLLM stops on the eos ids of the generation config it reads (plus the tokenizer's eos). They must
    include <|im_end|> (100265), which ends a call turn, and <|endoftext|> (100257), which ends a final answer;
    without 100265 a chat reply runs on past its end. `--generation-config` is vLLM's flag: "auto" reads
    MODEL_DIR/generation_config.json, a directory reads DIR/generation_config.json, "vllm" reads none.
  * tokenizer: ids 100270/100271 are named <raw_code>/</raw_code> (export-hf), or multi-line arguments are not
    written in the trained format. They must not be special tokens either, in tokenizer.json or in
    tokenizer_config.json: vLLM decodes replies with skip_special_tokens=True by default, which would drop the two
    tags, so the parser would never see a <raw_code> block. export-hf does all of this.

Exit 1 with the problems listed, 0 otherwise. A model that is not a local directory (a Hub id) cannot be checked
here: a note, exit 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import olmo3_chat as oc

RAW_TOKENS = ((oc.RAW_BEGIN, oc.S_RAW_BEGIN), (oc.RAW_END, oc.S_RAW_END))
FIX = ("run `python -m nenya_vllm.olmo3_chat export-hf {model}` once, or serve with "
       "GENERATION_CONFIG=serve/generation_config")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generation_config_path(model: str, generation_config: str = "auto") -> str | None:
    if generation_config == "vllm":
        return None
    folder = model if generation_config == "auto" else generation_config
    return os.path.join(folder, "generation_config.json")


def check(model: str, generation_config: str = "auto"):
    """(problems, notes) for serving `model` with vLLM's --generation-config `generation_config`."""
    problems, notes = [], []
    if not os.path.isdir(model):
        notes.append(f"{model} is not a local directory: stop ids and tokenizer not checked (export-hf a local copy)")
        return problems, notes
    path = generation_config_path(model, generation_config)
    fix = FIX.format(model=model)
    if path is None:
        problems.append(f"--generation-config vllm reads no generation config: only the tokenizer's eos stops a "
                        f"reply, so a chat reply runs on past <|im_end|>; {fix}")
    elif not os.path.exists(path):
        problems.append(f"no {path}: vLLM would not stop on <|im_end|>; {fix}")
    else:
        try:
            eos = _load(path).get("eos_token_id")
        except (OSError, ValueError, AttributeError) as e:
            problems.append(f"cannot read {path} ({e}); {fix}")
        else:
            ids = set(eos if isinstance(eos, list) else [eos])
            missing = [f"{name} ({i})" for name, i in (("<|im_end|>", oc.IM_END), ("<|endoftext|>", oc.EOT))
                       if i not in ids]
            if missing:
                problems.append(f"{path}: eos_token_id {eos} lacks {', '.join(missing)}, so replies do not stop "
                                f"there; {fix}")
    problems += _token_problems(model, notes)
    return problems, notes


def _token_problems(model, notes):
    """The tokenizer's entries for ids 100270/100271: named <raw_code>/</raw_code> and not special, in tokenizer.json
    (its added_tokens and, when the BPE vocabulary lists the id, the name there: tokenizers reads an added token's id
    from the vocabulary by name) and in tokenizer_config.json's added_tokens_decoder (transformers reads the added
    tokens from there too, special flag included). One problem per id at most."""
    export = f"run `python -m nenya_vllm.olmo3_chat export-hf {model}` once"
    found = {}                                   # id -> its problem
    tj = os.path.join(model, "tokenizer.json")
    if not os.path.exists(tj):
        notes.append(f"no {tj}: the tokenizer's <raw_code> names were not checked")
    else:
        try:
            data = _load(tj)
            entries = {t.get("id"): t for t in data.get("added_tokens", [])}
            vocab = data.get("model", {}).get("vocab")
        except (OSError, ValueError, AttributeError) as e:
            return [f"cannot read {tj} ({e})"]
        in_vocab = set(vocab.values()) if isinstance(vocab, dict) else set()
        for i, name in RAW_TOKENS:
            content = entries.get(i, {}).get("content")
            if content != name or (i in in_vocab and vocab.get(name) != i):
                found[i] = (f"{tj}: id {i} is {content!r}, not {name!r}, so multi-line arguments are not written in "
                            f"the trained format; {export}")
            elif entries[i].get("special"):
                found[i] = (f"{tj}: {name} (id {i}) is a special token, so vLLM's default skip_special_tokens drops "
                            f"it from replies and the parser never sees a <raw_code> block; {export}")
    tc = os.path.join(model, "tokenizer_config.json")
    if os.path.exists(tc):
        try:
            decoder = _load(tc).get("added_tokens_decoder")
        except (OSError, ValueError, AttributeError) as e:
            return list(found.values()) + [f"cannot read {tc} ({e})"]
        for i, name in RAW_TOKENS:
            entry = decoder.get(str(i)) if isinstance(decoder, dict) else None
            if i in found or not isinstance(entry, dict):
                continue
            if entry.get("content") != name:
                found[i] = f"{tc}: added_tokens_decoder names id {i} {entry.get('content')!r}, not {name!r}; {export}"
            elif entry.get("special"):
                found[i] = (f"{tc}: {name} (id {i}) is a special token in added_tokens_decoder, so vLLM's default "
                            f"skip_special_tokens drops it from replies; {export}")
    return [found[i] for i, _ in RAW_TOKENS if i in found]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--generation-config", default="auto", help="vLLM's --generation-config value (default: auto)")
    a = ap.parse_args(argv)
    problems, notes = check(a.model, a.generation_config)
    for n in notes:
        print(f"preflight: note: {n}", file=sys.stderr)
    for p in problems:
        print(f"preflight: FAIL {p}", file=sys.stderr)
    if not problems:
        print(f"preflight: ok ({a.model}, generation config {a.generation_config})", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
