"""B.4 script corpus audit for the converged edopro wrapper.

Drives N self-play steps at a couple of seeds, captures stderr (where
ygopro-core emits script `lua_error` calls), and reports:
  - Total "Passed invalid X flag" occurrences (should be 0 after the
    CHAININFO constant.lua fix)
  - Any other ERROR-level log lines
  - Per-script failure counts (grouped by card code from the stack trace)

Action picker uses a simple productive-action heuristic (prefer
Summon/SpSummon/Activate over phase advance) so the rollout actually
reaches spell/trap activations. Random seed controls reproducibility.

Not a regression assertion gate — the goal is coverage + evidence.
The only hard assertion is "zero CHAININFO errors" which was the
reported divergence from B.2.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
YGO_AGENT = REPO / "src" / "ygo-agent"

os.chdir(YGO_AGENT)
for p in (YGO_AGENT, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ygoenv  # noqa: E402
import ygoenv.edopro.registration  # noqa: E402,F401
from ygoenv.edopro import edopro_ygoenv as edopro_mod  # noqa: E402

# init_module must be called once before the first env construction — it
# populates cards_data_ which the env relies on. Mirrors spike_edopro_e2e.py's
# step2. Without this, env construction crashes in the lua script loader.
DB_PATH = str(YGO_AGENT / "assets" / "locale" / "en" / "cards.cdb")
CODE_LIST = str(YGO_AGENT / "scripts" / "code_list.txt")
DECK = str(YGO_AGENT / "assets" / "deck" / "YugiKaiba.ydk")
edopro_mod.init_module(DB_PATH, CODE_LIST, {"YugiKaiba": DECK})
_emb = REPO / "src" / "card_embeddings.bin"
if _emb.is_file() and hasattr(edopro_mod, "init_embeddings_store"):
    edopro_mod.init_embeddings_store(str(_emb))

# Run one seed per process invocation. ygopro-core's init_module shim in
# edopro doesn't cleanly support re-init with a fresh env pool in the same
# process — the second pool.reset() trips a map-lookup OOR or segfaults.
# The test driver invokes this script once per seed.
SEEDS = [int(s) for s in os.environ.get("B4_SEEDS", "42,7").split(",") if s]
STEPS_PER_SEED = int(os.environ.get("B4_STEPS", "500"))


def pick_action(obs, infos, rng):
    n_opts = int(infos["num_options"][0])
    if n_opts <= 0:
        return 0
    a = obs["actions_"][0]
    # Prefer productive actions. 12-col layout: col 3=msg, 4=act.
    # msg=1 IDLECMD; act=4 Summon, 5 MSet, 8 Activate.
    # msg=8 BATTLECMD; act=6 Attack.
    for pref in [(1, 4), (1, 8), (1, 5), (8, 6)]:
        for i in range(n_opts):
            if int(a[i, 3]) == pref[0] and int(a[i, 4]) == pref[1]:
                return i
    return int(rng.integers(n_opts))


CHAININFO_RE = re.compile(r"Passed invalid CHAININFO flag")
LUA_ERROR_RE = re.compile(r"Passed invalid \w+ flag")
STACK_CARD_RE = re.compile(r'"c(\d+)\.lua"')


def run_seed(seed: int) -> dict:
    # redirect stderr at the Python level. The engine emits lua_error
    # via a logHandler that writes to stderr in edopro.h's bot config;
    # capturing Python-level stderr catches those lines.
    buf = io.StringIO()
    env_term = 0
    steps_done = 0
    _debug = os.environ.get("B4_DEBUG") == "1"
    _cm = contextlib.nullcontext() if _debug else contextlib.redirect_stderr(buf)
    _cm2 = contextlib.nullcontext() if _debug else contextlib.redirect_stdout(buf)
    with _cm, _cm2:
        pool = ygoenv.make(
            task_id="EDOPro-v0",
            env_type="gymnasium",
            num_envs=1,
            num_threads=1,
            seed=seed,
            deck1="YugiKaiba",
            deck2="YugiKaiba",
            player=-1,
            play_mode="self",
            verbose=False,
            max_options=16,
            n_history_actions=16,
            record=False,
        )
        pool.num_envs = 1
        rng = np.random.default_rng(seed)
        obs, infos = pool.reset()
        for _ in range(STEPS_PER_SEED):
            idx = pick_action(obs, infos, rng)
            action = np.array([idx], dtype=np.int32)
            obs, _, term, trunc, infos = pool.step(action)
            steps_done += 1
            if bool(np.logical_or(term, trunc)[0]):
                env_term += 1
                obs, infos = pool.reset()

    output = buf.getvalue()
    chaininfo_hits = len(CHAININFO_RE.findall(output))
    any_lua_errors = LUA_ERROR_RE.findall(output)
    per_card = Counter(STACK_CARD_RE.findall(output))
    return {
        "seed": seed,
        "steps": steps_done,
        "terminations": env_term,
        "chaininfo_errors": chaininfo_hits,
        "all_lua_errors": Counter(any_lua_errors),
        "per_card_errors": per_card,
        "raw_tail": output[-800:] if chaininfo_hits or any_lua_errors else "",
    }


def main() -> None:
    results = []
    for seed in SEEDS:
        print(f"[seed {seed}] running {STEPS_PER_SEED} steps...")
        result = run_seed(seed)
        results.append(result)
        print(f"  steps              : {result['steps']}")
        print(f"  env terminations   : {result['terminations']}")
        print(f"  CHAININFO errors   : {result['chaininfo_errors']}")
        lua_err_summary = dict(result["all_lua_errors"])
        print(f"  lua_error summary  : {lua_err_summary or '{}'}")
        if result["per_card_errors"]:
            top = result["per_card_errors"].most_common(5)
            print(f"  top error sources  : {top}")
        if result["raw_tail"]:
            print(f"  raw tail:\n{result['raw_tail']}")

    total_chaininfo = sum(r["chaininfo_errors"] for r in results)
    print()
    print(f"[summary] total CHAININFO errors across all seeds: {total_chaininfo}")
    assert total_chaininfo == 0, \
        f"FAIL: {total_chaininfo} CHAININFO errors remain — constant.lua fix incomplete"
    print("[done] B.4 CHAININFO fix validated. No Pot of Greed / other "
          "CHAININFO-flag regressions under {}-step rollouts at seeds {}."
          .format(STEPS_PER_SEED, SEEDS))


if __name__ == "__main__":
    main()
