"""SPIKE successor — end-to-end determinism check for edopro_ygoenv.so.

Originally the minimal spike script (spike §4). Extended during Phase B.1
of the edo9300 migration to cover the two determinism tests the plan doc
§2 Phase B.1 requires:
  (a) reset determinism: two runs, same seed → byte-identical reset obs
  (b) step determinism:  two runs, same seed + same action sequence →
                         byte-identical step trajectories (obs + reward)

Pre-B.1 this script terminated during (a) with
  terminate called after throwing 'std::out_of_range'
  ankerl::unordered_dense::map::at(): key not found
caused by the query-buffer parser in get_cards_in_location assuming a
flat-struct layout while current edo9300 emits TLV records. The parser
was replaced with parse_one_card_tlv_() in Phase B.1.

This is still a narrow B.1 test — NOT a training env and NOT a parity
test vs. the Fluorohydride path. Observation values are deterministic
but not semantically audited. B.3-B.6 cover those concerns.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # ExodAI/
YGO_AGENT = REPO / "src" / "ygo-agent"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(YGO_AGENT))

# Env scripts (per read_card_script in edopro.h) are loaded from
# `./edopro_script/` relative to CWD.
os.chdir(str(YGO_AGENT))

import numpy as np  # noqa: E402
import ygoenv  # noqa: E402,F401  — needed for registration side-effects
from ygoenv.edopro import edopro_ygoenv as edopro_mod  # noqa: E402
import ygoenv.edopro.registration  # noqa: E402,F401  — registers EDOPro-v0


DB_PATH = str(YGO_AGENT / "assets" / "locale" / "en" / "cards.cdb")
CODE_LIST = str(YGO_AGENT / "scripts" / "code_list.txt")
DECK = str(YGO_AGENT / "assets" / "deck" / "YugiKaiba.ydk")


def step1_module_import() -> None:
    print(f"[1] module loaded: {edopro_mod}")
    print(f"    symbols: {[s for s in dir(edopro_mod) if not s.startswith('_')]}")


def step2_init_module() -> None:
    print("[2] init_module(...)")
    print(f"    db_path:   {DB_PATH} (exists={Path(DB_PATH).is_file()})")
    print(f"    code_list: {CODE_LIST} (exists={Path(CODE_LIST).is_file()})")
    print(f"    deck:      {DECK} (exists={Path(DECK).is_file()})")
    edopro_mod.init_module(DB_PATH, CODE_LIST, {"YugiKaiba": DECK})
    print("    → returned cleanly")
    # B.3.a Chunk 3.7: load card embeddings so the state encoder writes
    # non-zero floats. Without this the obs:state_ tensor is all zeros.
    emb_path = REPO / "src" / "card_embeddings.bin"
    if emb_path.is_file() and hasattr(edopro_mod, "init_embeddings_store"):
        edopro_mod.init_embeddings_store(str(emb_path))
        print(f"    → embeddings store loaded from {emb_path.name}")


def step3_construct_envpool():
    print("[3] construct EnvPool via ygoenv.make(EDOPro-v0)")
    pool = ygoenv.make(
        task_id="EDOPro-v0",
        env_type="gymnasium",
        num_envs=1,
        num_threads=1,
        seed=42,
        deck1="YugiKaiba",
        deck2="YugiKaiba",
        player=-1,
        play_mode="self",
        verbose=False,
        max_options=16,
        n_history_actions=16,
        record=False,
    )
    print(f"    → pool={type(pool).__name__}")
    return pool


def obs_fingerprint(obs) -> str:
    h = hashlib.sha256()
    for k in sorted(obs.keys()):
        v = obs[k]
        if hasattr(v, "tobytes"):
            h.update(k.encode())
            h.update(v.tobytes())
    return h.hexdigest()[:16]


def _extract_obs(o):
    """Normalize pool.reset() / pool.step() output to a dict of ndarrays."""
    if isinstance(o, tuple) and len(o) >= 1:
        o = o[0] if isinstance(o[0], dict) else dict(
            (k, getattr(o, k)) for k in o._fields if hasattr(o, k))
    if isinstance(o, dict):
        return {k: np.asarray(v) for k, v in o.items() if v is not None}
    d = {}
    for k in ("obs", "observation"):
        v = getattr(o, k, None)
        if v is not None and isinstance(v, dict):
            d.update({kk: np.asarray(vv) for kk, vv in v.items()})
            return d
    return d


def step4_deterministic_reset(pool_a, pool_b) -> bool:
    print("[4] reset determinism: two runs, same seed → identical obs?")
    obs_a = pool_a.reset()
    obs_b = pool_b.reset()

    a_dict = _extract_obs(obs_a)
    b_dict = _extract_obs(obs_b)

    if not a_dict or not b_dict:
        print(f"    ! could not coerce obs to dict. type(obs_a)={type(obs_a)}")
        return False

    fp_a = obs_fingerprint(a_dict)
    fp_b = obs_fingerprint(b_dict)
    print(f"    run A fingerprint: {fp_a}")
    print(f"    run B fingerprint: {fp_b}")
    match = (fp_a == fp_b)
    print(f"    → match: {match}")

    for k in sorted(a_dict.keys()):
        a = a_dict[k]
        b = b_dict[k]
        same = np.array_equal(a, b)
        print(f"      - {k:25s} shape={a.shape} dtype={a.dtype} match={same}")

    return match and obs_a is not None


def _rollout(pool, obs0, n_steps: int):
    """Step the env n_steps times; at each step pick action 0 (first legal).

    Returns a list of (obs_fingerprint, reward, done, truncated) per step.
    Stops early on episode termination (all envs done)."""
    trajectory = []
    obs = obs0
    for step_i in range(n_steps):
        d = _extract_obs(obs)
        # The env exposes info:num_options; if available, clamp action to it.
        # Action 0 is always legal by construction of the action space.
        action = np.array([0], dtype=np.int32)
        try:
            stepped = pool.step(action)
        except Exception as e:
            trajectory.append(("EXC", str(e)[:200], True, False))
            break

        if isinstance(stepped, tuple):
            if len(stepped) == 5:
                obs, reward, done, truncated, info = stepped
            elif len(stepped) == 4:
                obs, reward, done, info = stepped
                truncated = None
            else:
                obs = stepped[0]
                reward = stepped[1] if len(stepped) > 1 else None
                done = stepped[2] if len(stepped) > 2 else None
                truncated = None
        else:
            obs = stepped
            reward = None
            done = None
            truncated = None

        fp = obs_fingerprint(_extract_obs(obs))
        r = float(np.asarray(reward).flatten()[0]) if reward is not None else None
        tr = bool(np.asarray(truncated).flatten()[0]) if truncated is not None else None
        dn = bool(np.asarray(done).flatten()[0]) if done is not None else None
        trajectory.append((fp, r, dn, tr))

        if dn:
            break

    return trajectory


def step5_deterministic_rollout(n_steps: int = 30) -> bool:
    print(f"[5] step determinism: same seed + same actions ({n_steps} steps)")
    pool_a = step3_construct_envpool()
    pool_b = step3_construct_envpool()
    obs_a = pool_a.reset()
    obs_b = pool_b.reset()
    traj_a = _rollout(pool_a, obs_a, n_steps)
    traj_b = _rollout(pool_b, obs_b, n_steps)

    print(f"    A rolled {len(traj_a)} steps; B rolled {len(traj_b)} steps")
    if len(traj_a) != len(traj_b):
        print("    ✗ trajectory length mismatch")
        return False

    mismatches = 0
    for i, (a, b) in enumerate(zip(traj_a, traj_b)):
        if a != b:
            mismatches += 1
            if mismatches <= 5:
                print(f"    ✗ step {i}: A={a} B={b}")
    if mismatches == 0:
        print(f"    → all {len(traj_a)} steps byte-identical between A and B")
        return True
    print(f"    ✗ {mismatches} of {len(traj_a)} steps diverged")
    return False


def main() -> None:
    step1_module_import()
    step2_init_module()
    pool_a = step3_construct_envpool()
    pool_b = step3_construct_envpool()
    reset_ok = step4_deterministic_reset(pool_a, pool_b)
    step_ok = step5_deterministic_rollout(n_steps=30)
    print()
    print(f"[summary] reset-determinism: {'PASS' if reset_ok else 'FAIL'}")
    print(f"          step-determinism:  {'PASS' if step_ok else 'FAIL'}")
    if not (reset_ok and step_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
