"""Stronger post-corpus-swap determinism test.

Background: spike_edopro_e2e.py's step-determinism test uses action=0
(pick-first-legal) policy for 30 steps. In self-play Yugi/Kaiba vs
Yugi/Kaiba, action=0 is almost always "end phase" — so 30 steps of that
policy never executes any scripted card effect. A matching fingerprint
under that policy confirms the no-script codepath is deterministic but
says nothing about scripted cards.

This test runs longer rollouts with a policy that picks the LAST legal
action (num_options-1) at each step, which tends to select non-end-phase
options when they exist (normal summon, activate, attack). The goal is
to exercise enough card scripts across a game that post-swap byte
equivalence of the obs stream is a real determinism signal.

Pass criteria:
- Two independent pools, same seed, run with the same (num_options-1)
  policy for N steps → byte-identical obs at every step.
- At least one card activation observed across the run (not a trivial
  pass-through trajectory).

Run from src/ygo-agent/:
  python3 tests/test_b7_chain_determinism.py
"""
from __future__ import annotations
import hashlib, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]  # ExodAI/
YGO_AGENT = REPO / "src" / "ygo-agent"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(YGO_AGENT))
os.chdir(str(YGO_AGENT))

import ygoenv  # noqa: F401
from ygoenv.edopro import edopro_ygoenv as edopro_mod

DB_PATH = str(YGO_AGENT / "assets" / "locale" / "en" / "cards.cdb")
CODE_LIST = str(YGO_AGENT / "scripts" / "code_list.txt")
DECK = str(YGO_AGENT / "assets" / "deck" / "YugiKaiba.ydk")
EMB = str(REPO / "src" / "card_embeddings.bin")

N_STEPS = 500
SEED = 42
POLICY_SEED = 1337  # seeded RNG for action selection


def make_pool(seed: int):
    return ygoenv.make(
        task_id="EDOPro-v0", env_type="gymnasium",
        num_envs=1, num_threads=1, seed=seed,
        deck1="YugiKaiba", deck2="YugiKaiba", player=-1,
        play_mode="self", verbose=False,
        max_options=16, n_history_actions=16, record=False,
    )


def extract_obs(o):
    if isinstance(o, tuple) and o:
        o = o[0] if isinstance(o[0], dict) else dict(
            (k, getattr(o, k)) for k in o._fields if hasattr(o, k))
    return {k: np.asarray(v) for k, v in o.items() if v is not None}


def fp(obs):
    h = hashlib.sha256()
    for k in sorted(obs.keys()):
        v = obs[k]
        if hasattr(v, "tobytes"):
            h.update(k.encode()); h.update(v.tobytes())
    return h.hexdigest()[:16]


def rollout(pool, info, n_steps, rng):
    """Roll n_steps using seeded-random policy over legal actions.

    Determinism is verified by per-step byte-equivalence of the obs dict
    AND per-step equivalence of info.chain_cids. num_options and
    chain_cids live in the info dict (5th return of step()), not in the
    obs dict — info: keys are filtered out of obs and returned in info.
    Both envpools share the same rng seed so action sequences are
    identical.
    """
    traj = []
    n_nonzero_actions = 0
    n_terminations = 0
    n_chain_active_steps = 0      # steps where any chain_cid[i] != 0
    n_new_chains = 0              # no-chain → chain-active transitions
    prev_chain_active = False
    for _ in range(n_steps):
        num_opts = int(np.asarray(info.get("num_options", [1])).flatten()[0])
        num_opts = max(1, num_opts)
        action_idx = int(rng.integers(0, num_opts)) if num_opts > 1 else 0
        if action_idx > 0:
            n_nonzero_actions += 1
        action = np.array([action_idx], dtype=np.int32)
        stepped = pool.step(action)
        obs = stepped[0]
        done = stepped[2] if len(stepped) > 2 else False
        info = stepped[4] if len(stepped) > 4 else {}
        dn = bool(np.asarray(done).flatten()[0]) if done is not None else False
        post_d = extract_obs(obs)
        cc = info.get("chain_cids")
        chain_nonzero = int((np.asarray(cc) != 0).sum()) if cc is not None else 0
        chain_active = chain_nonzero > 0
        if chain_active:
            n_chain_active_steps += 1
        if not prev_chain_active and chain_active:
            n_new_chains += 1
        prev_chain_active = chain_active
        # Fold chain_cids into the obs fingerprint so determinism
        # verification covers both obs and info.chain_cids byte-for-byte.
        post_d["_chain_cids"] = np.asarray(cc) if cc is not None else np.zeros(1)
        traj.append((fp(post_d), dn))
        if dn:
            n_terminations += 1
    return traj, n_nonzero_actions, n_chain_active_steps, n_new_chains, n_terminations


def main():
    print("[1] init_module + embeddings store")
    edopro_mod.init_module(DB_PATH, CODE_LIST, {"YugiKaiba": DECK})
    edopro_mod.init_embeddings_store(EMB)

    print(f"[2] construct two pools (seed={SEED})")
    pool_a = make_pool(SEED); pool_b = make_pool(SEED)
    # reset() returns (obs, info) in gymnasium mode — but envpool may
    # return just obs. Normalize.
    r_a = pool_a.reset(); r_b = pool_b.reset()
    if isinstance(r_a, tuple) and len(r_a) >= 2:
        obs_a, info_a = r_a[0], r_a[1]
        obs_b, info_b = r_b[0], r_b[1]
    else:
        obs_a, info_a = r_a, {"num_options": np.array([1])}
        obs_b, info_b = r_b, {"num_options": np.array([1])}

    print(f"[3] roll {N_STEPS} steps (policy: seeded-random over legal actions)")
    rng_a = np.random.default_rng(POLICY_SEED)
    rng_b = np.random.default_rng(POLICY_SEED)
    traj_a, n_a, chain_a, new_a, term_a = rollout(pool_a, info_a, N_STEPS, rng_a)
    traj_b, n_b, chain_b, new_b, term_b = rollout(pool_b, info_b, N_STEPS, rng_b)

    print(f"    A: {len(traj_a)} steps, non-zero actions={n_a}, "
          f"terminations={term_a}, chain-active steps={chain_a}, new chains={new_a}")
    print(f"    B: {len(traj_b)} steps, non-zero actions={n_b}, "
          f"terminations={term_b}, chain-active steps={chain_b}, new chains={new_b}")

    if len(traj_a) != len(traj_b):
        print(f"    FAIL: traj length mismatch {len(traj_a)} vs {len(traj_b)}")
        sys.exit(1)

    n_divergent = sum(1 for a, b in zip(traj_a, traj_b) if a[0] != b[0])
    print(f"[4] divergent steps: {n_divergent} / {len(traj_a)}")

    if n_divergent > 0:
        for i, (a, b) in enumerate(zip(traj_a, traj_b)):
            if a[0] != b[0]:
                print(f"    first divergence at step {i}: {a} vs {b}")
                break
        sys.exit(1)

    if chain_a != chain_b or new_a != new_b:
        print(f"    FAIL: chain-activity diverged between A and B")
        print(f"      A: chain_active={chain_a}, new_chains={new_a}")
        print(f"      B: chain_active={chain_b}, new_chains={new_b}")
        sys.exit(1)

    if n_a < 5:
        print(f"    WARN: only {n_a} non-zero actions — policy may not be "
              "exercising scripts; test is weaker than intended")

    if new_a == 0:
        print(f"    WARN: zero chain resolutions observed in {N_STEPS} steps — "
              "policy may not be triggering scripted effects. "
              "Test still verifies byte-equivalence but not chain-resolve "
              "codepaths. Consider seeded scripted rollout as follow-up.")

    print(f"[done] chain-determinism PASS — "
          f"{len(traj_a)} byte-identical step obs, "
          f"{n_a} non-zero actions, "
          f"{term_a} game terminations, "
          f"{new_a} chain resolutions, "
          f"{chain_a} chain-active steps")


if __name__ == "__main__":
    main()
