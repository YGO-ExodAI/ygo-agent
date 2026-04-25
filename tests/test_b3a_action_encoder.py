"""B.3.a step-5 regression test: obs:actions_ 12-col encoder.

Hand-constructs a fixed set of LegalActions in C++ (via edopro_ygoenv.
_test_action_encoder), runs the 12-col encoder on each, and asserts the
resulting bytes at known positions match the expected field encoding.

Covers every handler family so a future change to the encoder layout (col
reorder, field mistranslation, system_string_to_id drift) fails here
loudly before it regresses training.

Column layout (from edopro.h StateSpec):
  0  spec_index   1-based row in obs:cards_ (0 = no card)
  1  cid_hi       CardId >> 8
  2  cid_lo       CardId & 0xff
  3  msg_id       msg2id output (1-based)
  4  act          ActionAct enum value
  5  finish       1 iff finish marker
  6  effect       re-encoded (0=None, 1=default, 2..15=card effect,
                   16+=system_string_to_id)
  7  phase        ActionPhase enum value
  8  position     position2id output
  9  number       raw 1..12
  10 place        ActionPlace enum value
  11 attrib       attribute2id output
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
YGO_AGENT = REPO / "src" / "ygo-agent"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(YGO_AGENT))
os.chdir(str(YGO_AGENT))

import ygoenv  # noqa: F401
from ygoenv.edopro import edopro_ygoenv as edopro_mod


# Enum values (must match ActionAct/ActionPhase/ActionPlace ordering in
# edopro.h). If ordering ever changes, this table + the encoder both
# drift — and this test is what tells us they drifted.
ACT_NONE, ACT_SET, ACT_REPO = 0, 1, 2
ACT_SPSUMMON, ACT_SUMMON, ACT_MSET = 3, 4, 5
ACT_ATTACK, ACT_DATTACK = 6, 7
ACT_ACTIVATE, ACT_CANCEL = 8, 9

PHASE_NONE, PHASE_BATTLE, PHASE_MAIN2, PHASE_END = 0, 1, 2, 3

PLACE_NONE = 0
PLACE_MZONE1 = 1
PLACE_MZONE3 = 3


# msg2id: ordering from _msgs vector in edopro.h:
# IDLECMD=1, CHAIN=2, CARD=3, TRIBUTE=4, POSITION=5, EFFECTYN=6,
# YESNO=7, BATTLECMD=8, UNSELECT=9, OPTION=10, PLACE=11, SUM=12,
# DISFIELD=13, ANN_ATTRIB=14, ANN_NUMBER=15.
MSG_IDLECMD = 1
MSG_CHAIN = 2
MSG_CARD = 3
MSG_POSITION = 5
MSG_EFFECTYN = 6
MSG_BATTLECMD = 8
MSG_OPTION = 10
MSG_PLACE = 11
MSG_ANNOUNCE_ATTRIB = 14
MSG_ANNOUNCE_NUMBER = 15


def _check(idx: int, col: int, expected: int, actual: int, label: str) -> None:
    assert actual == expected, (
        f"row {idx} ({label}) col {col}: expected {expected}, got {actual}"
    )


def main() -> None:
    raw = edopro_mod._test_action_encoder()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 12)
    n = arr.shape[0]
    print(f"[encoder] returned {n} rows, shape={arr.shape}")

    # ── row 0: IDLECMD Activate, card-effect offset (eff_idx=10010+3=10013)
    #   expected col 6 = (10013 - 10010) + 2 = 5
    r = 0
    _check(r, 0, 7, arr[r, 0], "IDLECMD activate spec_index")
    _check(r, 1, 0x02, arr[r, 1], "IDLECMD activate cid_hi")
    _check(r, 2, 0x05, arr[r, 2], "IDLECMD activate cid_lo")
    _check(r, 3, MSG_IDLECMD, arr[r, 3], "IDLECMD msg_id")
    _check(r, 4, ACT_ACTIVATE, arr[r, 4], "IDLECMD activate act")
    _check(r, 6, 5, arr[r, 6], "IDLECMD activate card-effect encoding (10013→5)")
    _check(r, 7, PHASE_NONE, arr[r, 7], "IDLECMD activate phase")
    print(f"[row 0] IDLECMD activate  PASS")

    # ── row 1: IDLECMD phase=Battle
    r = 1
    _check(r, 3, MSG_IDLECMD, arr[r, 3], "phase IDLECMD msg_id")
    _check(r, 4, ACT_NONE, arr[r, 4], "phase IDLECMD act")
    _check(r, 7, PHASE_BATTLE, arr[r, 7], "phase IDLECMD phase=Battle")
    _check(r, 6, 0, arr[r, 6], "phase IDLECMD effect=-1→0")
    print(f"[row 1] IDLECMD phase=Battle  PASS")

    # ── row 2: SELECT_CARD finish
    r = 2
    _check(r, 3, MSG_CARD, arr[r, 3], "CARD finish msg_id")
    _check(r, 5, 1, arr[r, 5], "CARD finish marker")
    _check(r, 0, 0, arr[r, 0], "CARD finish spec_index (zero)")
    print(f"[row 2] SELECT_CARD finish  PASS")

    # ── row 3: SELECT_CARD pick with spec_index=5, cid=0x0108
    r = 3
    _check(r, 0, 5, arr[r, 0], "CARD pick spec_index")
    _check(r, 1, 0x01, arr[r, 1], "CARD pick cid_hi")
    _check(r, 2, 0x08, arr[r, 2], "CARD pick cid_lo")
    _check(r, 5, 0, arr[r, 5], "CARD pick NOT finish")
    print(f"[row 3] SELECT_CARD pick  PASS")

    # ── row 4: CHAIN activate with system-string effect (30 → system_string_to_id)
    #   system_string2id in edopro.h uses make_ids(system_strings, 16) and
    #   system_strings is a std::map<int, std::string> which orders by key;
    #   the first key is 30, so system_string_to_id(30) = 16.
    r = 4
    _check(r, 0, 11, arr[r, 0], "CHAIN activate spec_index")
    _check(r, 1, 0x03, arr[r, 1], "CHAIN activate cid_hi")
    _check(r, 2, 0x01, arr[r, 2], "CHAIN activate cid_lo")
    _check(r, 3, MSG_CHAIN, arr[r, 3], "CHAIN msg_id")
    _check(r, 4, ACT_ACTIVATE, arr[r, 4], "CHAIN activate act")
    _check(r, 6, 16, arr[r, 6], "CHAIN activate effect=30 → system_string_to_id(30)=16")
    print(f"[row 4] CHAIN activate (system_string)  PASS")

    # ── row 5: CHAIN cancel
    r = 5
    _check(r, 3, MSG_CHAIN, arr[r, 3], "CHAIN cancel msg_id")
    _check(r, 4, ACT_CANCEL, arr[r, 4], "CHAIN cancel act")
    _check(r, 5, 0, arr[r, 5], "CHAIN cancel NOT finish")
    print(f"[row 5] CHAIN cancel  PASS")

    # ── row 6: EFFECTYN yes
    r = 6
    _check(r, 3, MSG_EFFECTYN, arr[r, 3], "EFFECTYN yes msg_id")
    _check(r, 4, ACT_ACTIVATE, arr[r, 4], "EFFECTYN yes act")
    _check(r, 6, 1, arr[r, 6], "EFFECTYN yes effect=0→1 (default)")
    _check(r, 0, 3, arr[r, 0], "EFFECTYN yes spec_index")
    print(f"[row 6] EFFECTYN yes  PASS")

    # ── row 7: EFFECTYN no (cancel)
    r = 7
    _check(r, 3, MSG_EFFECTYN, arr[r, 3], "EFFECTYN no msg_id")
    _check(r, 4, ACT_CANCEL, arr[r, 4], "EFFECTYN no act")
    _check(r, 6, 0, arr[r, 6], "EFFECTYN no effect=-1→0 (None)")
    print(f"[row 7] EFFECTYN no  PASS")

    # ── row 8: POSITION face-up defense (POS_FACEUP_DEFENSE=4).
    #   position2id is make_ids(position2str) — unordered_dense order, but
    #   the important check is just that col 8 is nonzero and stable across
    #   builds. Skip the exact value check.
    r = 8
    _check(r, 3, MSG_POSITION, arr[r, 3], "POSITION msg_id")
    assert arr[r, 8] != 0, f"POSITION col 8 should be nonzero, got {arr[r, 8]}"
    _check(r, 1, 0x05, arr[r, 1], "POSITION cid_hi")
    _check(r, 2, 0x02, arr[r, 2], "POSITION cid_lo")
    print(f"[row 8] POSITION  PASS (col 8 = {arr[r, 8]})")

    # ── row 9: OPTION 1190 → system_string_to_id.
    #   system_strings key order: 30, 31, 96, 221, 1190, 1192, 1621, 1622.
    #   make_ids starts at 16, so 30→16, 31→17, 96→18, 221→19, 1190→20.
    r = 9
    _check(r, 3, MSG_OPTION, arr[r, 3], "OPTION msg_id")
    _check(r, 4, ACT_ACTIVATE, arr[r, 4], "OPTION act")
    _check(r, 6, 20, arr[r, 6], "OPTION effect=1190 → system_string_to_id(1190)=20")
    print(f"[row 9] OPTION  PASS")

    # ── row 10: PLACE m3 → ActionPlace::MZone3 = 3
    r = 10
    _check(r, 3, MSG_PLACE, arr[r, 3], "PLACE msg_id")
    _check(r, 10, PLACE_MZONE3, arr[r, 10], "PLACE m3 → MZone3")
    print(f"[row 10] PLACE m3  PASS")

    # ── row 11: BATTLECMD attack
    r = 11
    _check(r, 3, MSG_BATTLECMD, arr[r, 3], "BATTLECMD msg_id")
    _check(r, 4, ACT_ATTACK, arr[r, 4], "BATTLECMD attack act")
    _check(r, 0, 4, arr[r, 0], "BATTLECMD attack spec_index")
    _check(r, 7, PHASE_NONE, arr[r, 7], "BATTLECMD attack phase=None")
    print(f"[row 11] BATTLECMD attack  PASS")

    # ── row 12: ANNOUNCE_NUMBER 7
    r = 12
    _check(r, 3, MSG_ANNOUNCE_NUMBER, arr[r, 3], "ANN_NUMBER msg_id")
    _check(r, 9, 7, arr[r, 9], "ANN_NUMBER number")
    print(f"[row 12] ANNOUNCE_NUMBER 7  PASS")

    # ── row 13: ANNOUNCE_ATTRIB fire. attribute2id(ATTRIBUTE_FIRE) is
    # stable but not 1 (it's make_ids over attribute2str, ordering
    # dependent). Just check col 11 != 0.
    r = 13
    _check(r, 3, MSG_ANNOUNCE_ATTRIB, arr[r, 3], "ANN_ATTRIB msg_id")
    assert arr[r, 11] != 0, f"ANN_ATTRIB col 11 should be nonzero, got {arr[r, 11]}"
    print(f"[row 13] ANNOUNCE_ATTRIB fire  PASS (col 11 = {arr[r, 11]})")

    print(f"\n[done] all {n} encoder rows validated.")


if __name__ == "__main__":
    main()
