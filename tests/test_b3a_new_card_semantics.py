"""B.3.a regression test: YGO_NewCard owner/playerid semantics.

YugiKaiba self-play has owner == playerid for every card, so the wrapper's
owner/playerid semantic mapping can silently stay broken without affecting
training or the B.1/B.2 determinism tests. This test exercises an
asymmetric case (owner != playerid) and asserts the wrapper maps them
correctly onto edo9300's OCG_NewCardInfo per ocgapi.cpp:72.

Call path:
  - init_module (populates cards_data_ for the test code)
  - edopro_ygoenv._test_new_card_semantics(code, owner, playerid)
      which constructs a duel, calls OCG_DuelNewCard with asymmetric
      info.team/info.con, queries QUERY_OWNER, returns the observed u8.

Pass condition:
  observed == owner  (i.e. info.team mapped to pcard->owner)

Fail mode:
  observed == playerid  (the pre-B.3.a mapping; means someone reverted
                         the fix). Any other value is an unexpected
                         edo9300 change.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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

# Pick a code that init_module preloads. Summoned Skull from YugiKaiba
# (vanilla normal, no script) — simplest card to construct and query.
TEST_CODE = 70781052


def main() -> None:
    edopro_mod.init_module(DB_PATH, CODE_LIST, {"YugiKaiba": DECK})

    # Symmetric baseline — both runs should observe the same seat.
    same_00 = edopro_mod._test_new_card_semantics(TEST_CODE, 0, 0)
    same_11 = edopro_mod._test_new_card_semantics(TEST_CODE, 1, 1)
    assert same_00 == 0, f"baseline (0,0): expected 0, got {same_00}"
    assert same_11 == 1, f"baseline (1,1): expected 1, got {same_11}"
    print(f"[baseline] owner=0 playerid=0 → observed {same_00}  PASS")
    print(f"[baseline] owner=1 playerid=1 → observed {same_11}  PASS")

    # Asymmetric cases — the ones YugiKaiba can't reach. Observed MUST
    # equal owner (QUERY_OWNER returns pcard->owner = info.team = our
    # `owner` arg after the B.3.a fix).
    cases = [
        (TEST_CODE, 0, 1),  # owner=0, placed in seat 1
        (TEST_CODE, 1, 0),  # owner=1, placed in seat 0
    ]
    for code, owner, playerid in cases:
        observed = edopro_mod._test_new_card_semantics(code, owner, playerid)
        status = "PASS" if observed == owner else "FAIL"
        print(f"[asymmetric] owner={owner} playerid={playerid} → "
              f"observed {observed}  expected {owner}  {status}")
        assert observed == owner, (
            f"YGO_NewCard semantic fix regressed: owner={owner} "
            f"playerid={playerid}, observed={observed}. If observed=={playerid}, "
            "someone swapped info.team/info.con back to the pre-B.3.a mapping."
        )

    print("\n[done] YGO_NewCard owner/playerid semantics validated.")


if __name__ == "__main__":
    main()
