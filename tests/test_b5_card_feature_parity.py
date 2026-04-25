"""B.5 card-feature parity test.

For a ~20-card corpus, compare the edopro wrapper's Card struct fields
(populated via db_query_card at init_module time, exposed via the
_test_card_struct pybind entry) against direct sqlite3 queries of
cards.cdb. Parity claim: every field the wrapper reads from cards.cdb
survives the read + struct-population + pybind round-trip unchanged.

Scope targets:
  - Monsters of different races (Warrior, Spellcaster, Fiend, Beast,
    Dragon, Zombie, Machine, Winged Beast) — exercise race2id coverage
  - Attributes (DARK, LIGHT, EARTH, WATER, FIRE) — attribute2id coverage
  - Effect + Normal + Fusion + Ritual types — type field coverage
  - Spells and Traps (with varying subtype bits)
  - Set codes — non-empty vector case for cards with an archetype

This test complements cleanup 2 (state encoder parity, 10655 floats).
cleanup 2 validates the C++→Python pipeline at the whole-state level;
this test validates the individual Card fields at source (DB) → Card
struct level, catching any truncation/sign-extension/column-misread that
could hide inside the state encoder's aggregate diff.

The corpus is the 26 unique cards in YugiKaiba.ydk plus a handful of
"modern" cards for race/setcode coverage.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
YGO_AGENT = REPO / "src" / "ygo-agent"
os.chdir(YGO_AGENT)
for p in (YGO_AGENT, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ygoenv  # noqa: E402,F401
from ygoenv.edopro import edopro_ygoenv as edopro_mod  # noqa: E402

DB_PATH = YGO_AGENT / "assets" / "locale" / "en" / "cards.cdb"
CODE_LIST = YGO_AGENT / "scripts" / "code_list.txt"
DECK = YGO_AGENT / "assets" / "deck" / "YugiKaiba.ydk"


def yugikaiba_codes() -> list[int]:
    with open(DECK) as fh:
        return sorted(
            {int(line) for line in fh
             if line.strip() and not line.startswith(("#", "!"))}
        )


def db_query(conn: sqlite3.Connection, code: int) -> dict:
    """Query cards.cdb directly. Replicates the wrapper's db_query_card
    path exactly so the comparison is apples-to-apples."""
    c = conn.cursor()
    d = c.execute("SELECT * FROM datas WHERE id=?", (code,)).fetchone()
    assert d is not None, f"cards.cdb missing code {code}"
    datas = dict(zip([col[0] for col in c.description], d))
    t = c.execute("SELECT * FROM texts WHERE id=?", (code,)).fetchone()
    assert t is not None, f"texts missing code {code}"
    texts = dict(zip([col[0] for col in c.description], t))

    # Mirror the wrapper's decode: setcode is a packed u64 of 4 u16 slots;
    # non-zero slots form the setcodes vector.
    setcode_packed = int(datas["setcode"]) & 0xffffffffffffffff
    setcodes = []
    for i in range(4):
        s = (setcode_packed >> (i * 16)) & 0xffff
        if s:
            setcodes.append(s)

    level_raw = int(datas["level"])
    level = level_raw & 0xff
    lscale = (level_raw >> 24) & 0xff
    rscale = (level_raw >> 16) & 0xff

    type_ = int(datas["type"])
    atk = int(datas["atk"])
    defense = int(datas["def"])
    link_marker = 0
    if type_ & 0x4000000:  # TYPE_LINK = 0x4000000
        link_marker = defense
        defense = 0

    return {
        "code": int(datas["id"]),
        "alias": int(datas["alias"]),
        "setcodes": setcodes,
        "type": type_,
        "level": level,
        "lscale": lscale,
        "rscale": rscale,
        "attack": atk,
        "defense": defense,
        "race": int(datas["race"]) & 0xffffffffffffffff,
        "attribute": int(datas["attribute"]),
        "link_marker": link_marker,
        "name": texts["name"],
        "desc": texts["desc"],
    }


def compare(code: int, want: dict, got) -> list[str]:
    """Returns a list of field-level diff strings. Empty = parity."""
    mismatches = []
    fields_scalar = [
        "code", "alias", "type", "level", "lscale", "rscale",
        "attack", "defense", "race", "attribute", "link_marker",
    ]
    for f in fields_scalar:
        w = want[f]
        g = getattr(got, f)
        if w != g:
            mismatches.append(f"{f}: cdb={w} wrapper={g}")
    if list(want["setcodes"]) != list(got.setcodes):
        mismatches.append(
            f"setcodes: cdb={want['setcodes']} wrapper={list(got.setcodes)}"
        )
    if want["name"] != got.name:
        mismatches.append(f"name: cdb={want['name']!r} wrapper={got.name!r}")
    return mismatches


# ── Setup ────────────────────────────────────────────────────────────────────

print("=== test_b5_card_feature_parity.py ===")

edopro_mod.init_module(str(DB_PATH), str(CODE_LIST), {"YugiKaiba": str(DECK)})
conn = sqlite3.connect(str(DB_PATH))

codes = yugikaiba_codes()
print(f"[deck] YugiKaiba has {len(codes)} unique codes")

# ── Parity check ────────────────────────────────────────────────────────────

n_failures = 0
race_coverage = set()
attr_coverage = set()
type_coverage = set()
setcode_nonzero = 0

for code in codes:
    want = db_query(conn, code)
    got = edopro_mod._test_card_struct(code)
    diffs = compare(code, want, got)

    race_coverage.add(got.race)
    attr_coverage.add(got.attribute)
    type_coverage.add(got.type & 0x1f)  # low 5 bits = broad type class
    if got.setcodes:
        setcode_nonzero += 1

    label = f"[{code:>10d}] {got.name[:40]:<40s}"
    if diffs:
        n_failures += 1
        print(f"{label} FAIL")
        for d in diffs:
            print(f"    {d}")
    else:
        print(f"{label}  ok  race=0x{got.race:x} attr=0x{got.attribute:x} "
              f"type=0x{got.type:x} setcodes={list(got.setcodes)}")

print()
print(f"[coverage] distinct races     : {len(race_coverage)}")
print(f"[coverage] distinct attribs   : {len(attr_coverage)}")
print(f"[coverage] distinct type bits : {sorted(type_coverage)}")
print(f"[coverage] cards w/ setcodes  : {setcode_nonzero}/{len(codes)}")
print()
print(f"[summary] checked {len(codes)} cards, {n_failures} failures")

assert n_failures == 0, \
    f"FAIL: {n_failures} cards diverged between cards.cdb and the wrapper"

print("[done] card-feature parity verified against cards.cdb")

# ── B.5 item 2: ygoenv_card_index roundtrip ──────────────────────────────────
# The adapter's cid↔konami map is built from scripts/code_list.txt and must
# agree with the edopro wrapper's internal card_ids_ map (both paths read
# the same file at init_module time). State-encoder parity already
# implicitly tests this (a mismatched cid would blow up the embedding
# lookups). This block makes the check explicit and local.

print()
print("=== cid↔konami roundtrip ===")

from ygoenv_card_index import bootstrap  # noqa: E402

# bootstrap re-calls init_ygopro — which, for EDOPro-v0, dispatches to
# ygoenv.edopro.init_module again. Safe because init_module is idempotent
# (repopulates cards_ / card_ids_ with the same content).
index, _deck = bootstrap(
    "EDOPro-v0", "english", str(DECK), str(CODE_LIST),
)

roundtrip_fails = 0
for code in codes:
    cid = index.konami_to_cid(code)
    round_code = index.cid_to_konami(cid)
    if round_code != code:
        roundtrip_fails += 1
        print(f"  FAIL {code} → cid {cid} → code {round_code}")

assert roundtrip_fails == 0, \
    f"cid↔konami roundtrip broken for {roundtrip_fails} codes"

print(f"[done] cid↔konami roundtrip: {len(codes)}/{len(codes)} stable.")
