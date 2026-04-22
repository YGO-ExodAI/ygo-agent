#ifndef YGOENV_EDOPro_EDOPro_H_
#define YGOENV_EDOPro_EDOPro_H_

// edo9300-lineage ygocore wrapper. Default training path as of B.8
// cutover (2026-04-21). TLV parser + encoder port landed in B.3.a;
// full regression history in src/docs/edopro_b*.md.

// clang-format off
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cstring>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <iostream>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/VariadicBind.h>
#include <ankerl/unordered_dense.h>

#include "ygoenv/core/async_envpool.h"
#include "ygoenv/core/env.h"

#include "edopro-core/common.h"
#include "edopro-core/card.h"
#include "edopro-core/ocgapi.h"

// Shared with ygopro wrapper — moved to common/ during B.3.a Chunk 1.
#include "ygoenv/common/card_embedding_store.h"

// clang-format on

namespace edopro {

inline std::vector<std::vector<int>> combinations(int n, int r) {
  std::vector<std::vector<int>> combs;
  std::vector<bool> m(n);
  std::fill(m.begin(), m.begin() + r, true);

  do {
    std::vector<int> cs;
    cs.reserve(r);
    for (int i = 0; i < n; ++i) {
      if (m[i]) {
        cs.push_back(i);
      }
    }
    combs.push_back(cs);
  } while (std::prev_permutation(m.begin(), m.end()));

  return combs;
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind, int i,
                   int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  if (i == n - 1) {
    return r == 1 || (w[ind[i]] == r);
  }
  return sum_to(w, ind, i + 1, r - 1) || sum_to(w, ind, i + 1, r - w[ind[i]]);
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind,
                   int r) {
  return sum_to(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight(const std::vector<int> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int i, int r,
                    bool max = false) {
  if (r <= 0) {
    if (max) {
      return true;
    } else {
      return false;
    }
  }
  int n = ind.size();
  const auto &w_ = w[ind[i]];
  if (i == n - 1) {
    if (w_.size() == 1) {
      if (max) {
        return w_[0] >= r;
      } else {
        return w_[0] == r;
      }
    } else {
      if (max) {
        return w_[0] >= r || w_[1] >= r;
      } else {
        return w_[0] == r || w_[1] == r;
      }
    }
  }
  if (w_.size() == 1) {
    return sum_to2(w, ind, i + 1, r - w_[0], max);
  } else {
    return sum_to2(w, ind, i + 1, r - w_[0], max) ||
           sum_to2(w, ind, i + 1, r - w_[1], max);
  }
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int r, bool max = false) {
  return sum_to2(w, ind, 0, r, max);
}

inline std::vector<std::vector<int>>
combinations_with_weight2(const std::vector<std::vector<int>> &weights,
                          int r, bool max = false) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to2(weights, comb, r, max)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

static std::string msg_to_string(int msg) {
  switch (msg) {
  case MSG_RETRY:
    return "retry";
  case MSG_HINT:
    return "hint";
  case MSG_WIN:
    return "win";
  case MSG_SELECT_BATTLECMD:
    return "select_battlecmd";
  case MSG_SELECT_IDLECMD:
    return "select_idlecmd";
  case MSG_SELECT_EFFECTYN:
    return "select_effectyn";
  case MSG_SELECT_YESNO:
    return "select_yesno";
  case MSG_SELECT_OPTION:
    return "select_option";
  case MSG_SELECT_CARD:
    return "select_card";
  case MSG_SELECT_CHAIN:
    return "select_chain";
  case MSG_SELECT_PLACE:
    return "select_place";
  case MSG_SELECT_POSITION:
    return "select_position";
  case MSG_SELECT_TRIBUTE:
    return "select_tribute";
  case MSG_SELECT_COUNTER:
    return "select_counter";
  case MSG_SELECT_SUM:
    return "select_sum";
  case MSG_SELECT_DISFIELD:
    return "select_disfield";
  case MSG_SORT_CARD:
    return "sort_card";
  case MSG_SELECT_UNSELECT_CARD:
    return "select_unselect_card";
  case MSG_CONFIRM_DECKTOP:
    return "confirm_decktop";
  case MSG_CONFIRM_CARDS:
    return "confirm_cards";
  case MSG_SHUFFLE_DECK:
    return "shuffle_deck";
  case MSG_SHUFFLE_HAND:
    return "shuffle_hand";
  case MSG_SWAP_GRAVE_DECK:
    return "swap_grave_deck";
  case MSG_SHUFFLE_SET_CARD:
    return "shuffle_set_card";
  case MSG_REVERSE_DECK:
    return "reverse_deck";
  case MSG_DECK_TOP:
    return "deck_top";
  case MSG_SHUFFLE_EXTRA:
    return "shuffle_extra";
  case MSG_NEW_TURN:
    return "new_turn";
  case MSG_NEW_PHASE:
    return "new_phase";
  case MSG_CONFIRM_EXTRATOP:
    return "confirm_extratop";
  case MSG_MOVE:
    return "move";
  case MSG_POS_CHANGE:
    return "pos_change";
  case MSG_SET:
    return "set";
  case MSG_SWAP:
    return "swap";
  case MSG_FIELD_DISABLED:
    return "field_disabled";
  case MSG_SUMMONING:
    return "summoning";
  case MSG_SUMMONED:
    return "summoned";
  case MSG_SPSUMMONING:
    return "spsummoning";
  case MSG_SPSUMMONED:
    return "spsummoned";
  case MSG_FLIPSUMMONING:
    return "flipsummoning";
  case MSG_FLIPSUMMONED:
    return "flipsummoned";
  case MSG_CHAINING:
    return "chaining";
  case MSG_CHAINED:
    return "chained";
  case MSG_CHAIN_SOLVING:
    return "chain_solving";
  case MSG_CHAIN_SOLVED:
    return "chain_solved";
  case MSG_CHAIN_END:
    return "chain_end";
  case MSG_CHAIN_NEGATED:
    return "chain_negated";
  case MSG_CHAIN_DISABLED:
    return "chain_disabled";
  case MSG_RANDOM_SELECTED:
    return "random_selected";
  case MSG_BECOME_TARGET:
    return "become_target";
  case MSG_DRAW:
    return "draw";
  case MSG_DAMAGE:
    return "damage";
  case MSG_RECOVER:
    return "recover";
  case MSG_EQUIP:
    return "equip";
  case MSG_LPUPDATE:
    return "lpupdate";
  case MSG_CARD_TARGET:
    return "card_target";
  case MSG_CANCEL_TARGET:
    return "cancel_target";
  case MSG_PAY_LPCOST:
    return "pay_lpcost";
  case MSG_ADD_COUNTER:
    return "add_counter";
  case MSG_REMOVE_COUNTER:
    return "remove_counter";
  case MSG_ATTACK:
    return "attack";
  case MSG_BATTLE:
    return "battle";
  case MSG_ATTACK_DISABLED:
    return "attack_disabled";
  case MSG_DAMAGE_STEP_START:
    return "damage_step_start";
  case MSG_DAMAGE_STEP_END:
    return "damage_step_end";
  case MSG_MISSED_EFFECT:
    return "missed_effect";
  case MSG_TOSS_COIN:
    return "toss_coin";
  case MSG_TOSS_DICE:
    return "toss_dice";
  case MSG_ROCK_PAPER_SCISSORS:
    return "rock_paper_scissors";
  case MSG_HAND_RES:
    return "hand_res";
  case MSG_ANNOUNCE_RACE:
    return "announce_race";
  case MSG_ANNOUNCE_ATTRIB:
    return "announce_attrib";
  case MSG_ANNOUNCE_CARD:
    return "announce_card";
  case MSG_ANNOUNCE_NUMBER:
    return "announce_number";
  case MSG_CARD_HINT:
    return "card_hint";
  case MSG_TAG_SWAP:
    return "tag_swap";
  case MSG_RELOAD_FIELD:
    return "reload_field";
  case MSG_AI_NAME:
    return "ai_name";
  case MSG_SHOW_HINT:
    return "show_hint";
  case MSG_PLAYER_HINT:
    return "player_hint";
  case MSG_MATCH_KILL:
    return "match_kill";
  case MSG_CUSTOM_MSG:
    return "custom_msg";
  default:
    return "unknown_msg";
  }
}

// system string. Declared as std::map so make_ids can consume it (see
// make_ids overload at line ~541) — identical to ygopro.h:344.
static const std::map<int, std::string> system_strings = {
    {30, "Replay rules apply. Continue this attack?"},
    {31, "Attack directly with this monster?"},
    {96, "Use the effect of [%ls] to avoid destruction?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {1190, "Add to hand"},
    {1192, "Banish"},
    {1621, "Attack Negated"},
    {1622, "[%ls] Missed timing"}
};

static std::string get_system_string(uint32_t desc) {
  auto it = system_strings.find(desc);
  if (it != system_strings.end()) {
    return it->second;
  }
  return "system string " + std::to_string(desc);
}

static std::string ltrim(std::string s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

inline std::vector<std::string> flag_to_usable_cardspecs(uint32_t flag,
                                                         bool reverse = false) {
  std::string zone_names[4] = {"m", "s", "om", "os"};
  std::vector<std::string> specs;
  for (int j = 0; j < 4; j++) {
    uint32_t value = (flag >> (j * 8)) & 0xff;
    for (int i = 0; i < 8; i++) {
      bool avail = (value & (1 << i)) == 0;
      if (reverse) {
        avail = !avail;
      }
      if (avail) {
        specs.push_back(zone_names[j] + std::to_string(i + 1));
      }
    }
  }
  return specs;
}

inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos) {
  std::string spec;
  if (loc & LOCATION_HAND) {
    spec += "h";
  } else if (loc & LOCATION_MZONE) {
    spec += "m";
  } else if (loc & LOCATION_SZONE) {
    spec += "s";
  } else if (loc & LOCATION_GRAVE) {
    spec += "g";
  } else if (loc & LOCATION_REMOVED) {
    spec += "r";
  } else if (loc & LOCATION_EXTRA) {
    spec += "x";
  }
  spec += std::to_string(seq + 1);
  if (loc & LOCATION_OVERLAY) {
    spec.push_back('a' + pos);
  }
  return spec;
}

inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos,
                              bool opponent) {
  std::string spec = ls_to_spec(loc, seq, pos);
  if (opponent) {
    spec.insert(0, 1, 'o');
  }
  return spec;
}

inline std::tuple<uint8_t, uint8_t, uint8_t>
spec_to_ls(const std::string spec) {
  uint8_t loc;
  uint8_t seq;
  uint8_t pos = 0;
  int offset = 1;
  if (spec[0] == 'h') {
    loc = LOCATION_HAND;
  } else if (spec[0] == 'm') {
    loc = LOCATION_MZONE;
  } else if (spec[0] == 's') {
    loc = LOCATION_SZONE;
  } else if (spec[0] == 'g') {
    loc = LOCATION_GRAVE;
  } else if (spec[0] == 'r') {
    loc = LOCATION_REMOVED;
  } else if (spec[0] == 'x') {
    loc = LOCATION_EXTRA;
  } else if (std::isdigit(spec[0])) {
    loc = LOCATION_DECK;
    offset = 0;
  } else {
    throw std::runtime_error("Invalid location");
  }
  int end = offset;
  while (end < spec.size() && std::isdigit(spec[end])) {
    end++;
  }
  seq = std::stoi(spec.substr(offset, end - offset)) - 1;
  if (end < spec.size()) {
    pos = spec[end] - 'a';
  }
  return {loc, seq, pos};
}

inline uint32_t ls_to_spec_code(uint8_t loc, uint8_t seq, uint8_t pos,
                                bool opponent) {
  uint32_t c = opponent ? 1 : 0;
  c |= (loc << 8);
  c |= (seq << 16);
  c |= (pos << 24);
  return c;
}

inline uint32_t spec_to_code(const std::string &spec) {
  int offset = 0;
  bool opponent = false;
  if (spec[0] == 'o') {
    opponent = true;
    offset++;
  }
  auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
  return ls_to_spec_code(loc, seq, pos, opponent);
}

inline std::string code_to_spec(uint32_t spec_code) {
  uint8_t loc = (spec_code >> 8) & 0xff;
  uint8_t seq = (spec_code >> 16) & 0xff;
  uint8_t pos = (spec_code >> 24) & 0xff;
  bool opponent = (spec_code & 0xff) == 1;
  return ls_to_spec(loc, seq, pos, opponent);
}

static std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>> read_decks(const std::string &fp) {
  std::ifstream file(fp);
  std::string line;
  std::vector<uint32_t> main_deck, extra_deck, side_deck;
  bool found_extra = false;

  if (file.is_open()) {
    // Read the main deck
    while (std::getline(file, line)) {
      if (line.find("side") != std::string::npos) {
        break;
      }
      if (line.find("extra") != std::string::npos) {
        found_extra = true;
        break;
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        main_deck.push_back(std::stoul(line));
      }
    }

    // Read the extra deck
    if (found_extra) {
      while (std::getline(file, line)) {
        if (line.find("side") != std::string::npos) {
          break;
        }
        // Check if line contains only digits
        if (std::all_of(line.begin(), line.end(), ::isdigit)) {
          extra_deck.push_back(std::stoul(line));
        }
      }
    }

    // Read the side deck
    while (std::getline(file, line)) {
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        side_deck.push_back(std::stoul(line));
      }
    }

    file.close();
  } else {
    throw std::runtime_error(fmt::format("Unable to open deck file: {}", fp));
  }

  return std::make_tuple(main_deck, extra_deck, side_deck);
}

template <class K = uint8_t>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::map<K, std::string> &m, int id_offset = 0,
         int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  int i = 0;
  for (const auto &[k, v] : m) {
    if (i < m_offset) {
      i++;
      continue;
    }
    m2[k] = i - m_offset + id_offset;
    i++;
  }
  return m2;
}

template <class K = char>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::vector<K> &cmds, int id_offset = 0, int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  for (int i = m_offset; i < cmds.size(); i++) {
    m2[cmds[i]] = i - m_offset + id_offset;
  }
  return m2;
}

static std::string reason_to_string(uint8_t reason) {
  // !victory 0x0 Surrendered
  // !victory 0x1 LP reached 0
  // !victory 0x2 Cards can't be drawn
  // !victory 0x3 Time limit up
  // !victory 0x4 Lost connection
  switch (reason) {
  case 0x0:
    return "Surrendered";
  case 0x1:
    return "LP reached 0";
  case 0x2:
    return "Cards can't be drawn";
  case 0x3:
    return "Time limit up";
  case 0x4:
    return "Lost connection";
  default:
    return "Unknown";
  }
}

static const std::map<uint8_t, std::string> location2str = {
    {LOCATION_DECK, "Deck"},
    {LOCATION_HAND, "Hand"},
    {LOCATION_MZONE, "Main Monster Zone"},
    {LOCATION_SZONE, "Spell & Trap Zone"},
    {LOCATION_GRAVE, "Graveyard"},
    {LOCATION_REMOVED, "Banished"},
    {LOCATION_EXTRA, "Extra Deck"},
};

static const ankerl::unordered_dense::map<uint8_t, uint8_t> location2id =
    make_ids(location2str, 1);

inline uint8_t location_to_id(uint8_t location) {
  auto it = location2id.find(location);
  if (it != location2id.end()) {
    return it->second;
  }
  return 0;
}

// B.3.a step-4: system_string2id — mirrors ygopro.h:649. Starts at 16 so
// the 2..15 range is free for "card effect" encoding in the action obs
// col 6 (see _set_obs_action_effect below).
static const ankerl::unordered_dense::map<int, uint8_t> system_string2id =
    make_ids(system_strings, 16);

inline uint8_t system_string_to_id(int sys_id) {
  auto it = system_string2id.find(sys_id);
  if (it != system_string2id.end()) {
    return it->second;
  }
  return 0;
}

#define POS_NONE 0x0 // xyz materials (overlay)

static const std::map<uint8_t, std::string> position2str = {
    {POS_NONE, "none"},
    {POS_FACEUP_ATTACK, "face-up attack"},
    {POS_FACEDOWN_ATTACK, "face-down attack"},
    {POS_ATTACK, "attack"},
    {POS_FACEUP_DEFENSE, "face-up defense"},
    {POS_FACEUP, "face-up"},
    {POS_FACEDOWN_DEFENSE, "face-down defense"},
    {POS_FACEDOWN, "face-down"},
    {POS_DEFENSE, "defense"},
};

static const ankerl::unordered_dense::map<uint8_t, uint8_t> position2id =
    make_ids(position2str);

#define ATTRIBUTE_NONE 0x0 // token

static const std::map<uint8_t, std::string> attribute2str = {
    {ATTRIBUTE_NONE, "None"},   {ATTRIBUTE_EARTH, "Earth"},
    {ATTRIBUTE_WATER, "Water"}, {ATTRIBUTE_FIRE, "Fire"},
    {ATTRIBUTE_WIND, "Wind"},   {ATTRIBUTE_LIGHT, "Light"},
    {ATTRIBUTE_DARK, "Dark"},   {ATTRIBUTE_DIVINE, "Divine"},
};

static const ankerl::unordered_dense::map<uint8_t, uint8_t> attribute2id =
    make_ids(attribute2str);

#define RACE_NONE 0x0 // token

// B.5: race keys are u64 because edo9300's QUERY_RACE emits a u64 field
// (common.h:111+ defines RACE_* as 32-bit bits but the query payload is
// u64 so future RACE_* additions — RACE_YOKAI etc — can live above bit
// 31 without breaking the wire format). The u32 race constants auto-
// promote to u64 when the map is constructed.
static const std::map<uint64_t, std::string> race2str = {
    {uint64_t{RACE_NONE}, "None"},
    {uint64_t{RACE_WARRIOR}, "Warrior"},
    {uint64_t{RACE_SPELLCASTER}, "Spellcaster"},
    {uint64_t{RACE_FAIRY}, "Fairy"},
    {uint64_t{RACE_FIEND}, "Fiend"},
    {uint64_t{RACE_ZOMBIE}, "Zombie"},
    {uint64_t{RACE_MACHINE}, "Machine"},
    {uint64_t{RACE_AQUA}, "Aqua"},
    {uint64_t{RACE_PYRO}, "Pyro"},
    {uint64_t{RACE_ROCK}, "Rock"},
    {uint64_t{RACE_WINGEDBEAST}, "Windbeast"},
    {uint64_t{RACE_PLANT}, "Plant"},
    {uint64_t{RACE_INSECT}, "Insect"},
    {uint64_t{RACE_THUNDER}, "Thunder"},
    {uint64_t{RACE_DRAGON}, "Dragon"},
    {uint64_t{RACE_BEAST}, "Beast"},
    {uint64_t{RACE_BEASTWARRIOR}, "Beast Warrior"},
    {uint64_t{RACE_DINOSAUR}, "Dinosaur"},
    {uint64_t{RACE_FISH}, "Fish"},
    {uint64_t{RACE_SEASERPENT}, "Sea Serpent"},
    {uint64_t{RACE_REPTILE}, "Reptile"},
    {uint64_t{RACE_PSYCHIC}, "Psycho"},
    {uint64_t{RACE_DIVINE}, "Divine"},
    {uint64_t{RACE_CREATORGOD}, "Creator God"},
    {uint64_t{RACE_WYRM}, "Wyrm"},
    {uint64_t{RACE_CYBERSE}, "Cyberse"},
    {uint64_t{RACE_ILLUSION}, "Illusion'"}};

static const ankerl::unordered_dense::map<uint64_t, uint8_t> race2id =
    make_ids(race2str);

static const std::map<uint32_t, std::string> type2str = {
    {TYPE_MONSTER, "Monster"},
    {TYPE_SPELL, "Spell"},
    {TYPE_TRAP, "Trap"},
    {TYPE_NORMAL, "Normal"},
    {TYPE_EFFECT, "Effect"},
    {TYPE_FUSION, "Fusion"},
    {TYPE_RITUAL, "Ritual"},
    {TYPE_TRAPMONSTER, "Trap Monster"},
    {TYPE_SPIRIT, "Spirit"},
    {TYPE_UNION, "Union"},
    {TYPE_GEMINI, "Dual"},
    {TYPE_TUNER, "Tuner"},
    {TYPE_SYNCHRO, "Synchro"},
    {TYPE_TOKEN, "Token"},
    {TYPE_QUICKPLAY, "Quick-play"},
    {TYPE_CONTINUOUS, "Continuous"},
    {TYPE_EQUIP, "Equip"},
    {TYPE_FIELD, "Field"},
    {TYPE_COUNTER, "Counter"},
    {TYPE_FLIP, "Flip"},
    {TYPE_TOON, "Toon"},
    {TYPE_XYZ, "XYZ"},
    {TYPE_PENDULUM, "Pendulum"},
    {TYPE_SPSUMMON, "Special"},
    {TYPE_LINK, "Link"},
};

inline std::vector<uint8_t> type_to_ids(uint32_t type) {
  std::vector<uint8_t> ids;
  ids.reserve(type2str.size());
  for (const auto &[k, v] : type2str) {
    ids.push_back(std::min(1u, type & k));
  }
  return ids;
}

static const std::map<int, std::string> phase2str = {
    {PHASE_DRAW, "draw phase"},
    {PHASE_STANDBY, "standby phase"},
    {PHASE_MAIN1, "main1 phase"},
    {PHASE_BATTLE_START, "battle start phase"},
    {PHASE_BATTLE_STEP, "battle step phase"},
    {PHASE_DAMAGE, "damage phase"},
    {PHASE_DAMAGE_CAL, "damage calculation phase"},
    {PHASE_BATTLE, "battle phase"},
    {PHASE_MAIN2, "main2 phase"},
    {PHASE_END, "end phase"},
};

static const ankerl::unordered_dense::map<int, uint8_t> phase2id =
    make_ids(phase2str);

static const std::vector<int> _msgs = {
    MSG_SELECT_IDLECMD,  MSG_SELECT_CHAIN,     MSG_SELECT_CARD,
    MSG_SELECT_TRIBUTE,  MSG_SELECT_POSITION,  MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,    MSG_SELECT_BATTLECMD, MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_OPTION,   MSG_SELECT_PLACE,     MSG_SELECT_SUM,
    MSG_SELECT_DISFIELD, MSG_ANNOUNCE_ATTRIB,  MSG_ANNOUNCE_NUMBER,
};

static const ankerl::unordered_dense::map<int, uint8_t> msg2id =
    make_ids(_msgs, 1);

static const ankerl::unordered_dense::map<char, uint8_t> cmd_act2id =
    make_ids({'t', 'r', 'c', 's', 'm', 'a', 'v'}, 1);

static const ankerl::unordered_dense::map<char, uint8_t> cmd_phase2id =
    make_ids(std::vector<char>({'b', 'm', 'e'}), 1);

static const ankerl::unordered_dense::map<char, uint8_t> cmd_yesno2id =
    make_ids(std::vector<char>({'y', 'n'}), 1);

static const ankerl::unordered_dense::map<std::string, uint8_t> cmd_place2id =
    make_ids(std::vector<std::string>(
                 {"m1",  "m2",  "m3",  "m4",  "m5",  "m6",  "m7",  "s1",
                  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",  "s8",  "om1",
                  "om2", "om3", "om4", "om5", "om6", "om7", "os1", "os2",
                  "os3", "os4", "os5", "os6", "os7", "os8"}),
             1);

inline std::string phase_to_string(int phase) {
  auto it = phase2str.find(phase);
  if (it != phase2str.end()) {
    return it->second;
  }
  return "unknown";
}

inline std::string position_to_string(int position) {
  auto it = position2str.find(position);
  if (it != position2str.end()) {
    return it->second;
  }
  return "unknown";
}

inline std::pair<uint8_t, uint8_t> float_transform(int x) {
  x = x % 65536;
  return {
      static_cast<uint8_t>(x >> 8),
      static_cast<uint8_t>(x & 0xff),
  };
}

static std::vector<int> find_substrs(const std::string &str,
                                     const std::string &substr) {
  std::vector<int> res;
  int pos = 0;
  while ((pos = str.find(substr, pos)) != std::string::npos) {
    res.push_back(pos);
    pos += substr.length();
  }
  return res;
}

inline std::string time_now() {
  // strftime %Y-%m-%d %H-%M-%S
  time_t now = time(0);
  tm *ltm = localtime(&now);
  char buffer[80];
  strftime(buffer, 80, "%Y-%m-%d %H-%M-%S", ltm);
  return std::string(buffer);
}

// from Multirole/YGOPro/Replay.cpp

enum ReplayTypes
{
	REPLAY_YRP1 = 0x31707279,
	REPLAY_YRPX = 0x58707279
};

enum ReplayFlags
{
	REPLAY_COMPRESSED      = 0x1,
	REPLAY_TAG             = 0x2,
	REPLAY_DECODED         = 0x4,
	REPLAY_SINGLE_MODE     = 0x8,
	REPLAY_LUA64           = 0x10,
	REPLAY_NEWREPLAY       = 0x20,
	REPLAY_HAND_TEST       = 0x40,
	REPLAY_DIRECT_SEED     = 0x80,
	REPLAY_64BIT_DUELFLAG  = 0x100,
	REPLAY_EXTENDED_HEADER = 0x200,
};

struct ReplayHeader
{
	uint32_t type; // See ReplayTypes.
	uint32_t version; // Unused atm, should be set to YGOPro::ClientVersion.
	uint32_t flags; // See ReplayFlags.
	uint32_t timestamp; // Unix timestamp.
	uint32_t size; // Uncompressed size of whatever is after this header.
	uint32_t hash; // Unused.
	uint8_t props[8U]; // Used for LZMA compression (check their apis).
	ReplayHeader()
		: type(0), version(0), flags(0), timestamp(0), size(0), hash(0), props{ 0 } {}
};

struct ExtendedReplayHeader
{
	static constexpr uint64_t CURRENT_VERSION = 1U;

	ReplayHeader base;
	uint64_t version; // Version of this extended header.
	uint64_t seed[4U]; // New 256bit seed.
};

// end from Multirole/YGOPro/Replay.cpp

using PlayerId = uint8_t;
using CardCode = uint32_t;
using CardId = uint16_t;

// B.3.a Chunk 3.5/3.6: constants used by SELECT_* handlers to decode
// the `desc` (description) field in MSG_SELECT_OPTION / MSG_SELECT_
// EFFECTYN. desc < DESCRIPTION_LIMIT → plain system-string index.
// desc ≥ DESCRIPTION_LIMIT → packed (card_code << 4 | effect_index)
// with effect_index ending up in [0, 14). CARD_EFFECT_OFFSET separates
// the packed-card-effect range from the system-string range in the
// LegalAction.effect_ field. Mirrors ygopro.h:1116-1117.
constexpr int DESCRIPTION_LIMIT = 10000;
constexpr int CARD_EFFECT_OFFSET = 10010;

struct loc_info {
	uint8_t controler;
	uint8_t location;
	uint32_t sequence;
	uint32_t position;
};

// Forward declaration for the B.5 synthetic test accessor — needs friend
// access to the Card protected fields without exposing them publicly.
struct CardSnapshot;
inline CardSnapshot test_card_struct(uint32_t code);

class Card {
  friend class EDOProEnv;
  friend CardSnapshot test_card_struct(uint32_t code);

protected:
  CardCode code_ = 0;
  uint32_t alias_;
  // B.5: setcodes are variable-length in edo9300 (common.h/cards.cdb).
  // db_query_card_data packs the u64 "setcode" column as 4 consecutive
  // u16 slots. Restore them here as a vector so the Card class can expose
  // the full list to the card-feature encoder + parity test. The 16-card
  // starter format uses at most 1 setcode per card, but the container
  // generalizes for when the training pool expands.
  std::vector<uint16_t> setcodes_;
  uint32_t type_;
  uint32_t level_;
  uint32_t lscale_;
  uint32_t rscale_;
  int32_t attack_;
  int32_t defense_;
  // B.5: race is a u64 because edo9300's QUERY_RACE emits a u64 field.
  // The legacy u32 was lossy for the RACE_CYBORG..RACE_GALAXY range (the
  // upper half of the low 32 bits) and won't accommodate future RACE_*
  // additions that overflow bit 31 at all. Yugi/Kaiba starter format
  // races all fit in low bits, so the widening is a type-correctness
  // change, not a behavior change.
  uint64_t race_;
  uint32_t attribute_;
  uint32_t link_marker_;
  // uint32_t category_;
  std::string name_;
  std::string desc_;
  std::vector<std::string> strings_;

  uint32_t data_ = 0;

  PlayerId controler_ = 0;
  uint32_t location_ = 0;
  uint32_t sequence_ = 0;
  uint32_t position_ = 0;
  uint32_t counter_ = 0;
  // Populated from QUERY_STATUS by the TLV decoder; consumed by
  // _set_obs_card_ (disabled/forbidden detection). Ported from the
  // ygopro Card struct during Phase B.1 of the edo9300 migration.
  uint32_t status_ = 0;
  // Populated from QUERY_OWNER (u8) by the TLV decoder. The "original
  // owner" seat (0 or 1), which differs from controler_ when a card has
  // been stolen (Change of Heart, Snatch Steal). Consumed by
  // _set_obs_card_ col 41 (owner_relative) during the Phase B.3.a encoder
  // port. QUERY_OWNER is already emitted by edo9300 (card.cpp:189); no
  // fork patch needed.
  uint8_t owner_ = 0;
  // Populated from QUERY_ATTACKED_COUNT (u8) by the TLV decoder. Number
  // of times this face-up monster has attacked this turn. Consumed by
  // _set_obs_card_ col 42 (attacked_count) during B.3.a encoder port.
  // QUERY_ATTACKED_COUNT is added in B.3.a Chunk 2a as a ~10-LOC patch
  // to edo9300 core (see repo/packages/e/edopro-core/xmake.lua).
  uint8_t attacked_count_ = 0;

public:
  Card() = default;

  Card(CardCode code, uint32_t alias, const std::vector<uint16_t> &setcodes,
       uint32_t type, uint32_t level, uint32_t lscale, uint32_t rscale,
       int32_t attack, int32_t defense, uint64_t race, uint32_t attribute,
       uint32_t link_marker,
       const std::string &name, const std::string &desc,
       const std::vector<std::string> &strings)
      : code_(code), alias_(alias), setcodes_(setcodes), type_(type),
        level_(level), lscale_(lscale), rscale_(rscale), attack_(attack),
        defense_(defense), race_(race), attribute_(attribute),
        link_marker_(link_marker), name_(name), desc_(desc), strings_(strings) {
  }

  ~Card() = default;

  void set_location(uint32_t location) {
    controler_ = location & 0xff;
    location_ = (location >> 8) & 0xff;
    sequence_ = (location >> 16) & 0xff;
    position_ = (location >> 24) & 0xff;
  }

  void set_location(const loc_info &info) {
    controler_ = info.controler;
    location_ = info.location;
    sequence_ = info.sequence;
    position_ = info.position;
  }

  const CardCode &code() const { return code_; }
  const std::string &name() const { return name_; }
  const std::string &desc() const { return desc_; }
  const uint32_t &type() const { return type_; }
  const uint32_t &level() const { return level_; }
  const std::vector<std::string> &strings() const { return strings_; }

  std::string get_spec(bool opponent) const {
    return ls_to_spec(location_, sequence_, position_, opponent);
  }

  std::string get_spec(PlayerId player) const {
    return get_spec(player != controler_);
  }

  uint32_t get_spec_code(PlayerId player) const {
    return ls_to_spec_code(location_, sequence_, position_,
                           player != controler_);
  }

  std::string get_position() const { return position_to_string(position_); }

  std::string get_effect_description(uint32_t desc,
                                     bool existing = false) const {
    std::string s;
    bool e = false;
    auto code = code_;
    if (desc > 10000) {
      code = desc >> 4;
    }
    uint32_t offset = desc - code_ * 16;
    bool in_range = (offset >= 0) && (offset < strings_.size());
    std::string str = "";
    if (in_range) {
      str = ltrim(strings_[offset]);
    }
    if (in_range || desc == 0) {
      if ((desc == 0) || str.empty()) {
        s = "Activate " + name_ + ".";
      } else {
        s = name_ + " (" + str + ")";
        e = true;
      }
    } else {
      s = get_system_string(desc);
      if (!s.empty()) {
        e = true;
      }
    }
    if (existing && !e) {
      s = "";
    }
    return s;
  }
};

inline std::string ls_to_spec(const loc_info &info, PlayerId player) {
  return ls_to_spec(info.location, info.sequence, info.position, player != info.controler);
}

inline uint32_t ls_to_spec_code(const loc_info &info, PlayerId player) {
  uint32_t c = player != info.controler ? 1 : 0;
  c |= (info.location << 8);
  c |= (info.sequence << 16);
  c |= (info.position << 24);
  return c;
}


// TODO: 7% performance loss
static std::shared_timed_mutex duel_mtx;

inline Card db_query_card(const SQLite::Database &db, CardCode code, bool may_absent = false) {
  SQLite::Statement query1(db, "SELECT * FROM datas WHERE id=?");
  query1.bind(1, code);
  bool found = query1.executeStep();
  if (!found) {
    if (may_absent) {
      return Card();
    }
    std::string msg = "[db_query_card] Card not found: " + std::to_string(code);
    throw std::runtime_error(msg);
  }

  uint32_t alias = query1.getColumn("alias");

  // B.5: setcodes are packed in the `setcode` column as four u16 slots
  // inside a u64. Extract the non-zero slots into a vector — matches the
  // db_query_card_data convention (which populates OCG_CardData.setcodes
  // for engine-side effect scripts).
  uint64_t setcode_packed =
      static_cast<uint64_t>(query1.getColumn("setcode").getInt64());
  std::vector<uint16_t> setcodes;
  for (int i = 0; i < 4; ++i) {
    uint16_t s = (setcode_packed >> (i * 16)) & 0xffff;
    if (s) setcodes.push_back(s);
  }

  uint32_t type = query1.getColumn("type");
  uint32_t level_ = query1.getColumn("level");
  uint32_t level = level_ & 0xff;
  uint32_t lscale = (level_ >> 24) & 0xff;
  uint32_t rscale = (level_ >> 16) & 0xff;
  int32_t attack = query1.getColumn("atk");
  int32_t defense = query1.getColumn("def");
  uint32_t link_marker = 0;
  if (type & TYPE_LINK) {
    defense = 0;
    link_marker = defense;
  }
  // B.5: race is stored as u64 in the datas.race column.
  uint64_t race =
      static_cast<uint64_t>(query1.getColumn("race").getInt64());
  uint32_t attribute = query1.getColumn("attribute");

  SQLite::Statement query2(db, "SELECT * FROM texts WHERE id=?");
  query2.bind(1, code);
  query2.executeStep();

  std::string name = query2.getColumn(1);
  std::string desc = query2.getColumn(2);
  std::vector<std::string> strings;
  for (int i = 3; i < query2.getColumnCount(); ++i) {
    std::string str = query2.getColumn(i);
    strings.push_back(str);
  }
  return Card(code, alias, setcodes, type, level, lscale, rscale, attack,
              defense, race, attribute, link_marker, name, desc, strings);
}

inline OCG_CardData db_query_card_data(
  const SQLite::Database &db, CardCode code, bool may_absent = false) {
  SQLite::Statement query(db, "SELECT * FROM datas WHERE id=?");
  query.bind(1, code);
  query.executeStep();
  OCG_CardData card;
  card.code = code;
  card.alias = query.getColumn("alias");
  uint64_t setcodes_ = query.getColumn("setcode").getInt64();

  std::vector<uint16_t> setcodes;
  for(int i = 0; i < 4; i++) {
    uint16_t setcode = (setcodes_ >> (i * 16)) & 0xffff;
    if (setcode) {
      setcodes.push_back(setcode);
    }
  }
  if (setcodes.size()) {
    setcodes.push_back(0);
    // memory leak here, but we only use it globally
    uint16_t* setcodes_p = new uint16_t[setcodes.size()];
    for (int i = 0; i < setcodes.size(); i++) {
      setcodes_p[i] = setcodes[i];
    }
    card.setcodes = setcodes_p;
  } else {
    card.setcodes = nullptr;
  }

  card.type = query.getColumn("type");
  card.attack = query.getColumn("atk");
  card.defense = query.getColumn("def");
  if (card.type & TYPE_LINK) {
    card.link_marker = card.defense;
    card.defense = 0;
  } else {
    card.link_marker = 0;
  }
  int level_ = query.getColumn("level");
  if (level_ < 0) {
    card.level = -(level_ & 0xff);
  }
  else {
    card.level = level_ & 0xff;
  }
  card.lscale = (level_ >> 24) & 0xff;
  card.rscale = (level_ >> 16) & 0xff;
  card.race = query.getColumn("race").getInt64();
  card.attribute = query.getColumn("attribute");
  return card;
}

struct card_script {
  const char *buf;
  int len;
};

static ankerl::unordered_dense::map<CardCode, Card> cards_;
static ankerl::unordered_dense::map<CardCode, CardId> card_ids_;
static ankerl::unordered_dense::map<CardCode, OCG_CardData> cards_data_;
static ankerl::unordered_dense::map<std::string, card_script> cards_script_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    main_decks_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    extra_decks_;
static std::vector<std::string> deck_names_;

// B.3.a Chunk 3.7: reverse map cid (1-based code_list index) → CardCode.
// Built once at end of init_module from card_ids_. Used by the state
// encoder to translate cids read from obs:cards_ back to konami codes
// for CardEmbeddingStore lookups.
static std::vector<CardCode> cid_to_code_;

inline const Card &c_get_card(CardCode code) { return cards_.at(code); }

inline CardId &c_get_card_id(CardCode code) { return card_ids_.at(code); }

inline void sort_extra_deck(std::vector<CardCode> &deck) {
  std::vector<CardCode> c;
  std::vector<std::pair<CardCode, int>> fusion, xyz, synchro, link;

  for (auto code : deck) {
    const Card &cc = c_get_card(code);
    if (cc.type() & TYPE_FUSION) {
      fusion.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_XYZ) {
      xyz.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_SYNCHRO) {
      synchro.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_LINK) {
      link.push_back({code, cc.level()});
    } else {
      throw std::runtime_error("Not extra deck card");
    }
  }

  auto cmp = [](const std::pair<CardCode, int> &a,
                const std::pair<CardCode, int> &b) {
    return a.second < b.second;
  };
  std::sort(fusion.begin(), fusion.end(), cmp);
  std::sort(xyz.begin(), xyz.end(), cmp);
  std::sort(synchro.begin(), synchro.end(), cmp);
  std::sort(link.begin(), link.end(), cmp);

  for (const auto &tc : fusion) {
    c.push_back(tc.first);
  }
  for (const auto &tc : xyz) {
    c.push_back(tc.first);
  }
  for (const auto &tc : synchro) {
    c.push_back(tc.first);
  }
  for (const auto &tc : link) {
    c.push_back(tc.first);
  }

  deck = c;
}

// Load one code's Card + OCG_CardData into the global lookup tables.
// Shared helper so preload_deck can also chase alias targets without
// re-implementing the body.
//
// `enforce_code_list` is true for directly-requested YDK codes (which
// must appear in code_list.txt so the cid↔konami mapping works) and
// false for alias-target codes chased transitively (which need to be
// in cards_data_ for g_DataReader but may legitimately be absent from
// code_list.txt).
inline bool preload_one_code(const SQLite::Database &db, CardCode code,
                             bool may_absent, bool enforce_code_list) {
  auto it = cards_.find(code);
  if (it == cards_.end()) {
    auto card = db_query_card(db, code, may_absent);
    if ((card.code() == 0) && may_absent) {
      fmt::println("[preload_deck] Card not found: {}", code);
      return false;
    }
    cards_[code] = card;
    if (enforce_code_list && card_ids_.find(code) == card_ids_.end()) {
      throw std::runtime_error("Card not found in code list: " +
                               std::to_string(code));
    }
  }

  auto it2 = cards_data_.find(code);
  if (it2 == cards_data_.end()) {
    cards_data_[code] = db_query_card_data(db, code, may_absent);
  }
  return true;
}

inline void preload_deck(const SQLite::Database &db,
                         const std::vector<CardCode> &deck,
                         bool may_absent = false) {
  for (const auto &code : deck) {
    if (!preload_one_code(db, code, may_absent, /*enforce_code_list=*/true)) {
      continue;
    }
    // Alias-target chase.
    //
    // cards.cdb encodes alt-art / re-print lineages via the `alias`
    // column: for alt-art code X with canonical code Y, X's alias=Y.
    // When the engine instantiates a card at X, edo9300's ocgcore
    // may (and in practice does) later call our g_DataReader with
    // the canonical code Y — e.g. Monster Reborn alt-art 83764719
    // resolves to canonical 83764718 during effect resolution. If
    // Y isn't populated in cards_data_, g_DataReader throws and the
    // async worker aborts the process.
    //
    // Found in Phase B.2 script-corpus audit (5-seed rollout, fired
    // reproducibly once Monster Reborn entered play). Fix: when
    // preloading code X, also populate cards_data_[alias_target] and
    // cards_[alias_target]. Alias targets are not required to appear
    // in code_list.txt (cid mapping is keyed off the YDK code).
    const auto &data = cards_data_[code];
    CardCode alias = data.alias;
    if (alias != 0 && alias != code) {
      preload_one_code(db, alias, /*may_absent=*/true,
                       /*enforce_code_list=*/false);
    }
  }
}

inline void g_DataReader(void* payload, uint32_t code, OCG_CardData* data) {
  auto it = cards_data_.find(code);
  if (it == cards_data_.end()) {
    throw std::runtime_error("[g_DataReader] Card not found: " + std::to_string(code));
  }
  *data = it->second;
}

static std::shared_timed_mutex scripts_mtx;

inline const char *read_card_script(const std::string &path, int *lenptr) {
  // edopro_script/c*.lua copied from ProjectIgnis/script/official
  auto full_path = "edopro_script/" + path;
  std::ifstream file(full_path, std::ios::binary);
  if (!file) {
    fmt::print("Unable to open script file: {}\n", full_path);
    *lenptr = 0;
    return nullptr;
  }
  file.seekg(0, std::ios::end);
  int len = file.tellg();
  file.seekg(0, std::ios::beg);
  const char *buf = new char[len];
  file.read((char *)buf, len);
  *lenptr = len;
  return buf;
}

inline int g_ScriptReader(void* payload, OCG_Duel duel, const char* name) {
  std::string path(name);
  std::shared_lock<std::shared_timed_mutex> lock(scripts_mtx);
  auto it = cards_script_.find(path);
  if (it == cards_script_.end()) {
    lock.unlock();
    int len;
    const char *buf = read_card_script(path, &len);
    std::unique_lock<std::shared_timed_mutex> ulock(scripts_mtx);
    cards_script_[path] = {buf, len};
    it = cards_script_.find(path);
  }
  int len = it->second.len;
  auto res = len && OCG_LoadScript(duel, it->second.buf, static_cast<uint32_t>(len), name);
  // if (!res) {
  //   fmt::print("Failed to load script: {}\n", path);
  // }
  return res;
}

void g_LogHandler(void* payload, const char* string, int type) {
  fmt::println("[LOG] type: {}, string: {}", type, string);
}

static void init_module(const std::string &db_path,
                        const std::string &code_list_file,
                        const std::map<std::string, std::string> &decks) {
  // parse code from code_list_file
  std::ifstream file(code_list_file);
  std::string line;
  int i = 0;
  while (std::getline(file, line)) {
    i++;
    CardCode code = std::stoul(line);
    card_ids_[code] = i;
  }

  // B.3.a Chunk 3.7: build reverse map cid → CardCode. cid is 1-based;
  // cid_to_code_[0] is unused. Matches ygopro.h:1546-1558 pattern.
  {
    CardId max_cid = 0;
    for (const auto& [code, cid] : card_ids_) {
      if (cid > max_cid) max_cid = cid;
    }
    cid_to_code_.assign(max_cid + 1, 0);
    for (const auto& [code, cid] : card_ids_) {
      cid_to_code_[cid] = code;
    }
  }

  SQLite::Database db(db_path, SQLite::OPEN_READONLY);

  for (const auto &[name, deck] : decks) {
    auto [main_deck, extra_deck, side_deck] = read_decks(deck);
    main_decks_[name] = main_deck;
    extra_decks_[name] = extra_deck;
    if (name[0] != '_') {
      deck_names_.push_back(name);
    }

    preload_deck(db, main_deck);
    preload_deck(db, extra_deck);
    preload_deck(db, side_deck, true);
  }

  for (auto &[name, deck] : extra_decks_) {
    sort_extra_deck(deck);
  }

}

// B.3.a Chunk 3.7: load the pre-baked card embedding + annotation store
// used by the state / events / actions encoders. Path points to the
// `EAIENC01` binary produced by `data_pipeline/bake_c_encoder_data.py`.
// Called once at training setup. Idempotent. Mirrors
// ygopro.h::init_embeddings_store (line 1407).
inline void init_embeddings_store(const std::string &path) {
  exodai::CardEmbeddingStore::load(path);
}

// from edopro/gframe/RNG/SplitMix64.hpp
class SplitMix64
{
public:
	using ResultType = uint64_t;
	using StateType = uint64_t;

	constexpr SplitMix64(StateType initialState) noexcept : s(initialState)
	{}

	ResultType operator()() noexcept
	{
		uint64_t z = (s += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}

	// NOTE: std::shuffle requires these.
	using result_type = ResultType;
	static constexpr ResultType min() noexcept { return ResultType(0U); }
	static constexpr ResultType max() noexcept { return ResultType(~ResultType(0U)); }
private:
	StateType s;
};


// ─── B.3.a synthetic test: YGO_NewCard owner/playerid semantics ───────────
//
// Validates that when YGO_NewCard is called with (code, owner=X, playerid=Y)
// where X != Y, the resulting card has pcard->owner == X (the original
// owner) and is placed in seat Y. Call-pattern the ygopro-agent callers
// expect; edopro.h YGO_NewCard maps it to edo9300's OCG_NewCardInfo
// (info.team=owner, info.con=playerid) per ocgapi.cpp:72 semantics.
//
// Must be called AFTER init_module so cards_data_ is populated for `code`.
// Returns the observed owner-seat from QUERY_OWNER; in a correctly-mapped
// wrapper equals the `owner` arg. If the mapping is ever reverted, this
// returns `playerid` instead and the Python test fails loudly.
inline int test_new_card_semantics(uint32_t code, uint8_t owner,
                                    uint8_t playerid) {
  static std::mutex test_mtx;
  std::lock_guard<std::mutex> lock(test_mtx);

  auto data_it = cards_data_.find(code);
  if (data_it == cards_data_.end()) {
    throw std::runtime_error(
        "test_new_card_semantics: code not in cards_data_ — "
        "call init_module first. code=" + std::to_string(code));
  }

  OCG_DuelOptions opts{};
  SplitMix64 generator(0xabcdef01u);
  for (int i = 0; i < 4; i++) opts.seed[i] = generator();
  opts.flags = 0;
  opts.team1 = {8000, 5, 1};
  opts.team2 = {8000, 5, 1};
  opts.cardReader = &g_DataReader;
  opts.payload1 = nullptr;
  opts.scriptReader = &g_ScriptReader;
  opts.payload2 = nullptr;
  opts.logHandler = [](void*, const char*, int) {};
  opts.payload3 = nullptr;
  opts.cardReaderDone = [](void*, OCG_CardData*) {};
  opts.payload4 = nullptr;
  opts.enableUnsafeLibraries = 1;

  OCG_Duel pduel = nullptr;
  int status = OCG_CreateDuel(&pduel, &opts);
  if (status != OCG_DUEL_CREATION_SUCCESS) {
    throw std::runtime_error("test_new_card_semantics: CreateDuel failed "
                             "(status=" + std::to_string(status) + ")");
  }

  // Place the card with asymmetric owner/playerid. Seat `playerid` is the
  // placement seat; `owner` is the pcard->owner assignment.
  OCG_NewCardInfo info{};
  info.team = owner;       // via B.3.a fix: maps to pcard->owner
  info.duelist = 0;
  info.code = code;
  info.con = playerid;     // via B.3.a fix: passed to add_card's seat arg
  info.loc = LOCATION_HAND;
  info.seq = 0;
  info.pos = POS_FACEUP;
  OCG_DuelNewCard(pduel, &info);

  // Query the card at the placement seat's hand[0]. QUERY_OWNER returns
  // pcard->owner as a u8 scalar TLV (card.cpp:189).
  OCG_QueryInfo qinfo{};
  qinfo.flags = QUERY_OWNER;
  qinfo.con = playerid;
  qinfo.loc = LOCATION_HAND;
  qinfo.seq = 0;
  qinfo.overlay_seq = 0;
  uint32_t length = 0;
  auto* buf = static_cast<uint8_t*>(OCG_DuelQuery(pduel, &length, &qinfo));
  int observed_owner = -1;
  if (buf != nullptr && length >= 7) {
    // Walk TLVs to find QUERY_OWNER explicitly rather than assume fixed
    // offsets — other tags may appear first in some configs.
    size_t p = 0;
    while (p + 6 <= length) {
      uint16_t size;
      uint32_t tag;
      std::memcpy(&size, buf + p, 2); p += 2;
      std::memcpy(&tag, buf + p, 4);  p += 4;
      const uint32_t value_size = size >= 4 ? size - 4 : 0;
      if (tag == QUERY_OWNER && value_size == 1) {
        observed_owner = buf[p];
        break;
      }
      if (tag == QUERY_END) break;
      p += value_size;
    }
  }

  OCG_DestroyDuel(pduel);
  return observed_owner;
}


// B.5 synthetic test accessor — returns the fields of the Card struct
// that init_module populated via db_query_card. Must be called AFTER
// init_module. Throws if `code` isn't in the `cards_` cache. Python
// caller compares against direct sqlite3 queries on cards.cdb to verify
// no field is truncated, mistyped, or missing after the B.5 widening.
struct CardSnapshot {
  uint32_t code;
  uint32_t alias;
  std::vector<uint16_t> setcodes;
  uint32_t type;
  uint32_t level;
  uint32_t lscale;
  uint32_t rscale;
  int32_t attack;
  int32_t defense;
  uint64_t race;
  uint32_t attribute;
  uint32_t link_marker;
  std::string name;
  std::string desc;
};

inline CardSnapshot test_card_struct(uint32_t code) {
  auto it = cards_.find(code);
  if (it == cards_.end()) {
    throw std::runtime_error(
        "test_card_struct: code not in cards_ — call init_module first. code=" +
        std::to_string(code));
  }
  const Card &c = it->second;
  CardSnapshot s;
  s.code = c.code_;
  s.alias = c.alias_;
  s.setcodes = c.setcodes_;
  s.type = c.type_;
  s.level = c.level_;
  s.lscale = c.lscale_;
  s.rscale = c.rscale_;
  s.attack = c.attack_;
  s.defense = c.defense_;
  s.race = c.race_;
  s.attribute = c.attribute_;
  s.link_marker = c.link_marker_;
  s.name = c.name_;
  s.desc = c.desc_;
  return s;
}


inline std::string getline() {
  char *line = nullptr;
  size_t len = 0;
  ssize_t read;

  read = getline(&line, &len, stdin);

  if (read != -1) {
    // Remove line ending character(s)
    if (line[read - 1] == '\n')
      line[read - 1] = '\0'; // Replace newline character with null terminator
    else if (line[read - 2] == '\r' && line[read - 1] == '\n') {
      line[read - 2] = '\0'; // Replace carriage return and newline characters
                             // with null terminator
      line[read - 1] = '\0';
    }

    std::string input(line);
    free(line);
    return input;
  } else {
    exit(0);
  }

  free(line);
  return "";
}

// ═══════════════════════════════════════════════════════════════════════════
// B.3.a Chunk 3.5/3.6 — LegalAction action model.
//
// Replaces the legacy `std::vector<std::string> options_` + string-based
// handler dispatch with a typed struct (mirrors ygopro.h:1119-1192). The
// obs:actions_ / obs:h_actions_ / obs:actions_encoded_ tensors are
// populated from LegalAction fields directly (no option-string reparse).
// The Player::think() interface consumes LegalAction too — HumanPlayer
// matches user input against spec_ (plus literals like "f"/"c"/"n" for
// finish/cancel).
// ═══════════════════════════════════════════════════════════════════════════

enum class ActionAct : uint8_t {
  None,
  Set,
  Repo,
  SpSummon,
  Summon,
  MSet,
  Attack,
  DirectAttack,
  Activate,
  Cancel,
};

enum class ActionPhase : uint8_t {
  None,
  Battle,
  Main2,
  End,
};

enum class ActionPlace : uint8_t {
  None,
  MZone1, MZone2, MZone3, MZone4, MZone5, MZone6, MZone7,
  SZone1, SZone2, SZone3, SZone4, SZone5, SZone6, SZone7, SZone8,
  OpMZone1, OpMZone2, OpMZone3, OpMZone4, OpMZone5, OpMZone6, OpMZone7,
  OpSZone1, OpSZone2, OpSZone3, OpSZone4, OpSZone5, OpSZone6, OpSZone7,
  OpSZone8,
};

class LegalAction {
public:
  std::string spec_ = "";
  ActionAct act_ = ActionAct::None;
  ActionPhase phase_ = ActionPhase::None;
  bool finish_ = false;
  uint8_t position_ = 0;
  int effect_ = -1;
  uint8_t number_ = 0;
  ActionPlace place_ = ActionPlace::None;
  uint8_t attribute_ = 0;

  int spec_index_ = 0;
  CardId cid_ = 0;
  int msg_ = 0;
  uint32_t response_ = 0;

  // Per-spec activation ordinal (0-based) for MSG_SELECT_IDLECMD activate
  // actions. Multiple LegalActions can share the same spec_ when a card has
  // multiple activatable effects — the obs encoder expands cmd_act2id['v']=7
  // into 7,8,9... via `+ act_offset_`. Unused (0) for all other msg/act types.
  uint8_t act_offset_ = 0;

  static LegalAction from_spec(const std::string &spec) {
    LegalAction la;
    la.spec_ = spec;
    return la;
  }

  static LegalAction act_spec(ActionAct act, const std::string &spec) {
    LegalAction la;
    la.act_ = act;
    la.spec_ = spec;
    return la;
  }

  static LegalAction finish() {
    LegalAction la;
    la.finish_ = true;
    return la;
  }

  static LegalAction cancel() {
    LegalAction la;
    la.act_ = ActionAct::Cancel;
    return la;
  }

  static LegalAction activate_spec(int effect_idx, const std::string &spec) {
    LegalAction la;
    la.act_ = ActionAct::Activate;
    la.effect_ = effect_idx;
    la.spec_ = spec;
    return la;
  }

  static LegalAction phase(ActionPhase phase) {
    LegalAction la;
    la.phase_ = phase;
    return la;
  }

  static LegalAction number(uint8_t number) {
    LegalAction la;
    la.number_ = number;
    return la;
  }

  static LegalAction place(ActionPlace place) {
    LegalAction la;
    la.place_ = place;
    return la;
  }

  static LegalAction attribute(int attribute) {
    LegalAction la;
    la.attribute_ = attribute;
    return la;
  }
};

class Player {
  friend class EDOProEnv;

protected:
  const std::string nickname_;
  const int init_lp_;
  const PlayerId duel_player_;
  const bool verbose_;

  bool seen_waiting_ = false;

public:
  Player(const std::string &nickname, int init_lp, PlayerId duel_player,
         bool verbose = false)
      : nickname_(nickname), init_lp_(init_lp), duel_player_(duel_player),
        verbose_(verbose) {}
  virtual ~Player() = default;

  void notify(const std::string &text) {
    if (verbose_) {
      fmt::println("{} {}", duel_player_, text);
    }
  }

  const int &init_lp() const { return init_lp_; }

  // B.3.a step-3: think() operates on LegalAction directly. HumanPlayer
  // matches user input against the spec_ field (and well-known literals
  // like "f"/"c" for finish/cancel). AI players ignore the argument.
  virtual int think(const std::vector<LegalAction> &las) = 0;
};

class GreedyAI : public Player {
protected:
public:
  GreedyAI(const std::string &nickname, int init_lp, PlayerId duel_player,
           bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &las) override { return 0; }
};

class RandomAI : public Player {
protected:
  std::mt19937 gen_;
  std::uniform_int_distribution<int> dist_;

public:
  RandomAI(int max_options, int seed, const std::string &nickname, int init_lp,
           PlayerId duel_player, bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose), gen_(seed),
        dist_(0, max_options - 1) {}

  int think(const std::vector<LegalAction> &las) override {
    return dist_(gen_) % las.size();
  }
};

class HumanPlayer : public Player {
protected:
public:
  HumanPlayer(const std::string &nickname, int init_lp, PlayerId duel_player,
              bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &las) override {
    while (true) {
      std::string input = getline();
      if (input == "quit") {
        exit(0);
      }
      // Match against spec_ first, then well-known shortcuts (f = finish,
      // c = CHAIN cancel / y/n = (E)YESNO). This is a debug-only interface
      // so the matching is intentionally simple — multi-effect activate
      // disambiguation isn't supported.
      for (size_t i = 0; i < las.size(); ++i) {
        const auto &la = las[i];
        if (la.finish_ && input == "f") return static_cast<int>(i);
        if (la.act_ == ActionAct::Cancel && la.msg_ == MSG_SELECT_CHAIN &&
            input == "c") return static_cast<int>(i);
        if (la.act_ == ActionAct::Cancel && (la.msg_ == MSG_SELECT_YESNO ||
            la.msg_ == MSG_SELECT_EFFECTYN) && input == "n")
          return static_cast<int>(i);
        if (!la.spec_.empty() && la.spec_ == input) return static_cast<int>(i);
      }
      std::string all;
      for (const auto &la : las) {
        if (!all.empty()) all += ", ";
        all += la.spec_.empty() ? "<non-spec>" : la.spec_;
      }
      fmt::println("{} Choose from [{}]", duel_player_, all);
    }
  }
};

// B.3.a Chunk 3.8: events ring-buffer infrastructure. Mirrors
// ygopro.h:857-953 / ADAPTER_EVENTS_DESIGN.md.

enum class EventType : uint8_t {
  TurnChange = 0,
  PhaseChange = 1,
  Draw = 2,
  CardMoved = 3,
  Chain = 4,
  Activate = 5,
  Summon = 6,
  Set = 7,
  Attack = 8,
};

constexpr uint8_t EVENT_LOC_NONE      = 0;
constexpr uint8_t EVENT_LOC_DECK      = 1;
constexpr uint8_t EVENT_LOC_HAND      = 2;
constexpr uint8_t EVENT_LOC_MONSTER   = 3;
constexpr uint8_t EVENT_LOC_SPELL     = 4;
constexpr uint8_t EVENT_LOC_GRAVE     = 5;
constexpr uint8_t EVENT_LOC_BANISHED  = 6;
constexpr uint8_t EVENT_LOC_EXTRA     = 7;
constexpr uint8_t EVENT_LOC_OVERLAY   = 8;
constexpr uint8_t EVENT_LOC_UNKNOWN   = 9;
constexpr uint8_t EVENT_LOC_PH_DRAW        = 10;
constexpr uint8_t EVENT_LOC_PH_STANDBY     = 11;
constexpr uint8_t EVENT_LOC_PH_MAIN1       = 12;
constexpr uint8_t EVENT_LOC_PH_BATTLE_START= 13;
constexpr uint8_t EVENT_LOC_PH_BATTLE_STEP = 14;
constexpr uint8_t EVENT_LOC_PH_DAMAGE      = 15;
constexpr uint8_t EVENT_LOC_PH_DAMAGE_CAL  = 16;
constexpr uint8_t EVENT_LOC_PH_BATTLE      = 17;
constexpr uint8_t EVENT_LOC_PH_MAIN2       = 18;
constexpr uint8_t EVENT_LOC_PH_END         = 19;

inline uint8_t edo_location_to_event_loc(uint32_t location) {
  uint32_t loc = location & 0x7f;
  bool overlay = (location & LOCATION_OVERLAY) != 0;
  if (overlay) return EVENT_LOC_OVERLAY;
  switch (loc) {
    case LOCATION_DECK:    return EVENT_LOC_DECK;
    case LOCATION_HAND:    return EVENT_LOC_HAND;
    case LOCATION_MZONE:   return EVENT_LOC_MONSTER;
    case LOCATION_SZONE:   return EVENT_LOC_SPELL;
    case LOCATION_GRAVE:   return EVENT_LOC_GRAVE;
    case LOCATION_REMOVED: return EVENT_LOC_BANISHED;
    case LOCATION_EXTRA:   return EVENT_LOC_EXTRA;
    default:               return EVENT_LOC_UNKNOWN;
  }
}

inline uint8_t edo_phase_to_event_loc(uint32_t phase) {
  switch (phase) {
    case PHASE_DRAW:         return EVENT_LOC_PH_DRAW;
    case PHASE_STANDBY:      return EVENT_LOC_PH_STANDBY;
    case PHASE_MAIN1:        return EVENT_LOC_PH_MAIN1;
    case PHASE_BATTLE_START: return EVENT_LOC_PH_BATTLE_START;
    case PHASE_BATTLE_STEP:  return EVENT_LOC_PH_BATTLE_STEP;
    case PHASE_DAMAGE:       return EVENT_LOC_PH_DAMAGE;
    case PHASE_DAMAGE_CAL:   return EVENT_LOC_PH_DAMAGE_CAL;
    case PHASE_BATTLE:       return EVENT_LOC_PH_BATTLE;
    case PHASE_MAIN2:        return EVENT_LOC_PH_MAIN2;
    case PHASE_END:          return EVENT_LOC_PH_END;
    default:                 return EVENT_LOC_NONE;
  }
}

struct EventRecord {
  uint8_t type;
  uint32_t code;
  uint8_t abs_player;
  uint8_t from_loc;
  uint8_t to_loc;
  uint8_t reason;
  uint8_t valid;
};

constexpr int kMaxEventsInRing = 64;

class EDOProEnvFns {
public:
  static decltype(auto) DefaultConfig() {
    // B.3.a Chunk 3.1: max_cards bumped 75→80 to match ygopro.h default.
    // max_multi_select kept — internal logic uses it (edopro.h:4180-4210).
    // The actions_/h_actions_ column count is decoupled from max_multi_select
    // in the spec below (deferred to sub-task 5 where _set_obs_action_* is
    // ported to ygopro.h's 12-col semantics atomically with the shape
    // change).
    return MakeDict("deck1"_.Bind(std::string("OldSchool")),
                    "deck2"_.Bind(std::string("OldSchool")), "player"_.Bind(-1),
                    "play_mode"_.Bind(std::string("bot")),
                    "verbose"_.Bind(false), "max_options"_.Bind(16),
                    "max_cards"_.Bind(80), "n_history_actions"_.Bind(16),
                    "max_multi_select"_.Bind(5), "record"_.Bind(false));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config &conf) {
    // B.3.a Chunk 3.5/3.6 step-4: ported to ygopro.h:1716-1790's 12-col
    // obs:actions_ / 14-col obs:h_actions_ layout.
    //
    // obs:actions_ columns (12):
    //   0  spec_index (1-based row in obs:cards_; 0 if no card)
    //   1  cid_hi     (CardId >> 8)
    //   2  cid_lo     (CardId & 0xff)
    //   3  msg_id     (msg2id output, 1-based)
    //   4  act        (ActionAct enum, raw)
    //   5  finish     (1 iff finish marker for SELECT_CARD/TRIBUTE/SUM/UNSELECT)
    //   6  effect     (0=None, 1=default, 2..15=card effect, 16+=system_string)
    //   7  phase      (ActionPhase enum, raw)
    //   8  position   (position2id output)
    //   9  number     (raw 1..12 for MSG_ANNOUNCE_NUMBER)
    //   10 place      (ActionPlace enum, raw)
    //   11 attrib     (attribute2id output)
    //
    // obs:h_actions_ extends to 14 cols: col 12 = turn_count_ snapshot
    // (transformed to turn_diff at read), col 13 = phase_to_id snapshot.
    int n_action_feats = 12;
    return MakeDict(
        "obs:cards_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 43})),
        "obs:global_"_.Bind(Spec<uint8_t>({23})),
        "obs:actions_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], n_action_feats})),
        "obs:h_actions_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], n_action_feats + 2})),
        "obs:mask_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 14})),
        "obs:events_"_.Bind(Spec<uint8_t>({64, 8})),
        "obs:state_"_.Bind(Spec<float>({10655})),
        "obs:events_encoded_"_.Bind(Spec<float>({64, 306})),
        "obs:event_mask_encoded_"_.Bind(Spec<float>({64})),
        "obs:actions_encoded_"_.Bind(Spec<float>({40, 745})),
        "obs:action_mask_encoded_"_.Bind(Spec<float>({40})),
        "info:num_options"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})),
        "info:to_play"_.Bind(Spec<int>({}, {0, 1})),
        "info:is_selfplay"_.Bind(Spec<int>({}, {0, 1})),
        "info:win_reason"_.Bind(Spec<int>({}, {-1, 1})),
        "info:step_time"_.Bind(Spec<double>({2})),
        "info:deck"_.Bind(Spec<int>({2})),
        "info:chain_cids"_.Bind(Spec<int>({8})),
        "info:n_events"_.Bind(Spec<int>({}, {0, 64})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config &conf) {
    return MakeDict(
        "action"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})));
  }
};

using EDOProEnvSpec = EnvSpec<EDOProEnvFns>;

enum PlayMode { kHuman, kSelfPlay, kRandomBot, kGreedyBot, kCount };

// parse play modes seperated by '+'
inline std::vector<PlayMode> parse_play_modes(const std::string &play_mode) {
  std::vector<PlayMode> modes;
  std::istringstream ss(play_mode);
  std::string token;
  while (std::getline(ss, token, '+')) {
    if (token == "human") {
      modes.push_back(kHuman);
    } else if (token == "self") {
      modes.push_back(kSelfPlay);
    } else if (token == "bot") {
      modes.push_back(kGreedyBot);
    } else if (token == "random") {
      modes.push_back(kRandomBot);
    } else {
      throw std::runtime_error("Unknown play mode: " + token);
    }
  }
  // human mode can't be combined with other modes
  if (std::find(modes.begin(), modes.end(), kHuman) != modes.end() &&
      modes.size() > 1) {
    throw std::runtime_error("Human mode can't be combined with other modes");
  }
  return modes;
}


// from edopro-deskbot/src/client.cpp#L46
constexpr uint64_t duel_options_ = DUEL_MODE_MR5;

class EDOProEnv : public Env<EDOProEnvSpec> {
  // B.3.a step-5 synthetic test harness needs access to the static
  // _set_obs_action helpers, which are otherwise private to EDOProEnv.
  friend std::vector<uint8_t> test_action_encoder();

protected:
  std::string deck1_;
  std::string deck2_;
  std::vector<uint32_t> main_deck0_;
  std::vector<uint32_t> main_deck1_;
  std::vector<uint32_t> extra_deck0_;
  std::vector<uint32_t> extra_deck1_;

  std::string deck_name_[2] = {"", ""};
  std::string nickname_[2] = {"Alice", "Bob"};

  const std::vector<PlayMode> play_modes_;

  // if play_mode_ == 'bot' or 'human', player_ is the order of the ai player
  // -1 means random, 0 and 1 means the first and second player respectively
  const int player_;

  PlayMode play_mode_;
  bool verbose_ = false;
  bool compat_mode_ = false;

  int max_episode_steps_, elapsed_step_;

  PlayerId ai_player_;

  OCG_Duel pduel_;
  Player *players_[2]; //  abstract class must be pointer

  std::uniform_int_distribution<uint64_t> dist_int_;
  bool done_{true};
  bool duel_started_{false};
  int duel_status_{OCG_DUEL_STATUS_CONTINUE};

  PlayerId winner_;
  uint8_t win_reason_;

  int lp_[2];

  // turn player
  PlayerId tp_;
  int current_phase_;
  int turn_count_;

  int msg_;
  // B.3.a Chunk 3.5/3.6: LegalAction action model replaces the legacy
  // `std::vector<std::string> options_` + string-based handler dispatch.
  // Handlers populate this; callback_(idx) reads legal_actions_[idx].response_.
  std::vector<LegalAction> legal_actions_;
  PlayerId to_play_;
  std::function<void(int)> callback_;

  uint8_t data_[4096];
  int dp_ = 0;
  int dl_ = 0;
  int fdl_ = 0;

  uint8_t query_buf_[16384];
  int qdp_ = 0;

  uint8_t resp_buf_[128];

  using IdleCardSpec = std::tuple<CardCode, std::string, uint32_t>;

  // chain
  PlayerId chaining_player_;

  double step_time_ = 0;
  uint64_t step_time_count_ = 0;

  double reset_time_ = 0;
  uint64_t reset_time_count_ = 0;

  const int n_history_actions_;

  // circular buffer for history actions of player 0
  TArray<uint8_t> history_actions_0_;
  int ha_p_0_ = 0;

  // circular buffer for history actions of player 1
  TArray<uint8_t> history_actions_1_;
  int ha_p_1_ = 0;

  std::vector<std::string> revealed_;

  // B.3.a Chunk 3.8: events ring buffer. Mirrors ygopro.h:1914-1918.
  std::array<EventRecord, kMaxEventsInRing> events_ring_{};
  int events_head_ = 0;
  int events_count_ = 0;

  // B.3.a Chunk 3.8: chain stack tracker for info:chain_cids. Populated
  // by MSG_CHAINING (push) and cleared by MSG_CHAIN_END.
  // Uses raw CardCode so chain_cids can translate to cid at WriteState.
  std::vector<CardCode> chain_stack_;

  // B.3.a Chunk 3.5/3.6 multi-select convergence. Sequenced multi-select
  // state — one policy decision per card pick, not per combination.
  // See src/docs/edopro_multiselect_convergence.md + edopro_serve_model
  // _multiselect_check.md for the rationale (deployment is sequenced,
  // training must match for priority-#1 parity).
  //
  // ms_idx_ = -1 means "no selection in progress". Set to 0 on entry
  // to init_multi_select, incremented on each pick, reset to -1 when
  // the selection completes or on env reset. Direct 1:1 mirror of
  // ygopro.h:1924-1932.
  int ms_idx_ = -1;
  int ms_mode_ = 0;
  int ms_min_ = 0;
  int ms_max_ = 0;
  int ms_must_ = 0;
  std::vector<std::string> ms_specs_;
  std::vector<std::vector<int>> ms_combs_;
  ankerl::unordered_dense::map<std::string, int> ms_spec2idx_;
  std::vector<int> ms_r_idxs_;

  // discard hand cards
  bool discard_hand_ = false;

  // replay
  bool record_ = false;
  FILE* fp_ = nullptr;
  bool is_recording = false;

public:
  EDOProEnv(const Spec &spec, int env_id)
      : Env<EDOProEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1), dist_int_(0, 0xffffffff),
        deck1_(spec.config["deck1"_]), deck2_(spec.config["deck2"_]),
        player_(spec.config["player"_]),
        play_modes_(parse_play_modes(spec.config["play_mode"_])),
        verbose_(spec.config["verbose"_]), record_(spec.config["record"_]),
        n_history_actions_(spec.config["n_history_actions"_]) {
    if (record_) {
      if (!verbose_) {
        throw std::runtime_error("record mode must be used with verbose mode and num_envs=1");
      }
    }

    int n_action_feats = spec.state_spec["obs:actions_"_].shape[1];
    // B.3.a step-4: history_actions_ is 14-col (n_action_feats + 2) per
    // ygopro.h:1985 — the extra 2 cols carry turn_count_ snapshot and
    // phase-id snapshot.
    history_actions_0_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
    history_actions_1_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
  }

  ~EDOProEnv() {
    for (int i = 0; i < 2; i++) {
      if (players_[i] != nullptr) {
        delete players_[i];
      }
    }
  }

  int max_options() const { return spec_.config["max_options"_]; }

  int max_cards() const { return spec_.config["max_cards"_]; }

  bool IsDone() override { return done_; }

  bool random_mode() const { return play_modes_.size() > 1; }

  bool self_play() const {
    return std::find(play_modes_.begin(), play_modes_.end(), kSelfPlay) !=
           play_modes_.end();
  }

  void Reset() override {
    // clock_t start = clock();
    if (random_mode()) {
      play_mode_ = play_modes_[dist_int_(gen_) % play_modes_.size()];
    } else {
      play_mode_ = play_modes_[0];
    }

    if (play_mode_ != kSelfPlay) {
      if (player_ == -1) {
        ai_player_ = dist_int_(gen_) % 2;
      } else {
        ai_player_ = player_;
      }
    }

    turn_count_ = 0;

    // B.3.a Chunk 3.8: clear events ring + chain stack on reset.
    for (auto& e : events_ring_) e = EventRecord{};
    events_head_ = 0;
    events_count_ = 0;
    chain_stack_.clear();

    // B.3.a Chunk 3.5/3.6: reset multi-select state. Any leftover
    // selection from a prior episode would break the main loop.
    ms_idx_ = -1;
    ms_specs_.clear();
    ms_combs_.clear();
    ms_spec2idx_.clear();
    ms_r_idxs_.clear();

    history_actions_0_.Zero();
    history_actions_1_.Zero();
    ha_p_0_ = 0;
    ha_p_1_ = 0;

    auto duel_seed = dist_int_(gen_);

    constexpr uint32_t init_lp = 8000;
    constexpr uint32_t startcount = 5;
    constexpr uint32_t drawcount = 1;

    std::unique_lock<std::shared_timed_mutex> ulock(duel_mtx);
    auto opts = YGO_CreateDuel(duel_seed, init_lp, startcount, drawcount);
    ulock.unlock();

    for (PlayerId i = 0; i < 2; i++) {
      if (players_[i] != nullptr) {
        delete players_[i];
      }
      std::string nickname = i == 0 ? "Alice" : "Bob";
      if (i == ai_player_) {
        nickname = "Agent";
      }
      nickname_[i] = nickname;
      if ((play_mode_ == kHuman) && (i != ai_player_)) {
        players_[i] = new HumanPlayer(nickname_[i], init_lp, i, verbose_);
      } else if (play_mode_ == kRandomBot) {
        players_[i] = new RandomAI(max_options(), dist_int_(gen_), nickname_[i],
                                   init_lp, i, verbose_);
      } else {
        players_[i] = new GreedyAI(nickname_[i], init_lp, i, verbose_);
      }
      load_deck(i);
      lp_[i] = players_[i]->init_lp_;
    }

    if (record_) {
      if (is_recording && fp_ != nullptr) {
        fclose(fp_);
      }
      auto time_str = time_now();
      // Use last 4 digits of seed as unique id
      auto seed_ = duel_seed % 10000;
      std::string fname;
      while (true) {
        fname = fmt::format("./replay/a{} {:04d}.yrp", time_str, seed_);
        // check existence
        if (std::filesystem::exists(fname)) {
          seed_ = (seed_ + 1) % 10000;
        } else {
          break;
        } 
      }
      fp_ = fopen(fname.c_str(), "wb");
      if (!fp_) {
        throw std::runtime_error("Failed to open file for replay: " + fname);
      }

      is_recording = true;

      ReplayHeader rh;
      rh.type = REPLAY_YRP1;
      rh.version = 0x000A0128;
      rh.flags = REPLAY_LUA64 | REPLAY_64BIT_DUELFLAG | REPLAY_NEWREPLAY | REPLAY_EXTENDED_HEADER;
      rh.timestamp = (uint32_t)time(nullptr);

      ExtendedReplayHeader erh;
      erh.base = rh;
      erh.version = 1U;
      for (int i = 0; i < 4; i++) {
        erh.seed[i] = opts.seed[i];
      }

      fwrite(&erh, sizeof(erh), 1, fp_);

      for (PlayerId i = 0; i < 2; i++) {
        uint16_t name[20];
        memset(name, 0, 40);
        std::string name_str = fmt::format("{} {}", nickname_[i], deck_name_[i]);
        if (name_str.size() > 20) {
          // truncate
          name_str = name_str.substr(0, 20);
        }
        fmt::println("name: {}", name_str);
        str_to_uint16(name_str.c_str(), name);
        ReplayWriteInt32(1);
        fwrite(name, 40, 1, fp_);
      }

      ReplayWriteInt32(init_lp);
      ReplayWriteInt32(startcount);
      ReplayWriteInt32(drawcount);
      ReplayWriteInt64(opts.flags);

      for (PlayerId i = 0; i < 2; i++) {
        auto &main_deck = i == 0 ? main_deck0_ : main_deck1_;
        auto &extra_deck = i == 0 ? extra_deck0_ : extra_deck1_;
        ReplayWriteInt32(main_deck.size());
        for (auto code : main_deck) {
          ReplayWriteInt32(code);
        }
        ReplayWriteInt32(extra_deck.size());
        for (int j = int(extra_deck.size()) - 1; j >= 0; --j) {
        // for (int j = 0; j < extra_deck.size(); ++j) {
          ReplayWriteInt32(extra_deck[j]);
        }
      }

      ReplayWriteInt32(0);

    }

    YGO_StartDuel(pduel_);
    duel_started_ = true;
    winner_ = 255;
    win_reason_ = 255;

    next();

    done_ = false;
    elapsed_step_ = 0;
    WriteState(0.0);

    // double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    // // update reset_time by moving average
    // reset_time_ = reset_time_* (static_cast<double>(reset_time_count_) /
    // (reset_time_count_ + 1)) + seconds / (reset_time_count_ + 1);
    // reset_time_count_++;
    // if (reset_time_count_ % 20 == 0) {
    //   fmt::println("Reset time: {:.3f}", reset_time_);
    // }
  }

  // B.3.a step-4: history_actions is 14-col (n_action_feats+2). Cols 0-11
  // come from _set_obs_action. Col 12 records turn_count_ at append time
  // (transformed to turn_diff at WriteState read). Col 13 records the
  // phase-id snapshot at append time. Mirrors ygopro.h:2359-2376.
  // Skip cancel actions — they carry no new information for the history.
  void update_history_actions(PlayerId player, int idx) {
    const auto &la = legal_actions_[idx];
    if (la.act_ == ActionAct::Cancel) {
      return;
    }
    auto &history_actions =
        player == 0 ? history_actions_0_ : history_actions_1_;
    auto &ha_p = player == 0 ? ha_p_0_ : ha_p_1_;

    ha_p--;
    if (ha_p < 0) {
      ha_p = n_history_actions_ - 1;
    }
    history_actions[ha_p].Zero();
    _set_obs_action(history_actions, ha_p, la);
    // Spec index isn't meaningful across turns (points into the current
    // obs:cards_ layout), so zero it in the history stream.
    history_actions[ha_p](kA_SPEC) = 0;
    history_actions[ha_p](12) = static_cast<uint8_t>(turn_count_);
    history_actions[ha_p](13) = static_cast<uint8_t>(phase2id.at(current_phase_));
  }

  void show_deck(const std::vector<CardCode> &deck, const std::string &prefix) const {
    fmt::print("{} deck: [", prefix);
    for (int i = 0; i < deck.size(); i++) {
      fmt::print(" '{}'", c_get_card(deck[i]).name());
    }
    fmt::print(" ]\n");
  }

  void show_turn() const {
    fmt::println("turn: {}, phase: {}, tplayer: {}", turn_count_, phase_to_string(current_phase_), tp_);
  }

  void show_deck(PlayerId player) const {
    fmt::print("Player {}'s deck:\n", player);
    show_deck(player == 0 ? main_deck0_ : main_deck1_, "Main");
    show_deck(player == 0 ? extra_deck0_ : extra_deck1_, "Extra");
  }

  // B.3.a Chunk 3.5/3.6: decode a MSG_SELECT_OPTION / MSG_SELECT_EFFECTYN
  // desc field into (card_code, effect_index). Mirrors ygopro.h:4204-4219.
  // desc < DESCRIPTION_LIMIT → system string, code=0, idx = desc.
  // desc ≥ DESCRIPTION_LIMIT → card effect, code = desc >> 4,
  //   idx = (desc & 0xf) + CARD_EFFECT_OFFSET.
  std::tuple<CardCode, int> unpack_desc(CardCode code, uint64_t desc) {
    if (desc < static_cast<uint64_t>(DESCRIPTION_LIMIT)) {
      return {0, static_cast<int>(desc)};
    }
    CardCode code_ = static_cast<CardCode>(desc >> 4);
    int idx = static_cast<int>(desc & 0xf);
    if (idx < 0 || idx >= 14) {
      fmt::print("Code: {}, Code_: {}, Desc: {}\n", code, code_, desc);
      throw std::runtime_error(
          fmt::format("Invalid effect index: {}", idx));
    }
    return {code_, idx + CARD_EFFECT_OFFSET};
  }

  // B.3.a step-4: debug dump of the 14-col history_actions buffer. Cols are
  // indexed per the step-4 layout (kA_* + timing cols 12/13).
  void show_history_actions(PlayerId player) const {
    const auto &ha = player == 0 ? history_actions_0_ : history_actions_1_;
    for (int i = 0; i < n_history_actions_; ++i) {
      uint8_t msg_id = uint8_t(ha(i, 3));  // kA_MSG = 3
      if (msg_id == 0) continue;
      int msg = _msgs[msg_id - 1];
      CardId cid = (static_cast<CardId>(ha(i, 1)) << 8) | ha(i, 2);
      fmt::print("history {} msg={} cid={} act={} finish={} effect={} phase={} "
                 "pos={} num={} place={} attr={} tdiff={} phase_snap={}\n",
                 i, msg_to_string(msg), cid, uint8_t(ha(i, 4)),
                 uint8_t(ha(i, 5)), uint8_t(ha(i, 6)), uint8_t(ha(i, 7)),
                 uint8_t(ha(i, 8)), uint8_t(ha(i, 9)), uint8_t(ha(i, 10)),
                 uint8_t(ha(i, 11)), uint8_t(ha(i, 12)), uint8_t(ha(i, 13)));
    }
  }

  void Step(const Action &action) override {
    // clock_t start = clock();

    int idx = action["action"_];
    callback_(idx);
    // update_history_actions(to_play_, idx);

    PlayerId player = to_play_;

    if (verbose_) {
      show_decision(idx);
    }

    next();

    float reward = 0;
    int reason = 0;
    if (done_) {
      float base_reward = 1.0;
      int win_turn = turn_count_ - winner_;
      if (win_turn <= 1) {
        base_reward = 8.0;
      } else if (win_turn <= 3) {
        base_reward = 4.0;
      } else if (win_turn <= 5) {
        base_reward = 2.0;
      } else {
        base_reward = 0.5 + 1.0 / (win_turn - 5);
      }
      if (play_mode_ == kSelfPlay) {
        // to_play_ is the previous player
        reward = winner_ == to_play_ ? base_reward : -base_reward;
      } else {
        reward = winner_ == ai_player_ ? base_reward : -base_reward;
      }

      if (win_reason_ == 0x01) {
        reason = 1;
      } else if (win_reason_ == 0x02) {
        reason = -1;
      }

      if (record_) {
        if (!is_recording || fp_ == nullptr) {
          throw std::runtime_error("Recording is not started");
        }
        fclose(fp_);
        is_recording = false;
      }
    }

    WriteState(reward, win_reason_);

    // double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    // // update step_time by moving average
    // step_time_ = step_time_* (static_cast<double>(step_time_count_) /
    // (step_time_count_ + 1)) + seconds / (step_time_count_ + 1);
    // step_time_count_++;
    // if (step_time_count_ % 500 == 0) {
    //   fmt::println("Step time: {:.3f}", step_time_);
    // }
  }

private:
  using SpecIndex = ankerl::unordered_dense::map<std::string, uint16_t>;

  // ════════════════════════════════════════════════════════════════════
  // B.3.a Chunk 3.5/3.6: sequenced multi-select infrastructure. Port of
  // ygopro.h:2185-2357 — init_multi_select, handle_multi_select,
  // _callback_multi_select, _callback_multi_select_2 + prepare/finish,
  // get_ms_spec_idx helper.
  //
  // Flow: a SELECT_CARD/TRIBUTE/SUM handler calls init_multi_select(),
  // which seeds legal_actions_ with the first sub-decision's options
  // and sets ms_idx_=0. The policy responds (callback fires
  // _callback_multi_select{,_2}). If more sub-picks needed,
  // handle_multi_select() rebuilds legal_actions_ for the next
  // sub-decision and the main loop re-enters via the ms_idx_ != -1
  // dispatch in next(). When the selection completes, _finish() sends
  // OCG_DuelSetResponse with the combined picks and sets ms_idx_ = -1.
  // ════════════════════════════════════════════════════════════════════

  void init_multi_select(
      int min, int max, int must, const std::vector<std::string> &specs,
      int mode = 0, const std::vector<std::vector<int>> &combs = {}) {
    // Dev-time assert: catch state-machine leaks where a handler is
    // entered while a prior selection is still pending.
    assert(ms_idx_ == -1 && "init_multi_select called while prior "
                            "selection still in progress");
    ms_idx_ = 0;
    ms_mode_ = mode;
    ms_min_ = min;
    ms_max_ = max;
    ms_must_ = must;
    ms_specs_ = specs;
    ms_r_idxs_.clear();
    ms_spec2idx_.clear();

    for (size_t j = 0; j < ms_specs_.size(); ++j) {
      ms_spec2idx_[ms_specs_[j]] = static_cast<int>(j);
    }

    if (ms_mode_ == 0) {
      for (size_t j = 0; j < ms_specs_.size(); ++j) {
        LegalAction la = LegalAction::from_spec(ms_specs_[j]);
        la.msg_ = msg_;
        legal_actions_.push_back(la);
      }
    } else {
      ms_combs_ = combs;
      _callback_multi_select_2_prepare();
    }
  }

  // Called from the main loop when ms_idx_ != -1 — advances one sub-
  // decision without reading the next engine message. Rebuilds
  // legal_actions_ for the next policy call, reassigns callback_.
  void handle_multi_select() {
    legal_actions_.clear();
    if (ms_mode_ == 0) {
      for (size_t j = 0; j < ms_specs_.size(); ++j) {
        if (ms_spec2idx_.find(ms_specs_[j]) != ms_spec2idx_.end()) {
          LegalAction la = LegalAction::from_spec(ms_specs_[j]);
          la.msg_ = msg_;
          legal_actions_.push_back(la);
        }
      }
      if (ms_idx_ == ms_max_ - 1) {
        if (ms_idx_ >= ms_min_) {
          LegalAction la = LegalAction::finish();
          la.msg_ = msg_;
          legal_actions_.push_back(la);
        }
        callback_ = [this](int idx) {
          _callback_multi_select(idx, true);
        };
      } else if (ms_idx_ >= ms_min_) {
        LegalAction la = LegalAction::finish();
        la.msg_ = msg_;
        legal_actions_.push_back(la);
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };
      } else {
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };
      }
    } else {
      _callback_multi_select_2_prepare();
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };
    }
  }

  int get_ms_spec_idx(const std::string &spec) const {
    auto it = ms_spec2idx_.find(spec);
    if (it != ms_spec2idx_.end()) {
      return it->second;
    }
    return -1;
  }

  // Mode-1+ prepare — rebuild legal_actions_ from the first elements
  // of remaining admissible combinations in ms_combs_. Mirrors
  // ygopro.h:2298-2307.
  void _callback_multi_select_2_prepare() {
    legal_actions_.clear();
    std::set<int> comb_heads;
    for (const auto &c : ms_combs_) {
      if (!c.empty()) comb_heads.insert(c[0]);
    }
    for (auto &i : comb_heads) {
      const auto &spec = ms_specs_[i];
      LegalAction la = LegalAction::from_spec(spec);
      la.msg_ = msg_;
      legal_actions_.push_back(la);
    }
  }

  // Build the combined picks array from ms_must_ (auto-included) +
  // ms_r_idxs_ (policy-picked), then emit the edo9300-compatible
  // response. edo9300 `parse_response_cards` (playerop.cpp:237) expects
  // a typed header: [u32 type][...]. We always use type=2 format:
  // [u32 type=2][u32 count][u8 indices...]. Non-compat only — the
  // compat_mode path (legacy Fluorohydride) is unused.
  void _emit_multi_select_response(const std::vector<int> &picks) {
    uint32_t type = 2;
    uint32_t count = static_cast<uint32_t>(picks.size());
    std::memcpy(resp_buf_ + 0, &type, sizeof(type));
    std::memcpy(resp_buf_ + 4, &count, sizeof(count));
    for (uint32_t i = 0; i < count; ++i) {
      resp_buf_[8 + i] = static_cast<uint8_t>(picks[i]);
    }
    YGO_SetResponseb(pduel_, resp_buf_, 8 + count);
  }

  // Mode-1+ finish — emit combined response with must-select cards
  // (indices 0..ms_must_-1) prepended to policy-picked ms_r_idxs_.
  // Mirrors ygopro.h:2309-2319, adapted to edo9300's typed-response
  // format.
  void _callback_multi_select_2_finish() {
    ms_idx_ = -1;
    std::vector<int> picks;
    picks.reserve(ms_must_ + ms_r_idxs_.size());
    for (int i = 0; i < ms_must_; ++i) picks.push_back(i);
    for (auto v : ms_r_idxs_) picks.push_back(v);
    _emit_multi_select_response(picks);
  }

  // Mode-1+ per-sub-pick handler. Removes combs whose first element
  // doesn't match the picked spec; when only one comb remains and is
  // empty, the selection is complete. Mirrors ygopro.h:2268-2296.
  void _callback_multi_select_2(int idx) {
    const auto &action = legal_actions_[idx];
    int spec_idx = get_ms_spec_idx(action.spec_);
    if (spec_idx == -1) {
      fmt::println("[_callback_multi_select_2] spec not found: '{}'",
                   action.spec_);
      // fall back: must-select cards only
      ms_idx_ = -1;
      std::vector<int> picks;
      for (int i = 0; i < ms_must_; ++i) picks.push_back(i);
      _emit_multi_select_response(picks);
      return;
    }
    ms_r_idxs_.push_back(spec_idx);
    std::vector<std::vector<int>> combs;
    for (auto &c : ms_combs_) {
      if (!c.empty() && c[0] == spec_idx) {
        c.erase(c.begin());
        if (c.empty()) {
          _callback_multi_select_2_finish();
          return;
        } else {
          combs.push_back(c);
        }
      }
    }
    ms_idx_++;
    ms_combs_ = combs;
  }

  // Mode-0 per-sub-pick handler. Records the pick in ms_r_idxs_,
  // removes it from the eligible set, and either finishes (emitting
  // the combined response via YGO_SetResponseb) or bumps ms_idx_ so
  // the main loop re-enters handle_multi_select() for the next
  // sub-decision. Mirrors ygopro.h:2321-2357.
  void _callback_multi_select(int idx, bool finish) {
    const auto &action = legal_actions_[idx];
    if (action.finish_) {
      finish = true;
    } else {
      int spec_idx = get_ms_spec_idx(action.spec_);
      if (spec_idx != -1) {
        ms_r_idxs_.push_back(spec_idx);
      } else {
        fmt::println("[_callback_multi_select] spec not found: '{}'",
                     action.spec_);
        // fall back: first ms_min_ indices
        ms_idx_ = -1;
        std::vector<int> picks;
        for (int i = 0; i < ms_min_; ++i) picks.push_back(i);
        _emit_multi_select_response(picks);
        return;
      }
    }
    if (finish) {
      ms_idx_ = -1;
      _emit_multi_select_response(ms_r_idxs_);
    } else {
      ms_idx_++;
      ms_spec2idx_.erase(action.spec_);
    }
  }

  // B.3.a Chunk 3.8: push a record into the events ring. code=0 for
  // player-level events (turn/phase change). Mirrors ygopro.h:2382.
  void push_event(EventType type, CardCode code, uint8_t abs_player,
                  uint8_t from_loc, uint8_t to_loc, uint8_t reason = 0) {
    const CardCode clean_code = code & 0x7fffffff;
    events_ring_[events_head_] = EventRecord{
      static_cast<uint8_t>(type),
      clean_code,
      abs_player,
      from_loc,
      to_loc,
      reason,
      1,
    };
    events_head_ = (events_head_ + 1) % kMaxEventsInRing;
    if (events_count_ < kMaxEventsInRing) {
      ++events_count_;
    }
  }

  // B.3.a Chunk 3.3: port _set_obs_cards to return {spec2index, loc_n_cards}
  // tuple mirroring ygopro.h:3307 ({SpecInfos, std::vector<int>}). The
  // loc_n_cards return is consumed by _set_obs_global to populate the per-
  // location card count features. Fixed-seat-offset scheme preserved
  // (player 0 rows 0..max_cards-1, player 1 rows max_cards..max_cards*2-1)
  // for compatibility with existing spec2index consumers; migrating to
  // ygopro's unified-offset is an unnecessary behavioral change at this
  // stage.
  std::tuple<SpecIndex, std::vector<int>>
  _set_obs_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    SpecIndex spec2index;
    std::vector<int> loc_n_cards;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      int offset = opponent ? spec_.config["max_cards"_] : 0;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        if (opponent && (location == LOCATION_HAND) &&
            (revealed_.size() != 0)) {
          hidden_for_opponent = false;
        }
        if (opponent && hidden_for_opponent) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          for (auto i = 0; i < n_cards; i++) {
            f_cards(offset, 2) = location2id.at(location);
            f_cards(offset, 4) = 1;
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = c.position_ & POS_FACEDOWN;
              if ((location == LOCATION_HAND) &&
                  (std::find(revealed_.begin(), revealed_.end(), spec) !=
                   revealed_.end())) {
                hide = false;
              }
            }
            _set_obs_card_(f_cards, offset, c, hide);
            offset++;
            spec2index[spec] = static_cast<uint16_t>(offset);
          }
        }
      }
    }
    return {spec2index, loc_n_cards};
  }

  // B.3.a Chunk 3.3: port of ygopro.h::_set_obs_g_cards (line 3361). The
  // "greedy-bot / global-vision" variant that writes ALL cards (including
  // opponent's hand) without the opponent-hiding logic. Used only when
  // play_mode == kGreedyBot (not currently a training target, but wired
  // for symmetry with ygopro). Does NOT populate spec2index — callers
  // that need spec→index look it up via _set_obs_mask.
  void _set_obs_g_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      std::vector<uint8_t> configs = {
          LOCATION_DECK,  LOCATION_HAND,    LOCATION_MZONE, LOCATION_SZONE,
          LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA,
      };
      for (auto location : configs) {
        std::vector<Card> cards = get_cards_in_location(player, location);
        int n_cards = cards.size();
        for (int i = 0; i < n_cards; ++i) {
          const auto &c = cards[i];
          _set_obs_card_(f_cards, offset, c, /*hide=*/false);
          offset++;
          if (offset == (spec_.config["max_cards"_] * 2 - 1)) {
            return;
          }
        }
      }
    }
  }

  // B.3.a Chunk 3.3: per-card write to obs:mask_. Mirror of
  // ygopro.h::_set_obs_mask_ (line 3522). Each column in obs:mask_ is 1
  // iff the corresponding obs:cards_ column holds valid (non-padded)
  // data; used by the transformer attention-pool to skip padded rows.
  void _set_obs_mask_(TArray<uint8_t> &mask, int offset, const Card &c,
                      bool hide, CardId card_id = 0) {
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      if (card_id != 0) {
        mask(offset, 0) = 1;
      }
    }
    mask(offset, 1) = 1;

    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      mask(offset, 2) = 1;
    }
    mask(offset, 3) = 1;
    if (overlay) {
      mask(offset, 4) = 1;
      mask(offset, 5) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND ||
          location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          mask(offset, 4) = 1;
        }
      } else {
        mask(offset, 4) = 1;
      }
    }
    if (!hide) {
      mask(offset, 6) = 1;
      mask(offset, 7) = 1;
      mask(offset, 8) = 1;
      mask(offset, 9) = 1;
      mask(offset, 10) = 1;
      mask(offset, 11) = 1;
      mask(offset, 12) = 1;
      mask(offset, 13) = 1;
    }
  }

  // B.3.a Chunk 3.3: port of ygopro.h::_set_obs_mask (line 3387). Same
  // loop shape as _set_obs_cards, same fixed-seat-offset scheme (so the
  // mask row indices align with cards_ row indices), but writes to
  // obs:mask_ via _set_obs_mask_ instead of obs:cards_ via _set_obs_card_.
  // Returns {spec2index, loc_n_cards} identical to _set_obs_cards —
  // WriteState chooses one or the other based on oppo_info config and
  // uses the returned spec2index for action encoding.
  std::tuple<SpecIndex, std::vector<int>>
  _set_obs_mask(TArray<uint8_t> &mask, PlayerId to_play) {
    SpecIndex spec2index;
    std::vector<int> loc_n_cards;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      int offset = opponent ? spec_.config["max_cards"_] : 0;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        if (opponent && (location == LOCATION_HAND) &&
            (revealed_.size() != 0)) {
          hidden_for_opponent = false;
        }
        if (opponent && hidden_for_opponent) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          for (auto i = 0; i < n_cards; i++) {
            mask(offset, 1) = 1;
            mask(offset, 3) = 1;
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = c.position_ & POS_FACEDOWN;
              if ((location == LOCATION_HAND) &&
                  (std::find(revealed_.begin(), revealed_.end(), spec) !=
                   revealed_.end())) {
                hide = false;
              }
            }
            CardId card_id = 0;
            if (!hide) {
              card_id = c_get_card_id(c.code_);
            }
            _set_obs_mask_(mask, offset, c, hide, card_id);
            offset++;
            spec2index[spec] = static_cast<uint16_t>(offset);
          }
        }
      }
    }
    return {spec2index, loc_n_cards};
  }

  // B.3.a Chunk 3.2: ported to ygopro.h's 43-col layout (see
  // ygopro.h:3442-3520). Column semantics per ygopro.h:1719-1739 comment.
  // Differences vs the legacy 40-col edopro layout:
  //   - col 11 is now disabled/forbidden (was atk-hi)
  //   - atk moved 11-12 → 12-13
  //   - def moved 13-14 → 14-15
  //   - type multi-hot moved 15-39 → 16-40
  //   - col 41 owner_relative — new, sourced from Card::owner_ (populated
  //     by QUERY_OWNER in the TLV decoder, no engine-pointer access)
  //   - col 42 attacked_count — new, sourced from Card::attacked_count_
  //     (populated by QUERY_ATTACKED_COUNT via our fork patch)
  void _set_obs_card_(TArray<uint8_t> &f_cards, int offset, const Card &c,
                      bool hide) {
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      auto card_id = c_get_card_id(c.code_);
      f_cards(offset, 0) = static_cast<uint8_t>(card_id >> 8);
      f_cards(offset, 1) = static_cast<uint8_t>(card_id & 0xff);
    }
    f_cards(offset, 2) = location2id.at(location);

    uint8_t seq = 0;
    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      seq = c.sequence_ + 1;
    }
    f_cards(offset, 3) = seq;
    f_cards(offset, 4) = (c.controler_ != to_play_) ? 1 : 0;
    if (overlay) {
      f_cards(offset, 5) = position2id.at(POS_FACEUP);
      f_cards(offset, 6) = 1;
    } else {
      f_cards(offset, 5) = position2id.at(c.position_);
    }
    if (!hide) {
      f_cards(offset, 7) = attribute2id.at(c.attribute_);
      f_cards(offset, 8) = race2id.at(c.race_);
      f_cards(offset, 9) = c.level_;
      f_cards(offset, 10) = std::min(c.counter_, static_cast<uint32_t>(15));
      f_cards(offset, 11) = static_cast<uint8_t>(
          (c.status_ & (STATUS_DISABLED | STATUS_FORBIDDEN)) != 0);
      auto [atk1, atk2] = float_transform(c.attack_);
      f_cards(offset, 12) = atk1;
      f_cards(offset, 13) = atk2;

      auto [def1, def2] = float_transform(c.defense_);
      f_cards(offset, 14) = def1;
      f_cards(offset, 15) = def2;

      auto type_ids = type_to_ids(c.type_);
      for (int j = 0; j < type_ids.size(); ++j) {
        f_cards(offset, 16 + j) = type_ids[j];
      }
    }

    // Cols 41-42: owner_relative + attacked_count. Mirrors ygopro.h:3507-
    // 3518 conditional (only MZONE/SZONE, non-hidden, non-overlay). Data
    // source is the TLV decoder's Card::owner_ / Card::attacked_count_
    // instead of Fluorohydride's engine-pointer access, which doesn't
    // port to edo9300's opaque-handle API. QUERY_OWNER is stock edo9300;
    // QUERY_ATTACKED_COUNT is added by our fork patch (see
    // fork-patches/0001-*.patch and repo/packages/e/edopro-core/xmake.lua).
    if (!hide && !overlay &&
        (location == LOCATION_MZONE || location == LOCATION_SZONE)) {
      uint8_t owner_seat = c.owner_;
      f_cards(offset, 41) = (owner_seat != to_play_) ? 1 : 0;
      f_cards(offset, 42) = c.attacked_count_;
    }
  }

  // B.3.a Chunk 3.4: ported to ygopro.h's 23-byte layout (ygopro.h:3570).
  //   0-3: me_lp (u16 split hi/lo), op_lp (u16 split)
  //   4:   turn_count clamped to 16 (was 8 in legacy edopro — aligned)
  //   5:   phase_id
  //   6:   is_player_0
  //   7:   is_turn_player
  //   8-21: loc_n_cards (DECK/HAND/MZONE/SZONE/GRAVE/REMOVED/EXTRA × 2)
  //   22:  n_options==0 marker (written by WriteState early-return path)
  void _set_obs_global(TArray<uint8_t> &feat, PlayerId player,
                       const std::vector<int> &loc_n_cards) {
    uint8_t me = player;
    uint8_t op = 1 - player;

    auto [me_lp_1, me_lp_2] = float_transform(lp_[me]);
    feat(0) = me_lp_1;
    feat(1) = me_lp_2;

    auto [op_lp_1, op_lp_2] = float_transform(lp_[op]);
    feat(2) = op_lp_1;
    feat(3) = op_lp_2;

    feat(4) = std::min(turn_count_, 16);
    feat(5) = phase2id.at(current_phase_);
    feat(6) = (me == 0) ? 1 : 0;
    feat(7) = (me == tp_) ? 1 : 0;

    for (size_t i = 0; i < loc_n_cards.size(); i++) {
      feat(8 + i) = static_cast<uint8_t>(std::min(loc_n_cards[i], 255));
    }
  }

  // ════════════════════════════════════════════════════════════════════
  // B.3.a Chunk 3.7: C++ state encoder (obs:state_) — mirrors
  // ygopro.h:2775-3075. Reads exclusively from obs:cards_, obs:global_,
  // and info:chain_cids; does NOT re-query the engine (the scoping pass
  // confirmed this path is pure-from-obs). Writes kStateDim=10655 float32
  // values total.
  //
  // State layout per ADAPTER_CPP_STATE_ENCODER_DESIGN.md §2. See ygopro.h
  // line 2775 for the byte-exact contract with state_encoder.py.
  // ════════════════════════════════════════════════════════════════════

  static constexpr int kStateDim = 10655;
  static constexpr int kEmbDim = 256;
  static constexpr int kAnnDim = 465;
  static constexpr int kMaxHandSize = 12;
  static constexpr int kMaxChainLength = 4;
  static constexpr int kMonsterSlotDim = 1 + kEmbDim + 7;   // 264
  static constexpr int kBackrowSlotDim = 1 + kEmbDim + 2;   // 259
  static constexpr int kHandSlotDim = 1 + kEmbDim;          // 257
  static constexpr int kPoolDim = 1 + kEmbDim;              // 257

  // obs:cards_ column indices (from ADAPTER_CPP_STATE_ENCODER_DESIGN.md §2,
  // mirrors ygopro.h:2817-2822).
  static constexpr int kC_CID_HI = 0, kC_CID_LO = 1;
  static constexpr int kC_LOC = 2, kC_SEQ = 3, kC_CTRL = 4, kC_POS = 5;
  static constexpr int kC_OVERLAY = 6;
  static constexpr int kC_ATK_HI = 12, kC_ATK_LO = 13;
  static constexpr int kC_DEF_HI = 14, kC_DEF_LO = 15;
  static constexpr int kC_OWNER = 41, kC_ATTACKED = 42;

  // obs:cards_ location_id values. edopro's make_ids(location2str, 1)
  // assigns these (same as ygopro's layout).
  static constexpr uint8_t kLOC_DECK = 1, kLOC_HAND = 2, kLOC_MZONE = 3;
  static constexpr uint8_t kLOC_SZONE = 4, kLOC_GRAVE = 5;
  static constexpr uint8_t kLOC_REMOVED = 6, kLOC_EXTRA = 7;

  // Position-id helpers matching ygopro's parity contract. Values are
  // position2id output (0..8) per edopro's make_ids(position2str). See
  // ygopro.h:2829-2845 for the Python-parity rationale.
  static bool _pos_id_is_faceup(uint8_t pos_id) {
    return pos_id == 2 || pos_id == 4 || pos_id == 5 || pos_id == 6;
  }

  static float _pos_id_to_scalar(uint8_t pos_id) {
    switch (pos_id) {
      case 2: return 1.0f / 5.0f;
      case 5: return 2.0f / 5.0f;
      case 7: return 3.0f / 5.0f;
      case 6: return 4.0f / 5.0f;
      case 8: return 5.0f / 5.0f;
      default: return 0.0f;
    }
  }

  static uint16_t _decode_u16(uint8_t hi, uint8_t lo) {
    return (static_cast<uint16_t>(hi) << 8) | lo;
  }

  // O(1) reverse lookup with bounds checking. Returns 0 for out-of-range
  // or unknown cids (which the embedding store handles as a zero-vector).
  static CardCode _cid_to_code(uint16_t cid) {
    if (cid_to_code_.empty() || cid == 0 ||
        static_cast<size_t>(cid) >= cid_to_code_.size()) {
      return 0;
    }
    return cid_to_code_[cid];
  }

  // Maps current MSG_* to a GameState decision_type index 0-6 matching
  // state_encoder.py (ygopro.h:2751 — preserves the SELECT_OPTION=0 quirk).
  inline int _decision_type_from_msg(int msg) const {
    switch (msg) {
      case MSG_SELECT_IDLECMD:
      case MSG_SELECT_TRIBUTE:
      case MSG_SELECT_SUM:
      case MSG_SELECT_UNSELECT_CARD:
        if (msg == MSG_SELECT_IDLECMD) return 0;
        return 2;
      case MSG_SELECT_BATTLECMD: return 1;
      case MSG_SELECT_CARD:      return 2;
      case MSG_SELECT_YESNO:     return 3;
      case MSG_SELECT_EFFECTYN:  return 4;
      case MSG_SELECT_CHAIN:     return 5;
      case MSG_SELECT_POSITION:  return 6;
      case MSG_SELECT_OPTION:    return 0;
      default:                   return 0;
    }
  }

  inline float* state_ptr(TArray<float>& t) {
    return reinterpret_cast<float*>(t.Data());
  }

  void _set_obs_state(TArray<float>& tstate,
                      const TArray<uint8_t>& cards_obs,
                      const TArray<uint8_t>& global_obs,
                      const TArray<int>& chain_cids,
                      int decision_type) {
    float* s = state_ptr(tstate);
    std::memset(s, 0, kStateDim * sizeof(float));
    if (!exodai::CardEmbeddingStore::is_loaded()) return;
    const auto& store = exodai::CardEmbeddingStore::get();
    int off = 0;

    // 1. Metadata (24 floats).
    s[off++] = _decode_u16(global_obs(0), global_obs(1)) / 8000.0f;
    s[off++] = _decode_u16(global_obs(2), global_obs(3)) / 8000.0f;
    s[off++] = static_cast<float>(global_obs(4)) / 30.0f;
    const int phase_id = static_cast<int>(global_obs(5));
    if (phase_id >= 1 && phase_id <= 10) s[off + phase_id - 1] = 1.0f;
    off += 10;
    s[off++] = (global_obs(7) == 1) ? 0.0f : 1.0f;
    if (decision_type >= 0 && decision_type < 7)
      s[off + decision_type] = 1.0f;
    off += 7;
    {
      int cc = 0;
      for (int i = 0; i < 8; ++i) {
        if (chain_cids(i) != 0) ++cc;
      }
      s[off++] = static_cast<float>(cc);
    }
    s[off++] = 0.0f;
    s[off++] = 0.0f;

    const int n_card_rows = cards_obs.Shape()[0];

    // 2. Bot zones (ctrl_rel = 0).
    off = _set_state_mzone_from_obs(s, off, cards_obs, n_card_rows, 0, store);
    off = _set_state_szone_from_obs(s, off, cards_obs, n_card_rows, 0, store);
    off = _set_state_hand_from_obs(s, off, cards_obs, n_card_rows, 0, store);
    off = _set_state_pool_from_obs(s, off, cards_obs, n_card_rows, 0, kLOC_GRAVE, store);
    off = _set_state_pool_from_obs(s, off, cards_obs, n_card_rows, 0, kLOC_REMOVED, store);
    off = _set_state_pool_from_obs(s, off, cards_obs, n_card_rows, 0, kLOC_DECK, store);

    // 3. Enemy zones (ctrl_rel = 1).
    off = _set_state_mzone_from_obs(s, off, cards_obs, n_card_rows, 1, store);
    off = _set_state_szone_from_obs(s, off, cards_obs, n_card_rows, 1, store);
    // Enemy hand + deck counts — read from obs:global_[15..16] (loc_n_cards
    // for player 1: DECK at [15], HAND at [16], per _set_obs_global layout).
    s[off++] = static_cast<float>(global_obs(16)) / 100.0f;  // op hand count
    s[off++] = static_cast<float>(global_obs(15)) / 100.0f;  // op deck count
    off = _set_state_pool_from_obs(s, off, cards_obs, n_card_rows, 1, kLOC_GRAVE, store);
    off = _set_state_pool_from_obs(s, off, cards_obs, n_card_rows, 1, kLOC_REMOVED, store);

    // 4. Chain.
    off = _set_state_chain_from_cids(s, off, chain_cids, store);

    if (off != kStateDim) {
      throw std::runtime_error(fmt::format(
          "_set_obs_state wrote {} floats, expected {}", off, kStateDim));
    }
  }

  int _set_state_mzone_from_obs(
      float* s, int off, const TArray<uint8_t>& cards,
      int n_rows, uint8_t ctrl_rel,
      const exodai::CardEmbeddingStore& store) {
    std::array<int, 5> slot_row{-1, -1, -1, -1, -1};
    for (int r = 0; r < n_rows; ++r) {
      if (cards(r, kC_LOC) != kLOC_MZONE) continue;
      if (cards(r, kC_CTRL) != ctrl_rel) continue;
      if (cards(r, kC_OVERLAY)) continue;
      int seq = static_cast<int>(cards(r, kC_SEQ));
      if (seq >= 1 && seq <= 5) slot_row[seq - 1] = r;
    }
    for (int i = 0; i < 5; ++i) {
      if (slot_row[i] < 0) { off += kMonsterSlotDim; continue; }
      const int r = slot_row[i];
      const uint16_t cid = _decode_u16(cards(r, kC_CID_HI), cards(r, kC_CID_LO));
      const uint8_t pos_id = cards(r, kC_POS);
      const bool face_up = _pos_id_is_faceup(pos_id);
      const bool base_card_visible = (cid > 0);

      s[off] = 1.0f;
      int slot_off = off + 1;
      if (base_card_visible) {
        if (face_up) {
          CardCode code = _cid_to_code(cid);
          const float* emb = store.embedding(code);
          if (emb) std::memcpy(&s[slot_off], emb, kEmbDim * sizeof(float));
        }
        slot_off += kEmbDim;
        s[slot_off + 0] = _decode_u16(cards(r, kC_ATK_HI), cards(r, kC_ATK_LO)) / 8000.0f;
        s[slot_off + 1] = _decode_u16(cards(r, kC_DEF_HI), cards(r, kC_DEF_LO)) / 8000.0f;
        s[slot_off + 2] = face_up ? 1.0f : 0.0f;
        s[slot_off + 3] = _pos_id_to_scalar(pos_id);
        s[slot_off + 4] = 0.0f;
        s[slot_off + 5] = (cards(r, kC_ATTACKED) > 0) ? 1.0f : 0.0f;
        s[slot_off + 6] = (cards(r, kC_OWNER) != cards(r, kC_CTRL)) ? 1.0f : 0.0f;
      }
      off += kMonsterSlotDim;
    }
    return off;
  }

  int _set_state_szone_from_obs(
      float* s, int off, const TArray<uint8_t>& cards,
      int n_rows, uint8_t ctrl_rel,
      const exodai::CardEmbeddingStore& store) {
    std::array<int, 5> slot_row{-1, -1, -1, -1, -1};
    for (int r = 0; r < n_rows; ++r) {
      if (cards(r, kC_LOC) != kLOC_SZONE) continue;
      if (cards(r, kC_CTRL) != ctrl_rel) continue;
      if (cards(r, kC_OVERLAY)) continue;
      int seq = static_cast<int>(cards(r, kC_SEQ));
      if (seq >= 1 && seq <= 5) slot_row[seq - 1] = r;
    }
    for (int i = 0; i < 5; ++i) {
      if (slot_row[i] < 0) { off += kBackrowSlotDim; continue; }
      const int r = slot_row[i];
      const uint16_t cid = _decode_u16(cards(r, kC_CID_HI), cards(r, kC_CID_LO));
      const uint8_t pos_id = cards(r, kC_POS);
      const bool face_up = _pos_id_is_faceup(pos_id);
      const bool base_card_visible = (cid > 0);

      s[off] = 1.0f;
      int slot_off = off + 1;
      if (base_card_visible) {
        if (face_up) {
          CardCode code = _cid_to_code(cid);
          const float* emb = store.embedding(code);
          if (emb) std::memcpy(&s[slot_off], emb, kEmbDim * sizeof(float));
        }
        slot_off += kEmbDim;
        s[slot_off + 0] = face_up ? 1.0f : 0.0f;
        s[slot_off + 1] = face_up ? 0.0f : 1.0f;
      }
      off += kBackrowSlotDim;
    }
    return off;
  }

  int _set_state_hand_from_obs(
      float* s, int off, const TArray<uint8_t>& cards,
      int n_rows, uint8_t ctrl_rel,
      const exodai::CardEmbeddingStore& store) {
    int hand_idx = 0;
    for (int r = 0; r < n_rows && hand_idx < kMaxHandSize; ++r) {
      if (cards(r, kC_LOC) != kLOC_HAND) continue;
      if (cards(r, kC_CTRL) != ctrl_rel) continue;
      const uint16_t cid = _decode_u16(cards(r, kC_CID_HI), cards(r, kC_CID_LO));
      s[off] = 1.0f;
      if (cid > 0) {
        CardCode code = _cid_to_code(cid);
        const float* emb = store.embedding(code);
        if (emb) std::memcpy(&s[off + 1], emb, kEmbDim * sizeof(float));
      }
      off += kHandSlotDim;
      ++hand_idx;
    }
    off += (kMaxHandSize - hand_idx) * kHandSlotDim;
    return off;
  }

  int _set_state_pool_from_obs(
      float* s, int off, const TArray<uint8_t>& cards,
      int n_rows, uint8_t ctrl_rel, uint8_t loc_id,
      const exodai::CardEmbeddingStore& store) {
    std::vector<exodai::CardEmbeddingStore::CardCode> codes;
    for (int r = 0; r < n_rows; ++r) {
      if (cards(r, kC_LOC) != loc_id) continue;
      if (cards(r, kC_CTRL) != ctrl_rel) continue;
      const uint16_t cid = _decode_u16(cards(r, kC_CID_HI), cards(r, kC_CID_LO));
      if (cid > 0) codes.push_back(_cid_to_code(cid));
      else codes.push_back(0);
    }
    s[off++] = static_cast<float>(codes.size()) / 100.0f;
    if (!codes.empty()) store.mean_embedding(codes, &s[off]);
    off += kEmbDim;
    return off;
  }

  // B.3.a Chunk 3.9: events encoder — per-event (306 floats) encoding
  // of the events ring buffer. Mirrors ygopro.h:3092-3152. Pure
  // transformation from events_ring_; event type one-hot (9) + card
  // embedding (256) + from-loc one-hot (20) + to-loc one-hot (20) +
  // player flag (1) = 306.
  static constexpr int kNumEventTypes = 9;
  static constexpr int kNumEventLocations = 20;
  static constexpr int kEventDim =
      kNumEventTypes + kEmbDim + 2 * kNumEventLocations + 1;  // = 306

  // ═════════════════════════════════════════════════════════════════════════
  // B.3.a step-4: action encoder. Mirrors ygopro.h:_set_obs_actions_encoded
  // (3228-3302). Converts the 12-col obs:actions_ uint8 tensor into a
  // (40, 745) float32 tensor matching the training-side adapter layout:
  //   [0..255]    card embedding (kEmbDim=256)
  //   [256..720]  annotation (kAnnDim=465)
  //   [721..733]  action type one-hot (kNumActionTypes=13)
  //   [734..740]  location one-hot (kNumLocations=7)
  //   [741..744]  extra context: [-4]=position/option, [-3]=chain_pass,
  //                               [-2]=yes/no, [-1]=phase marker
  // ═════════════════════════════════════════════════════════════════════════

  static constexpr int kActionDim = 745;
  static constexpr int kMaxActions = 40;
  static constexpr int kNumActionTypes = 13;
  static constexpr int kNumLocations = 7;

  // msg2id values (from _msgs, 1-based)
  static constexpr int kMSG_IDLECMD = 1, kMSG_CHAIN = 2;
  static constexpr int kMSG_CARD = 3, kMSG_TRIBUTE = 4;
  static constexpr int kMSG_POSITION = 5, kMSG_EFFECTYN = 6;
  static constexpr int kMSG_YESNO = 7, kMSG_BATTLECMD = 8;
  static constexpr int kMSG_UNSELECT = 9, kMSG_OPTION = 10;
  static constexpr int kMSG_SUM = 12;

  // ActionAct enum values
  static constexpr int kACT_NONE = 0, kACT_SET = 1, kACT_REPO = 2;
  static constexpr int kACT_SPSUMMON = 3, kACT_SUMMON = 4, kACT_MSET = 5;
  static constexpr int kACT_ATTACK = 6, kACT_DATTACK = 7;
  static constexpr int kACT_ACTIVATE = 8, kACT_CANCEL = 9;

  static int _action_type_index(int msg, int act) {
    if (msg == kMSG_IDLECMD) {
      switch (act) {
        case kACT_SUMMON:   return 0;
        case kACT_MSET:     return 1;
        case kACT_SPSUMMON: return 2;
        case kACT_ACTIVATE: return 3;
        case kACT_SET:      return 4;
        case kACT_REPO:     return 5;
        default:            return 7;  // phase_change
      }
    }
    if (msg == kMSG_BATTLECMD) {
      if (act == kACT_ATTACK || act == kACT_DATTACK) return 6;
      if (act == kACT_ACTIVATE) return 3;
      return 7;
    }
    if (msg == kMSG_CARD || msg == kMSG_TRIBUTE ||
        msg == kMSG_SUM || msg == kMSG_UNSELECT)
      return 8;
    if (msg == kMSG_CHAIN)    return 9;
    if (msg == kMSG_YESNO || msg == kMSG_EFFECTYN) return 10;
    if (msg == kMSG_POSITION) return 11;
    if (msg == kMSG_OPTION)   return 12;
    return -1;
  }

  // obs:cards_[r][kC_LOC] is 1-based location_id (1..7). Return 0-based
  // index into the 7-entry location one-hot.
  static int _loc_id_to_index(uint8_t loc_id) {
    if (loc_id >= 1 && loc_id <= 7) return loc_id - 1;
    return -1;
  }

  void _set_obs_actions_encoded(
      TArray<float>& tactions, TArray<float>& tmask,
      const TArray<uint8_t>& raw_actions,
      const TArray<uint8_t>& cards_obs,
      int n_options) {
    float* a = reinterpret_cast<float*>(tactions.Data());
    float* m = reinterpret_cast<float*>(tmask.Data());
    std::memset(a, 0, kMaxActions * kActionDim * sizeof(float));
    std::memset(m, 0, kMaxActions * sizeof(float));

    if (!exodai::CardEmbeddingStore::is_loaded()) return;
    const auto& store = exodai::CardEmbeddingStore::get();
    const int n = std::min(n_options, kMaxActions);
    const int n_card_rows = cards_obs.Shape()[0];

    const int ann_start = kEmbDim;                        // 256
    const int at_start  = kEmbDim + kAnnDim;              // 721
    const int loc_start = at_start + kNumActionTypes;     // 734

    for (int i = 0; i < n; ++i) {
      float* row = a + i * kActionDim;

      const uint16_t cid = _decode_u16(raw_actions(i, kA_CID_HI),
                                       raw_actions(i, kA_CID_LO));
      const int msg = static_cast<int>(raw_actions(i, kA_MSG));
      const int act = static_cast<int>(raw_actions(i, kA_ACT));
      const int spec_index = static_cast<int>(raw_actions(i, kA_SPEC));
      const int phase_marker = static_cast<int>(raw_actions(i, kA_PHASE));
      const int position_id = static_cast<int>(raw_actions(i, kA_POSITION));

      if (cid > 0) {
        const CardCode code = _cid_to_code(cid);
        if (code > 0) {
          const float* emb = store.embedding(code);
          if (emb) std::memcpy(row, emb, kEmbDim * sizeof(float));
          const float* ann = store.annotation(code);
          if (ann) std::memcpy(row + ann_start, ann, kAnnDim * sizeof(float));
        }
      }

      const int at_idx = _action_type_index(msg, act);
      if (at_idx >= 0 && at_idx < kNumActionTypes) {
        row[at_start + at_idx] = 1.0f;
      }

      if (spec_index > 0 && spec_index <= n_card_rows) {
        const uint8_t source_loc = cards_obs(spec_index - 1, kC_LOC);
        const int loc_idx = _loc_id_to_index(source_loc);
        if (loc_idx >= 0 && loc_idx < kNumLocations) {
          row[loc_start + loc_idx] = 1.0f;
        }
      }

      if (msg == kMSG_POSITION) {
        row[kActionDim - 4] = position_id / 10.0f;
      } else if (msg == kMSG_CHAIN && act == kACT_CANCEL) {
        row[kActionDim - 3] = 1.0f;
      } else if (msg == kMSG_YESNO || msg == kMSG_EFFECTYN) {
        row[kActionDim - 2] = (act == kACT_ACTIVATE) ? 1.0f : 0.0f;
      } else if ((msg == kMSG_IDLECMD || msg == kMSG_BATTLECMD) &&
                 act == kACT_NONE) {
        if (phase_marker == 1)      row[kActionDim - 1] = 0.33f;
        else if (phase_marker == 2) row[kActionDim - 1] = 0.66f;
        else if (phase_marker == 3) row[kActionDim - 1] = 1.0f;
      }

      m[i] = 1.0f;
    }
  }

  void _set_obs_events_encoded(TArray<float>& tevents, TArray<float>& tmask) {
    float* ev = reinterpret_cast<float*>(tevents.Data());
    float* mk = reinterpret_cast<float*>(tmask.Data());
    std::memset(ev, 0, kMaxEventsInRing * kEventDim * sizeof(float));
    std::memset(mk, 0, kMaxEventsInRing * sizeof(float));

    if (!exodai::CardEmbeddingStore::is_loaded()) return;
    const auto& store = exodai::CardEmbeddingStore::get();
    const int start = (events_head_ - events_count_ + kMaxEventsInRing)
                      % kMaxEventsInRing;
    for (int i = 0; i < events_count_; ++i) {
      const auto& e = events_ring_[(start + i) % kMaxEventsInRing];
      float* row = ev + i * kEventDim;
      int o = 0;

      if (e.type < kNumEventTypes) {
        row[o + e.type] = 1.0f;
      }
      o += kNumEventTypes;

      CardCode code = e.code;
      if (e.type == static_cast<uint8_t>(EventType::Draw) &&
          e.abs_player != to_play_) {
        code = 0;
      }
      if (code != 0) {
        const float* emb = store.embedding(code);
        if (emb != nullptr) {
          std::memcpy(row + o, emb, kEmbDim * sizeof(float));
        }
      }
      o += kEmbDim;

      if (e.from_loc < kNumEventLocations) {
        row[o + e.from_loc] = 1.0f;
      }
      o += kNumEventLocations;

      if (e.to_loc < kNumEventLocations) {
        row[o + e.to_loc] = 1.0f;
      }
      o += kNumEventLocations;

      row[o] = (e.abs_player != to_play_) ? 1.0f : 0.0f;
      mk[i] = 1.0f;
    }
  }

  int _set_state_chain_from_cids(
      float* s, int off, const TArray<int>& chain_cids,
      const exodai::CardEmbeddingStore& store) {
    int chain_count = 0;
    for (int i = 0; i < 8; ++i)
      if (chain_cids(i) != 0) ++chain_count;
    s[off++] = static_cast<float>(chain_count);
    s[off++] = -1.0f;
    for (int i = 0; i < kMaxChainLength; ++i) {
      if (i < chain_count) {
        const uint16_t cid = static_cast<uint16_t>(chain_cids(i));
        const CardCode code = _cid_to_code(cid);
        const float* emb = store.embedding(code);
        if (emb) std::memcpy(&s[off], emb, kEmbDim * sizeof(float));
        off += kEmbDim;
        s[off++] = 1.0f;
      } else {
        off += kEmbDim + 1;
      }
    }
    return off;
  }

  // B.3.a Chunk 3.5/3.6 step-4: obs:actions_ 12-col layout. Mirrors
  // ygopro.h:3614-3713. Per-column semantic documented on the
  // StateSpec declaration. Helpers populate one column each; the main
  // dispatcher `_set_obs_action` sets msg+cid unconditionally then
  // msg-specific columns per the handler's LegalAction fields.

  static constexpr int kA_SPEC = 0;
  static constexpr int kA_CID_HI = 1, kA_CID_LO = 2;
  static constexpr int kA_MSG = 3, kA_ACT = 4, kA_FINISH = 5;
  static constexpr int kA_EFFECT = 6, kA_PHASE = 7, kA_POSITION = 8;
  static constexpr int kA_NUMBER = 9, kA_PLACE = 10, kA_ATTRIB = 11;

  // All obs:actions_ col writers are static so the synthetic test harness
  // can exercise them without an EDOProEnv instance.
  static void _set_obs_action_spec(TArray<uint8_t> &feat, int i, int idx) {
    feat(i, kA_SPEC) = static_cast<uint8_t>(idx);
  }

  static void _set_obs_action_card_id(TArray<uint8_t> &feat, int i, CardId cid) {
    feat(i, kA_CID_HI) = static_cast<uint8_t>(cid >> 8);
    feat(i, kA_CID_LO) = static_cast<uint8_t>(cid & 0xff);
  }

  static void _set_obs_action_msg(TArray<uint8_t> &feat, int i, int msg) {
    feat(i, kA_MSG) = msg2id.at(msg);
  }

  static void _set_obs_action_act(TArray<uint8_t> &feat, int i, ActionAct act) {
    feat(i, kA_ACT) = static_cast<uint8_t>(act);
  }

  static void _set_obs_action_finish(TArray<uint8_t> &feat, int i) {
    feat(i, kA_FINISH) = 1;
  }

  // Canonical effect encoding (mirrors ygopro.h:3637-3652):
  //   -1                        → 0 (None)
  //    0                        → 1 (default)
  //   ≥ CARD_EFFECT_OFFSET      → 2..15 (card effect; idx = eff - offset + 2)
  //   otherwise (system string) → system_string_to_id(eff) ∈ [16, ...]
  static void _set_obs_action_effect(TArray<uint8_t> &feat, int i, int effect) {
    int encoded;
    if (effect == -1) {
      encoded = 0;
    } else if (effect == 0) {
      encoded = 1;
    } else if (effect >= CARD_EFFECT_OFFSET) {
      encoded = effect - CARD_EFFECT_OFFSET + 2;
    } else {
      encoded = system_string_to_id(effect);
    }
    feat(i, kA_EFFECT) = static_cast<uint8_t>(encoded);
  }

  static void _set_obs_action_phase(TArray<uint8_t> &feat, int i, ActionPhase phase) {
    feat(i, kA_PHASE) = static_cast<uint8_t>(phase);
  }

  static void _set_obs_action_position(TArray<uint8_t> &feat, int i, uint8_t position) {
    feat(i, kA_POSITION) = position2id.at(position);
  }

  static void _set_obs_action_number(TArray<uint8_t> &feat, int i, uint8_t number) {
    feat(i, kA_NUMBER) = number;
  }

  static void _set_obs_action_place(TArray<uint8_t> &feat, int i, ActionPlace place) {
    feat(i, kA_PLACE) = static_cast<uint8_t>(place);
  }

  static void _set_obs_action_attrib(TArray<uint8_t> &feat, int i, uint8_t attrib) {
    feat(i, kA_ATTRIB) = attribute2id.at(attrib);
  }

  static void _set_obs_action(TArray<uint8_t> &feat, int i, const LegalAction &la) {
    int msg = la.msg_;
    _set_obs_action_msg(feat, i, msg);
    _set_obs_action_card_id(feat, i, la.cid_);
    if (msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE ||
        msg == MSG_SELECT_SUM || msg == MSG_SELECT_UNSELECT_CARD) {
      if (la.finish_) {
        _set_obs_action_finish(feat, i);
      } else {
        _set_obs_action_spec(feat, i, la.spec_index_);
      }
    } else if (msg == MSG_SELECT_POSITION) {
      _set_obs_action_position(feat, i, la.position_);
    } else if (msg == MSG_SELECT_EFFECTYN) {
      _set_obs_action_spec(feat, i, la.spec_index_);
      _set_obs_action_act(feat, i, la.act_);
      _set_obs_action_effect(feat, i, la.effect_);
    } else if (msg == MSG_SELECT_YESNO || msg == MSG_SELECT_OPTION) {
      _set_obs_action_act(feat, i, la.act_);
      _set_obs_action_effect(feat, i, la.effect_);
    } else if (msg == MSG_SELECT_BATTLECMD || msg == MSG_SELECT_IDLECMD ||
               msg == MSG_SELECT_CHAIN) {
      _set_obs_action_phase(feat, i, la.phase_);
      _set_obs_action_spec(feat, i, la.spec_index_);
      _set_obs_action_act(feat, i, la.act_);
      _set_obs_action_effect(feat, i, la.effect_);
    } else if (msg == MSG_SELECT_PLACE || msg == MSG_SELECT_DISFIELD) {
      _set_obs_action_place(feat, i, la.place_);
    } else if (msg == MSG_ANNOUNCE_ATTRIB) {
      _set_obs_action_attrib(feat, i, la.attribute_);
    } else if (msg == MSG_ANNOUNCE_NUMBER) {
      _set_obs_action_number(feat, i, la.number_);
    } else {
      throw std::runtime_error("Unsupported message " + std::to_string(msg));
    }
  }

  CardId spec_to_card_id(const std::string &spec, PlayerId player) {
    int offset = 0;
    if (spec[0] == 'o') {
      player = 1 - player;
      offset++;
    }
    auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
    return card_ids_.at(get_card_code(player, loc, seq));
  }

  void _set_obs_actions(TArray<uint8_t> &feat,
                        const std::vector<LegalAction> &las) {
    for (int i = 0; i < las.size(); ++i) {
      _set_obs_action(feat, i, las[i]);
    }
  }


  void str_to_uint16(const char* src, uint16_t* dest) {
      for (int i = 0; i < strlen(src); i += 1) {
        dest[i] = src[i];
      }

      // Add null terminator
      dest[strlen(src) + 1] = '\0';
  }

  void ReplayWriteInt8(int8_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  void ReplayWriteInt32(int32_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  void ReplayWriteInt64(uint64_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  // edopro-core API
  // Thread-safety fix: edo9300's ocgcore — like the Fluorohydride lineage
  // — has hidden shared mutable state (the Lua-as-C++ global interpreter
  // state, script registration, RNG ctor sequences) that segfaults when
  // multiple duels are constructed or processed concurrently. The ygopro
  // wrapper hit this empirically with crashes at ~2400 games (num_threads=2)
  // and ~850 games (num_threads=16). B.3.a adds the same mutex pattern here
  // so num_threads=4 in B.7's apples-to-apples benchmark won't crash. Locked
  // paths: CreateDuel, NewCard, StartDuel, DestroyDuel, Process. Query and
  // SetResponse are left unlocked (per-duel state only).
  static std::mutex& core_mutex() {
    static std::mutex mtx;
    return mtx;
  }

  OCG_DuelOptions YGO_CreateDuel(uint32_t seed, uint32_t init_lp, uint32_t startcount, uint32_t drawcount) {
    std::lock_guard<std::mutex> lock(core_mutex());
    SplitMix64 generator(seed);
    OCG_DuelOptions opts;
    for (int i = 0; i < 4; i++) {
      opts.seed[i] = generator();
    }
    // from edopro-deskbot/src/client.cpp#L46
    opts.flags = duel_options_;
    opts.team1 = {init_lp, startcount, drawcount};
    opts.team2 = {init_lp, startcount, drawcount};
    opts.cardReader = &g_DataReader;
    opts.payload1 = nullptr;
    opts.scriptReader = &g_ScriptReader;
    opts.payload2 = nullptr;

		// opts.logHandler = [](void* /*payload*/, const char* /*string*/, int /*type*/) {};
    opts.logHandler = &g_LogHandler;
		opts.payload3 = nullptr;

		opts.cardReaderDone = [](void* /*payload*/, OCG_CardData* /*data*/) {};
		opts.payload4 = nullptr;

    opts.enableUnsafeLibraries = 1;
    int create_status = OCG_CreateDuel(&pduel_, &opts);
    if (create_status != OCG_DUEL_CREATION_SUCCESS) {
      throw std::runtime_error("Failed to create duel");
    }
    g_ScriptReader(nullptr, pduel_, "constant.lua");
    g_ScriptReader(nullptr, pduel_, "utility.lua");
    return opts;
  }

  void YGO_NewCard(OCG_Duel pduel, uint32_t code, uint8_t owner, uint8_t playerid, uint8_t location, uint8_t sequence, uint8_t position) {
    std::lock_guard<std::mutex> lock(core_mutex());
    // B.3.a fix (D3 from edopro_wrapper_sweep.md §1.1): edo9300's
    // OCG_NewCardInfo fields are `team` = original owner (written to
    // pcard->owner at ocgapi.cpp:72) and `con` = current controller
    // (placement seat, passed to game_field.add_card at ocgapi.cpp:73).
    // The Fluorohydride API our callers use expresses the same distinction
    // as `owner` (original) and `playerid` (placement). Map them by
    // semantic, not by field-order coincidence.
    OCG_NewCardInfo info;
    info.team = owner;
    info.duelist = 0;
    info.code = code;
    info.con = playerid;
    info.loc = location;
    info.seq = sequence;
    info.pos = position;
    OCG_DuelNewCard(pduel, &info);
  }

  void YGO_StartDuel(OCG_Duel pduel) {
    // Locked for the same reason as NewCard/Process: start_duel registers
    // continuous card effects (loads Lua scripts) and shuffles the deck,
    // both of which touch global state. Called once per game, so cost is
    // negligible.
    std::lock_guard<std::mutex> lock(core_mutex());
    OCG_StartDuel(pduel);
  }

  void YGO_EndDuel(OCG_Duel pduel) {
    std::lock_guard<std::mutex> lock(core_mutex());
    OCG_DestroyDuel(pduel);
  }

  uint32_t YGO_GetMessage(OCG_Duel pduel, uint8_t* buf) {
    uint32_t len;
    auto buf_ = OCG_DuelGetMessage(pduel, &len);
    memcpy(buf, buf_, len);
    return len;
  }

  int YGO_Process(OCG_Duel pduel) {
    std::lock_guard<std::mutex> lock(core_mutex());
    return OCG_DuelProcess(pduel);
  }

  int32_t YGO_QueryCard(OCG_Duel pduel, uint8_t playerid, uint8_t location, uint8_t sequence, uint32_t query_flag, uint8_t* buf) {
    // TODO: overlay
    OCG_QueryInfo info = {query_flag, playerid, location, sequence};
    uint32_t length;
    auto buf_ = static_cast<uint8_t*>(OCG_DuelQuery(pduel, &length, &info));
    if (length > 0) {
      memcpy(buf, buf_, length);      
    }
    return length;
  }

  int32_t YGO_QueryFieldCount(OCG_Duel pduel, uint8_t playerid, uint8_t location) {
    return OCG_DuelQueryCount(pduel, playerid, location);
  }

  int32_t OCG_QueryFieldCard(OCG_Duel pduel, uint8_t playerid, uint8_t location, uint32_t query_flag, uint8_t* buf, int32_t use_cache) {
    // TODO: overlay
    OCG_QueryInfo info = {query_flag, playerid, location};
    uint32_t length;
    auto buf_ = static_cast<uint8_t*>(OCG_DuelQueryLocation(pduel, &length, &info));
    if (length > 0) {
      memcpy(buf, buf_, length);      
    }
    return length;
  }

  void YGO_SetResponsei(OCG_Duel pduel, int32_t value) {
    if (record_) {
      ReplayWriteInt8(4);
      ReplayWriteInt32(value);
    }
    uint32_t len = sizeof(value);
    memcpy(resp_buf_, &value, len);
    OCG_DuelSetResponse(pduel, resp_buf_, len);
  }

  void YGO_SetResponseb(OCG_Duel pduel, uint8_t* buf, uint32_t len = 0) {
    if (record_) {
      if (len == 0) {
        // len = buf[0];
        // ReplayWriteInt8(len);
        // fwrite(buf + 1, len, 1, fp_);
        fwrite(buf, len, 1, fp_);
      } else {
        ReplayWriteInt8(len);
        fwrite(buf, len, 1, fp_);
      }
    }
    if (len == 0) {
      len = buf[0];
      OCG_DuelSetResponse(pduel, buf + 1, len);
    } else {
      OCG_DuelSetResponse(pduel, buf, len);
    }
  }

  // edopro-core API

  void WriteState(float reward, int win_reason = 0) {
    State state = Allocate();

    int n_options = legal_actions_.size();
    state["reward"_] = reward;
    state["info:to_play"_] = int(to_play_);
    state["info:is_selfplay"_] = int(play_mode_ == kSelfPlay);
    state["info:win_reason"_] = win_reason;

    if (n_options == 0) {
      state["info:num_options"_] = 1;
      // B.3.a Chunk 3.4: n_options==0 marker moved from global[8] (legacy
      // edopro) to global[22] (ygopro.h:2640). global[8..21] are now
      // loc_n_cards bytes.
      state["obs:global_"_][22] = uint8_t(1);
      return;
    }

    // B.3.a Chunk 3.3: _set_obs_cards now returns {spec2index, loc_n_cards}.
    // Also populate obs:mask_ so the transformer attention-pool can skip
    // padded rows.
    auto [spec2index, loc_n_cards] = _set_obs_cards(state["obs:cards_"_], to_play_);
    _set_obs_mask(state["obs:mask_"_], to_play_);

    _set_obs_global(state["obs:global_"_], to_play_, loc_n_cards);

    // B.3.a Chunk 3.8: populate info:chain_cids from chain_stack_. Stack
    // is updated on MSG_CHAINING push / MSG_CHAIN_END clear. Each entry
    // is translated from CardCode → cid via card_ids_.
    {
      const int max_chain = 8;
      const int n_chain = std::min<int>(chain_stack_.size(), max_chain);
      for (int i = 0; i < max_chain; ++i) {
        if (i < n_chain) {
          CardCode code = chain_stack_[i];
          auto it = card_ids_.find(code);
          state["info:chain_cids"_][i] =
              (it != card_ids_.end()) ? int(it->second) : 0;
        } else {
          state["info:chain_cids"_][i] = 0;
        }
      }
    }

    // B.3.a Chunk 3.8: serialize events ring to obs:events_ with player
    // relativization + info-hiding (opponent draws get cid=0). Mirrors
    // ygopro.h:2588-2625.
    {
      const int n_ev = std::min(events_count_, kMaxEventsInRing);
      for (int i = 0; i < kMaxEventsInRing; ++i) {
        for (int j = 0; j < 8; ++j) {
          state["obs:events_"_](i, j) = uint8_t(0);
        }
      }
      int start = (events_head_ - events_count_ + kMaxEventsInRing) %
                  kMaxEventsInRing;
      for (int i = 0; i < n_ev; ++i) {
        const auto &e = events_ring_[(start + i) % kMaxEventsInRing];
        uint8_t player_rel = (e.abs_player != to_play_) ? 1 : 0;
        uint16_t cid = 0;
        if (e.code != 0) {
          if (e.type == static_cast<uint8_t>(EventType::Draw) &&
              e.abs_player != to_play_) {
            cid = 0;
          } else {
            auto it = card_ids_.find(e.code);
            cid = (it != card_ids_.end()) ? it->second : 0;
          }
        }
        state["obs:events_"_](i, 0) = e.type;
        state["obs:events_"_](i, 1) = static_cast<uint8_t>(cid >> 8);
        state["obs:events_"_](i, 2) = static_cast<uint8_t>(cid & 0xff);
        state["obs:events_"_](i, 3) = player_rel;
        state["obs:events_"_](i, 4) = e.from_loc;
        state["obs:events_"_](i, 5) = e.to_loc;
        state["obs:events_"_](i, 6) = e.reason;
        state["obs:events_"_](i, 7) = e.valid;
      }
      state["info:n_events"_] = n_ev;
    }

    // B.3.a Chunk 3.7: C++ state encoder. Pure transformation of obs:cards_
    // + obs:global_ + info:chain_cids (3.8 now populates chain_cids). Wrap
    // in try/catch so a fault in encoder logic surfaces as a log message
    // instead of terminating the env worker; obs:state_ stays zeros on
    // fault.
    try {
      const int decision_type = _decision_type_from_msg(msg_);
      _set_obs_state(state["obs:state_"_], state["obs:cards_"_],
                     state["obs:global_"_], state["info:chain_cids"_],
                     decision_type);
      // B.3.a Chunk 3.9: events encoder. Pure transformation of
      // events_ring_ (populated by Chunk 3.8's push_event calls).
      _set_obs_events_encoded(state["obs:events_encoded_"_],
                              state["obs:event_mask_encoded_"_]);
    } catch (const std::exception& e) {
      fmt::println("[_set_obs_state/_events_encoded CRASH] {}", e.what());
    }

    // we can't shuffle because idx must be stable in callback
    if (n_options > max_options()) {
      legal_actions_.resize(max_options());
    }

    // B.3.a step-4: populate LegalAction.spec_index_ from the spec2index map
    // built by _set_obs_cards. mirrors ygopro.h:2692-2703. This happens
    // BEFORE _set_obs_actions so the 12-col obs encoder reads the right
    // card-row pointer for each action.
    n_options = legal_actions_.size();
    for (int i = 0; i < n_options; ++i) {
      auto &action = legal_actions_[i];
      action.msg_ = msg_;
      const auto &spec = action.spec_;
      if (!spec.empty()) {
        auto it = spec2index.find(spec);
        if (it != spec2index.end()) {
          action.spec_index_ = static_cast<int>(it->second);
          if (action.cid_ == 0 && action.spec_index_ > 0) {
            int row = action.spec_index_ - 1;
            uint16_t cid_hi =
                static_cast<uint16_t>(state["obs:cards_"_](row, 0));
            uint16_t cid_lo =
                static_cast<uint16_t>(state["obs:cards_"_](row, 1));
            action.cid_ = static_cast<CardId>((cid_hi << 8) | cid_lo);
          }
        }
      }
    }

    _set_obs_actions(state["obs:actions_"_], legal_actions_);

    state["info:num_options"_] = n_options;

    // write history actions — 14-col circular buffer, 12 action cols + the
    // two timing cols (turn_count snapshot + phase-id snapshot). Read it
    // out in chronological order starting at ha_p.
    const auto &ha_p = to_play_ == 0 ? ha_p_0_ : ha_p_1_;
    const auto &history_actions =
        to_play_ == 0 ? history_actions_0_ : history_actions_1_;
    int n1 = n_history_actions_ - ha_p;
    int n_h_action_feats = history_actions.Shape()[1];

    state["obs:h_actions_"_].Assign((uint8_t *)history_actions[ha_p].Data(),
                                    n_h_action_feats * n1);
    state["obs:h_actions_"_][n1].Assign((uint8_t *)history_actions.Data(),
                                        n_h_action_feats * ha_p);

    // Transform col 12 from raw turn_count_ snapshot → turn_diff (clamped
    // to 16). Skip empty rows (col 3 = msg_id is 0 if unused). Mirrors
    // ygopro.h:2720-2727.
    for (int i = 0; i < n_history_actions_; ++i) {
      if (uint8_t(state["obs:h_actions_"_](i, kA_MSG)) == 0) {
        break;
      }
      int turn_diff = std::min(16, turn_count_ -
                                       uint8_t(state["obs:h_actions_"_](i, 12)));
      state["obs:h_actions_"_](i, 12) = static_cast<uint8_t>(turn_diff);
    }

    // B.3.a step-4: action encoder — mirrors ygopro.h:_set_obs_actions_encoded
    // (3228-3302). Runs AFTER obs:actions_ has been written.
    try {
      _set_obs_actions_encoded(state["obs:actions_encoded_"_],
                               state["obs:action_mask_encoded_"_],
                               state["obs:actions_"_],
                               state["obs:cards_"_],
                               n_options);
    } catch (const std::exception& e) {
      fmt::println("[_set_obs_actions_encoded CRASH] {}", e.what());
    }
  }

  void show_decision(int idx) {
    const auto &picked = legal_actions_[idx];
    std::string picked_str = picked.spec_.empty() ? "<non-spec>" : picked.spec_;
    fmt::println("Player {} chose \"{}\" (msg={}, act={}, {} options)",
                 to_play_, picked_str, picked.msg_,
                 static_cast<int>(picked.act_), legal_actions_.size());
  }

  void load_deck(PlayerId player, bool shuffle = true) {
    std::string deck = player == 0 ? deck1_ : deck2_;
    std::vector<CardCode> &main_deck = player == 0 ? main_deck0_ : main_deck1_;
    std::vector<CardCode> &extra_deck =
        player == 0 ? extra_deck0_ : extra_deck1_;

    if (deck == "random") {
      // generate random deck name
      std::uniform_int_distribution<uint64_t> dist_int(0,
                                                       deck_names_.size() - 1);
      deck_name_[player] = deck_names_[dist_int(gen_)];
    } else {
      deck_name_[player] = deck;
    }
    deck = deck_name_[player];

    main_deck = main_decks_.at(deck);
    extra_deck = extra_decks_.at(deck);

    if (verbose_) {
      fmt::println("{} {}: {}, main({}), extra({})", player, nickname_[player],
        deck, main_deck.size(), extra_deck.size());
    }

    if (shuffle) {
      std::shuffle(main_deck.begin(), main_deck.end(), gen_);
    }

    // add main deck in reverse order following ygopro
    // but since we have shuffled deck, so just add in order

    for (int i = 0; i < main_deck.size(); i++) {
      YGO_NewCard(pduel_, main_deck[i], player, player, LOCATION_DECK, 0, POS_FACEDOWN_DEFENSE);
    }

    // TODO: check this for EDOPro
    // add extra deck in reverse order following ygopro
    for (int i = int(extra_deck.size()) - 1; i >= 0; --i) {
      YGO_NewCard(pduel_, extra_deck[i], player, player, LOCATION_EXTRA, 0, POS_FACEDOWN_DEFENSE);
    }
  }

  void next() {
    while (duel_started_) {
      if (duel_status_ == OCG_DUEL_STATUS_END) {
        break;
      }

      if (dp_ == fdl_ && ms_idx_ == -1) {
        duel_status_ = YGO_Process(pduel_);
        fdl_ = YGO_GetMessage(pduel_, data_);
        if (fdl_ == 0) {
          continue;
        }
        dp_ = 0;
      }
      // B.3.a Chunk 3.5/3.6: sequenced multi-select dispatch. When
      // ms_idx_ != -1 a prior SELECT_CARD/TRIBUTE/SUM handler left
      // selection state pending — advance one sub-decision via
      // handle_multi_select() instead of reading the next engine
      // message. Mirrors ygopro.h:3955-3957.
      while ((dp_ != fdl_) || (ms_idx_ != -1)) {
        if (ms_idx_ != -1) {
          handle_multi_select();
        } else {
          handle_message();
        }
        if (legal_actions_.empty()) {
          continue;
        }
        if ((play_mode_ == kSelfPlay) || (to_play_ == ai_player_)) {
          if (legal_actions_.size() == 1) {
            callback_(0);
            update_history_actions(to_play_, 0);
            if (verbose_) {
              show_decision(0);
            }
          } else {
            return;
          }
        } else {
          auto idx = players_[to_play_]->think(legal_actions_);
          callback_(idx);
          if (verbose_) {
            show_decision(idx);
          }
        }
      }
    }
    done_ = true;
    legal_actions_.clear();
  }

  uint8_t read_u8() { return data_[dp_++]; }

  uint16_t read_u16() {
    uint16_t v = *reinterpret_cast<uint16_t *>(data_ + dp_);
    dp_ += 2;
    return v;
  }

  uint32_t read_u32() {
    uint32_t v = *reinterpret_cast<uint32_t *>(data_ + dp_);
    dp_ += 4;
    return v;
  }

  uint32_t read_u64() {
    uint32_t v = *reinterpret_cast<uint64_t *>(data_ + dp_);
    dp_ += 8;
    return v;
  }

  template<typename T1, typename T2>
  T2 compat_read() {
    if(compat_mode_) {
      T1 v = *reinterpret_cast<T1 *>(data_ + dp_);
      dp_ += sizeof(T1);
      return static_cast<T2>(v);
    }
    T2 v = *reinterpret_cast<T2 *>(data_ + dp_);
    dp_ += sizeof(T2);
    return v;
  }

  uint32_t q_read_u8() {
    qdp_ += 6;
    uint8_t v = *reinterpret_cast<uint8_t *>(query_buf_ + qdp_);
    qdp_ += 1;
    return v;
  }

  uint32_t q_read_u16_() {
    uint32_t v = *reinterpret_cast<uint16_t *>(query_buf_ + qdp_);
    qdp_ += 2;
    return v;
  }

  uint32_t q_read_u16() {
    qdp_ += 6;
    uint32_t v = *reinterpret_cast<uint16_t *>(query_buf_ + qdp_);
    qdp_ += 2;
    return v;
  }

  uint32_t q_read_u32() {
    qdp_ += 6;
    uint32_t v = *reinterpret_cast<uint32_t *>(query_buf_ + qdp_);
    qdp_ += 4;
    return v;
  }

  uint32_t q_read_u32_() {
    uint32_t v = *reinterpret_cast<uint32_t *>(query_buf_ + qdp_);
    qdp_ += 4;
    return v;
  }

  // Raw readers for u8 / u64 value widths in the TLV stream. Phase B.1
  // of the edo9300 migration — see src/docs/edopro_envpool_spike.md §8
  // for wire format, and parse_one_card_tlv_ below for use.
  uint8_t q_read_u8_() {
    uint8_t v = *reinterpret_cast<uint8_t *>(query_buf_ + qdp_);
    qdp_ += 1;
    return v;
  }

  uint64_t q_read_u64_() {
    uint64_t v;
    std::memcpy(&v, query_buf_ + qdp_, sizeof(uint64_t));
    qdp_ += 8;
    return v;
  }

  // TLV decoder for one card's query-buffer TLV stream.
  //
  // Wire format (edo9300/ygopro-core card.cpp:111-208):
  //   Empty zone slot: [u16 size=0]                    (2 bytes)
  //   Real card:       one or more TLV records, terminated by QUERY_END:
  //                    [u16 size][u32 tag][size-4 bytes value]
  //     size = sizeof(u32) + sizeof(value), e.g.
  //       scalar u32 tag:       size=8  (tag=4 + value=4)
  //       scalar u8  tag:       size=5  (tag=4 + value=1)
  //       scalar u64 tag (RACE):size=12 (tag=4 + value=8)
  //       QUERY_END:            size=4  (tag only, no value)
  //       structured tags:      size varies; see card.cpp:135-205
  //
  // `c` is overwritten with a template fetched via c_get_card(code) on the
  // QUERY_CODE tag; subsequent tags mutate fields on `c`. For overlay
  // materials (XYZ mats emitted via QUERY_OVERLAY_CARD), if `overlay_out`
  // is non-null, one Card entry per material is pushed with controler=
  // `player`, location=`loc | LOCATION_OVERLAY`, sequence=0, position=i.
  //
  // Returns true if a card was decoded, false if the position was an empty
  // slot (first u16 == 0). qdp_ advances past the empty-slot marker or past
  // the QUERY_END TLV.
  //
  // Unknown tags throw std::runtime_error. Spike §8 guardrail:
  // "do not paper over unknown tags".
  bool parse_one_card_tlv_(Card &c,
                           std::vector<Card> *overlay_out,
                           PlayerId player, uint8_t loc) {
    const uint16_t first = q_read_u16_();
    if (first == 0) {
      return false;  // empty zone slot
    }
    // first was the first TLV's size field; rewind so the normal
    // loop re-reads it as [size][tag][value].
    qdp_ -= 2;

    bool got_code = false;

    while (true) {
      const uint16_t size = q_read_u16_();
      const uint32_t tag  = q_read_u32_();
      const size_t value_start = qdp_;
      const uint32_t value_size = size >= 4 ? size - 4 : 0;

      if (tag == QUERY_END) {
        // No value body. size == sizeof(uint32_t) == 4.
        break;
      }

      switch (tag) {
        case QUERY_CODE: {
          CardCode code = q_read_u32_();
          c = c_get_card(code);  // template from cards.cdb
          got_code = true;
          break;
        }
        case QUERY_POSITION:
          c.position_ = q_read_u32_() & 0xff;
          break;
        case QUERY_ALIAS:
          q_read_u32_();  // display-code override; not consumed here
          break;
        case QUERY_TYPE:
          c.type_ = q_read_u32_();
          break;
        case QUERY_LEVEL: {
          uint32_t v = q_read_u32_();
          if ((v & 0xff) > 0) c.level_ = v & 0xff;
          break;
        }
        case QUERY_RANK: {
          uint32_t v = q_read_u32_();
          if ((v & 0xff) > 0) c.level_ = v & 0xff;
          break;
        }
        case QUERY_ATTRIBUTE:
          c.attribute_ = q_read_u32_();
          break;
        case QUERY_RACE: {
          // B.5: c.race_ is now uint64_t so the full payload lands without
          // truncation. edo9300 emits u64 per common.h QUERY_RACE +
          // card.cpp:128.
          c.race_ = q_read_u64_();
          break;
        }
        case QUERY_ATTACK:
          c.attack_ = q_read_u32_();
          break;
        case QUERY_DEFENSE:
          c.defense_ = q_read_u32_();
          break;
        case QUERY_BASE_ATTACK:
        case QUERY_BASE_DEFENSE:
        case QUERY_REASON:
        case QUERY_COVER:
          q_read_u32_();  // not consumed by B.1 obs-write path
          break;
        case QUERY_REASON_CARD:
        case QUERY_EQUIP_CARD: {
          // [u8 con][u8 loc][u32 seq][u32 pos] = 10 bytes value body
          qdp_ += value_size;
          break;
        }
        case QUERY_TARGET_CARD: {
          // [u32 count][per target: u8 con, u8 loc, u32 seq, u32 pos]
          qdp_ += value_size;
          break;
        }
        case QUERY_OVERLAY_CARD: {
          // [u32 count][u32 × count: card codes]
          uint32_t count = q_read_u32_();
          for (uint32_t i = 0; i < count; ++i) {
            CardCode ovc = q_read_u32_();
            if (overlay_out != nullptr) {
              Card oc = c_get_card(ovc);
              oc.controler_ = player;
              oc.location_ = loc | LOCATION_OVERLAY;
              oc.sequence_ = 0;
              oc.position_ = i;
              overlay_out->push_back(std::move(oc));
            }
          }
          break;
        }
        case QUERY_COUNTERS: {
          // [u32 count][u32 × count: (counter_type) | (total << 16)]
          uint32_t count = q_read_u32_();
          for (uint32_t i = 0; i < count; ++i) {
            uint32_t rec = q_read_u32_();
            if (i == 0) c.counter_ = (rec >> 16) & 0xffff;
          }
          break;
        }
        case QUERY_OWNER:
          // B.3.a: populate Card::owner_ for downstream col 41
          // (owner_relative) in _set_obs_card_. u8 value body per
          // card.cpp:189 `CHECK_AND_INSERT_T(QUERY_OWNER, owner, uint8_t)`.
          c.owner_ = q_read_u8_();
          break;
        case QUERY_ATTACKED_COUNT:
          // B.3.a Chunk 2a: populate Card::attacked_count_ for downstream
          // col 42 (attacked_count) in _set_obs_card_. u8 value body per
          // the patched card.cpp:190 (see edopro-core xmake package).
          c.attacked_count_ = q_read_u8_();
          break;
        case QUERY_STATUS:
          c.status_ = q_read_u32_();
          break;
        case QUERY_IS_PUBLIC:
        case QUERY_IS_HIDDEN:
          q_read_u8_();  // u8, not consumed by B.1
          break;
        case QUERY_LSCALE:
          c.lscale_ = q_read_u32_();
          break;
        case QUERY_RSCALE:
          c.rscale_ = q_read_u32_();
          break;
        case QUERY_LINK: {
          // [u32 level][u32 marker]
          uint32_t level = q_read_u32_();
          uint32_t marker = q_read_u32_();
          if ((level & 0xff) > 0) c.level_ = level & 0xff;
          if (marker > 0) c.link_marker_ = marker;
          break;
        }
        default:
          throw std::runtime_error(fmt::format(
              "TLV decoder: unknown QUERY tag 0x{:x} "
              "(size={}, qdp_={}) — edo9300 upstream may have added "
              "a new tag, extend parse_one_card_tlv_.",
              tag, size, qdp_));
      }

      // Defensive: every tag branch must consume exactly value_size bytes.
      // Catches a case-missing-a-read bug immediately rather than letting
      // the loop walk off into garbage.
      if (qdp_ != value_start + value_size) {
        throw std::runtime_error(fmt::format(
            "TLV decoder: tag 0x{:x} read mismatch "
            "(qdp_={}, expected={}, value_size={})",
            tag, qdp_, value_start + value_size, value_size));
      }
    }

    return got_code;
  }

  CardCode get_card_code(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      throw std::runtime_error("[get_card_code] Invalid card");
    }
    return q_read_u32();
  }

  // Decode a single card via OCG_DuelQuery (singular). The flags set mirrors
  // what _set_obs_card_ downstream consumes. QUERY_LINK is requested because
  // downstream reads link_marker_ via get_card() for MZONE monsters.
  //
  // Phase B.1 rewrite: previous implementation assumed a flat fixed-offset
  // layout; current edo9300 emits TLVs. See parse_one_card_tlv_ above.
  Card get_card(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE | QUERY_POSITION | QUERY_LEVEL | QUERY_RANK |
                    QUERY_ATTACK | QUERY_DEFENSE | QUERY_LSCALE | QUERY_RSCALE |
                    QUERY_LINK;
    int32_t bl  = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      std::string err = fmt::format("Player: {}, loc: {}, seq: {}, length: {}", player, loc, seq, bl);
      throw std::runtime_error("[get_card] Invalid card " + err);
    }
    Card c;
    const bool got = parse_one_card_tlv_(c, /*overlay_out=*/nullptr, player, loc);
    if (!got) {
      throw std::runtime_error(fmt::format(
          "[get_card] unexpected empty-slot marker at player={} loc={} seq={}",
          player, loc, seq));
    }
    c.controler_ = player;
    c.location_  = loc;
    c.sequence_  = seq;
    return c;
  }

  // Decode every card in a given location via OCG_DuelQueryLocation.
  //
  // Wire format (edo9300/ygopro-core ocgapi.cpp:207-247):
  //   [u32 total_size][ per card: either [u16=0] empty-slot
  //                     or a TLV stream terminated by QUERY_END ]
  //
  // QUERY_* flags requested here cover every field consumed downstream by
  // _set_obs_card_ plus overlay materials (XYZ mats) and counters. Any
  // additional fields needed for the state encoder in Phase B.3.a will be
  // added to the flag set then.
  std::vector<Card> get_cards_in_location(PlayerId player, uint8_t loc) {
    int32_t flags = QUERY_CODE | QUERY_POSITION | QUERY_ALIAS | QUERY_TYPE |
                    QUERY_LEVEL | QUERY_RANK | QUERY_ATTRIBUTE | QUERY_RACE |
                    QUERY_ATTACK | QUERY_DEFENSE | QUERY_REASON |
                    QUERY_REASON_CARD | QUERY_EQUIP_CARD | QUERY_TARGET_CARD |
                    QUERY_OVERLAY_CARD | QUERY_COUNTERS | QUERY_OWNER |
                    QUERY_ATTACKED_COUNT | QUERY_STATUS | QUERY_LSCALE |
                    QUERY_RSCALE | QUERY_LINK;
    int32_t bl = OCG_QueryFieldCard(pduel_, player, loc, flags, query_buf_, 0);

    qdp_ = 4;  // skip the u32 total-size prefix
    std::vector<Card> cards;
    uint32_t seq = 0;  // sequence number within this location's list
    while (qdp_ < static_cast<size_t>(bl)) {
      Card c;
      std::vector<Card> overlay_cards;
      const bool got = parse_one_card_tlv_(c, &overlay_cards, player, loc);
      if (got) {
        c.controler_ = player;
        c.location_  = loc;
        c.sequence_  = seq;
        cards.push_back(std::move(c));
      }
      // Empty-slot markers are only emitted for fixed-layout zone lists
      // (MZONE / SZONE), where they DO consume a sequence slot. Dynamic
      // lists (HAND / GRAVE / DECK / etc.) never emit empty slots. Bump
      // seq in both cases so MZONE/SZONE sequences stay correct.
      for (Card &oc : overlay_cards) {
        cards.push_back(std::move(oc));
      }
      ++seq;
    }
    return cards;
  }

  std::vector<Card> read_cardlist(bool extra = false, bool extra8 = false) {
    std::vector<Card> cards;
    auto count = read_u8();
    cards.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto card = get_card(controller, loc, seq);
      if (extra) {
        if (extra8) {
          card.data_ = read_u8();
        } else {
          card.data_ = read_u32();
        }
      }
      cards.push_back(card);
    }
    return cards;
  }

  std::vector<IdleCardSpec> read_cardlist_spec(
    bool u32_seq = true, bool extra = false, bool extra8 = false) {
    std::vector<IdleCardSpec> card_specs;
    // TODO: different with ygopro-core
    auto count = compat_read<uint8_t, uint32_t>();
    card_specs.reserve(count);
    for (int i = 0; i < count; ++i) {
      CardCode code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      // TODO: different with ygopro-core
      uint32_t seq;
      if (u32_seq) {
        seq = compat_read<uint8_t, uint32_t>();;
      } else {
        seq = read_u8();
      }
      uint32_t data = -1;
      if (extra) {
        data = compat_read<uint32_t, uint64_t>();
        if (!compat_mode_) {
          // TODO: handle this
          read_u8();
        }
      }
      if (extra8) {
        read_u8();
      }
      card_specs.push_back({code, ls_to_spec(loc, seq, 0), data});
    }
    return card_specs;
  }

  loc_info read_loc_info() {
    loc_info info;
    info.controler = read_u8();
    info.location = read_u8();
    if (compat_mode_) {
      info.sequence = read_u8();
      info.position = read_u8();
    } else {
      info.sequence = read_u32();
      info.position = read_u32();
    }
    return info;
  }

  std::string cardlist_info_for_player(const Card &card, PlayerId pl) {
    std::string spec = card.get_spec(pl);
    if (card.location_ == LOCATION_DECK) {
      spec = "deck";
    }
    if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
      return position2str.at(card.position_) + "card (" + spec + ")";
    }
    return card.name_ + " (" + spec + ")";
  }

  void handle_message() {
    int l_ = read_u32();
    dl_ = dp_ + l_;
    msg_ = int(data_[dp_++]);
    legal_actions_.clear();

    if (verbose_) {
      fmt::println("Message {}, full {}, length {}, dp {}", msg_to_string(msg_), fdl_, dl_, dp_);
      // print byte by byte
      for (int i = dp_; i < dl_; ++i) {
        fmt::print("{:02x} ", data_[i]);
      }
      fmt::print("\n");
    }

    if (msg_ == MSG_DRAW) {
      // B.3.a Chunk 3.8: payload parity verified against edo9300
      // operations.cpp:478-484 — [u8 playerid][u32 drawn][for each:
      // u32 code, u32 position]. edopro's compat_read<u8,u32>() picks
      // u32 in non-compat mode (our setting), and dp_+=4 skips the
      // position field. Matches emit. Restructured so payload decode
      // + push_event fire in non-verbose mode too.
      auto player = read_u8();
      auto drawed = compat_read<uint8_t, uint32_t>();
      std::vector<uint32_t> codes;
      codes.reserve(drawed);
      for (uint32_t i = 0; i < drawed; ++i) {
        uint32_t code = read_u32();
        dp_ += 4;  // skip position
        codes.push_back(code & 0x7fffffff);
      }
      for (uint32_t code : codes) {
        push_event(EventType::Draw, code, player,
                   EVENT_LOC_DECK, EVENT_LOC_HAND);
      }
      if (!verbose_) {
        return;
      }
      const auto &pl = players_[player];
      pl->notify(fmt::format("Drew {} cards:", drawed));
      for (uint32_t i = 0; i < drawed; ++i) {
        const auto &c = c_get_card(codes[i]);
        pl->notify(fmt::format("{}: {}", i + 1, c.name_));
      }
      const auto &op = players_[1 - player];
      op->notify(fmt::format("Opponent drew {} cards.", drawed));
    } else if (msg_ == MSG_NEW_TURN) {
      // B.3.a Chunk 3.8: payload verified against edo9300
      // processor.cpp:3334-3335 — [u8 turn_player]. Match.
      tp_ = int(read_u8());
      turn_count_++;
      push_event(EventType::TurnChange, 0, static_cast<uint8_t>(tp_),
                 EVENT_LOC_NONE, EVENT_LOC_NONE);
      if (!verbose_) {
        return;
      }
      auto player = players_[tp_];
      player->notify("Your turn.");
      players_[1 - tp_]->notify(fmt::format("{}'s turn.", player->nickname_));
    } else if (msg_ == MSG_NEW_PHASE) {
      // B.3.a Chunk 3.8: payload verified against edo9300
      // processor.cpp:2788 and related — [u16 phase]. Match.
      current_phase_ = int(read_u16());
      push_event(EventType::PhaseChange, 0, static_cast<uint8_t>(tp_),
                 EVENT_LOC_NONE, edo_phase_to_event_loc(current_phase_));
      if (!verbose_) {
        return;
      }
      auto phase_str = phase_to_string(current_phase_);
      for (int i = 0; i < 2; ++i) {
        players_[i]->notify(fmt::format("Entering {} phase.", phase_str));
      }
    } else if (msg_ == MSG_MOVE) {
      // B.3.a 3.8 finish: payload parity verified against edo9300
      // operations.cpp:1008-1023 / field.cpp:461-482 / processor.cpp:5093.
      // Canonical emit: [u32 code][loc_info old 10B][loc_info new 10B][u32 reason].
      // edopro's decode matches byte-for-byte in non-compat mode.
      CardCode code = read_u32();
      loc_info location = read_loc_info();
      loc_info newloc = read_loc_info();
      uint32_t reason = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      Card cnew = c_get_card(code);
      cnew.set_location(newloc);
      push_event(EventType::CardMoved, code, card.controler_,
                 edo_location_to_event_loc(location.location),
                 edo_location_to_event_loc(newloc.location),
                 static_cast<uint8_t>(reason & 0xff));
      if (!verbose_) {
        return;
      }
      auto pl = players_[card.controler_];
      auto op = players_[1 - card.controler_];

      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto plnewspec = cnew.get_spec(false);
      auto opnewspec = cnew.get_spec(true);

      auto getspec = [&](Player *p) { return p == pl ? plspec : opspec; };
      auto getnewspec = [&](Player *p) {
        return p == pl ? plnewspec : opnewspec;
      };
      bool card_visible = true;
      if ((card.position_ & POS_FACEDOWN) && (cnew.position_ & POS_FACEDOWN)) {
        card_visible = false;
      }
      auto getvisiblename = [&](Player *p) {
        return card_visible ? card.name_ : "Face-down card";
      };

      if (card.location_ != cnew.location_) {
        if (reason & REASON_DESTROY) {
          pl->notify(fmt::format("Card {} ({}) destroyed.", plspec, card.name_));
          op->notify(fmt::format("Card {} ({}) destroyed.", opspec, card.name_));
        } else if (cnew.location_ == LOCATION_REMOVED) {
          pl->notify(
              fmt::format("Your card {} ({}) was banished.", plspec, card.name_));
          op->notify(fmt::format("{}'s card {} ({}) was banished.", pl->nickname_,
                                opspec, getvisiblename(op)));
          }
      } else if ((card.location_ == cnew.location_) &&
                 (card.location_ & LOCATION_ONFIELD)) {
        if (card.controler_ != cnew.controler_) {
          pl->notify(
              fmt::format("Your card {} ({}) changed controller to {} and is "
                          "now located at {}.",
                          plspec, card.name_, op->nickname_, plnewspec));
          op->notify(
              fmt::format("You now control {}'s card {} ({}) and it's located "
                          "at {}.",
                          pl->nickname_, opspec, card.name_, opnewspec));
        } else {
          pl->notify(fmt::format("Your card {} ({}) switched its zone to {}.",
                                 plspec, card.name_, plnewspec));
          op->notify(fmt::format("{}'s card {} ({}) switched its zone to {}.",
                                 pl->nickname_, opspec, card.name_, opnewspec));
        }
      } else if ((reason & REASON_DISCARD) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You discarded {} ({})", plspec, card.name_));
        op->notify(fmt::format("{} discarded {} ({})", pl->nickname_, opspec,
                               card.name_));
      } else if ((card.location_ == LOCATION_REMOVED) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your banished card {} ({}) returns to the field at "
                        "{}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s banished card {} ({}) returns to the field at "
                        "{}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((card.location_ == LOCATION_GRAVE) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((cnew.location_ == LOCATION_HAND) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(
            fmt::format("Card {} ({}) returned to hand.", plspec, card.name_));
      } else if ((reason & (REASON_RELEASE | REASON_SUMMON)) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You tribute {} ({}).", plspec, card.name_));
        op->notify(fmt::format("{} tributes {} ({}).", pl->nickname_, opspec,
                               getvisiblename(op)));
      } else if ((card.location_ == (LOCATION_OVERLAY | LOCATION_MZONE)) &&
                 (cnew.location_ & LOCATION_GRAVE)) {
        pl->notify(fmt::format("You detached {}.", card.name_));
        op->notify(fmt::format("{} detached {}.", pl->nickname_, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_GRAVE)) {
        pl->notify(fmt::format("Your card {} ({}) was sent to the graveyard.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was sent to the graveyard.",
                               pl->nickname_, opspec, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_DECK)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your deck.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) returned to their deck.",
                               pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_EXTRA)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your extra deck.",
                               plspec, card.name_));
        op->notify(
            fmt::format("{}'s card {} ({}) returned to their extra deck.",
                        pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ == LOCATION_DECK) &&
                 (cnew.location_ == LOCATION_SZONE) &&
                 (cnew.position_ != POS_FACEDOWN)) {
        pl->notify(fmt::format("Activating {} ({})", plnewspec, card.name_));
        op->notify(fmt::format("{} activating {} ({})", pl->nickname_, opspec,
                               cnew.name_));
      } else {
        fmt::println("Unknown move reason {}", reason);
      }
    } else if (msg_ == MSG_SWAP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code1 = read_u32();
      auto loc1 = read_loc_info();
      CardCode code2 = read_u32();
      auto loc2 = read_loc_info();
      Card cards[2];
      cards[0] = c_get_card(code1);
      cards[1] = c_get_card(code2);
      cards[0].set_location(loc1);
      cards[1].set_location(loc2);

      for (PlayerId pl = 0; pl < 2; pl++) {
        for (int i = 0; i < 2; i++) {
          auto c = cards[i];
          auto spec = c.get_spec(pl);
          auto plname = players_[1 - c.controler_]->nickname_;
          players_[pl]->notify("Card " + c.name_ + " swapped control towards " +
                               plname + " and is now located at " + spec + ".");
        }
      }
    } else if (msg_ == MSG_SET) {
      // B.3.a 3.8 finish: payload verified against edo9300
      // operations.cpp:2740-2742. Emit: [u32 code][loc_info 10B].
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_loc_info());
      push_event(EventType::Set, code, card.controler_,
                 EVENT_LOC_NONE,
                 edo_location_to_event_loc(card.location_));
      if (!verbose_) {
        return;
      }
      auto c = card.controler_;
      auto cpl = players_[c];
      auto opl = players_[1 - c];
      cpl->notify(fmt::format("You set {} ({}) in {} position.", card.name_,
                              card.get_spec(c), card.get_position()));
      opl->notify(fmt::format("{} sets {} in {} position.", cpl->nickname_,
                              card.get_spec(PlayerId(1 - c)),
                              card.get_position()));
    } else if (msg_ == MSG_EQUIP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto info = read_loc_info();
      auto c = info.controler;
      auto loc = info.location;
      auto seq = info.sequence;
      auto pos = info.position;
      Card card = get_card(c, loc, seq);
      info = read_loc_info();
      c = info.controler;
      loc = info.location;
      seq = info.sequence;
      pos = info.position;
      Card target = get_card(c, loc, seq);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto c = cardlist_info_for_player(card, pl);
        auto t = cardlist_info_for_player(target, pl);
        players_[pl]->notify(fmt::format("{} equipped to {}.", c, t));
      }
    } else if (msg_ == MSG_PLAYER_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto hint_type = read_u8();
      auto value = compat_read<uint32_t, uint64_t>();
      // TODO: handle this
      return;
    } else if (msg_ == MSG_HINT) {
      auto hint_type = read_u8();
      auto player = read_u8();
      auto value = compat_read<uint32_t, uint64_t>();

      if (hint_type == HINT_SELECTMSG && value == 501) {
        discard_hand_ = true;
      }
      // non-GUI don't need hint
      return;
      if (hint_type == HINT_SELECTMSG) {
        if (value > 2000) {
          CardCode code = value;
          players_[player]->notify(fmt::format("{} select {}",
                                               players_[player]->nickname_,
                                               c_get_card(code).name_));
        } else {
          players_[player]->notify(get_system_string(value));
        }
      } else if (hint_type == HINT_NUMBER) {
        players_[1 - player]->notify(
            fmt::format("Choice of player: {}", value));
      } else {
        fmt::println("Unknown hint type {} with value {}", hint_type, value);
      }
    } else if (msg_ == MSG_CARD_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto info = read_loc_info();
      uint8_t player = info.controler;
      uint8_t loc = info.location;
      uint8_t seq = info.sequence;
      uint8_t pos = info.position;
      uint8_t type = read_u8();
      uint32_t value = compat_read<uint32_t, uint64_t>();
      Card card = get_card(player, loc, seq);
      if (card.code_ == 0) {
        return;
      }
      if (type == CHINT_RACE) {
        std::string races_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           races_str));
        }
      } else if (type == CHINT_ATTRIBUTE) {
        std::string attributes_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           attributes_str));
        }
      } else {
        fmt::println("Unknown card hint type {} with value {}", type, value);
      }
    } else if (msg_ == MSG_POS_CHANGE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      uint8_t prevpos = card.position_;
      card.position_ = read_u8();

      auto pl = players_[card.controler_];
      auto op = players_[1 - card.controler_];
      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto prevpos_str = position_to_string(prevpos);
      auto pos_str = position_to_string(card.position_);
      pl->notify("The position of card " + plspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
      op->notify("The position of card " + opspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
    } else if (msg_ == MSG_BECOME_TARGET || msg_ == MSG_CARD_SELECTED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto count = compat_read<uint8_t, uint32_t>();
      std::vector<Card> cards;
      cards.reserve(count);
      for (int i = 0; i < count; ++i) {
        auto info = read_loc_info();
        auto c = info.controler;
        auto loc = info.location;
        auto seq = info.sequence;
        cards.push_back(get_card(c, loc, seq));
      }
      auto name = players_[chaining_player_]->nickname_;
      for (PlayerId pl = 0; pl < 2; pl++) {
        std::string str = name;
        if (msg_ == MSG_BECOME_TARGET) {
          str += " targets ";
        } else {
          str += " selects ";
        }
        for (int i = 0; i < count; ++i) {
          auto card = cards[i];
          auto spec = card.get_spec(pl);
          auto tcname = card.name_;
          if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
            tcname = position_to_string(card.position_) + " card";
          }
          str += spec + " (" + tcname + ")";
          if (i < count - 1) {
            str += ", ";
          }
        }
        players_[pl]->notify(str);
      }
    } else if (msg_ == MSG_CONFIRM_DECKTOP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto size = compat_read<uint8_t, uint32_t>();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = compat_read<uint8_t, uint32_t>();
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto p = players_[pl];
        if (pl == player) {
          p->notify(fmt::format("You reveal {} cards from your deck:", size));
        } else {
          p->notify(fmt::format("{} reveals {} cards from their deck:",
                                players_[player]->nickname_, size));
        }
        for (int i = 0; i < size; ++i) {
          p->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
        }
      }
    } else if (msg_ == MSG_RANDOM_SELECTED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = compat_read<uint8_t, uint32_t>();
      std::vector<Card> cards;

      for (int i = 0; i < count; ++i) {
        auto info = read_loc_info();
        auto c = info.controler;
        auto loc = info.location;
        if (loc & LOCATION_OVERLAY) {
          throw std::runtime_error("Overlay not supported for random selected");
        }
        auto seq = info.sequence;
        auto pos = info.position;
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto p = players_[pl];
        auto s = "card is";
        if (count > 1) {
          s = "cards are";
        }
        if (pl == player) {
          p->notify(fmt::format("Your {} {} randomly selected:", s, count));
        } else {
          p->notify(fmt::format("{}'s {} {} randomly selected:",
                                players_[player]->nickname_, s, count));
        }
        for (int i = 0; i < count; ++i) {
          p->notify(fmt::format("{}: {}", cards[i].get_spec(pl), cards[i].name_));
        }
      }

    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      auto size = compat_read<uint8_t, uint32_t>();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = compat_read<uint8_t, uint32_t>();
        if (verbose_) {
          cards.push_back(get_card(c, loc, seq));
        }
        revealed_.push_back(ls_to_spec(loc, seq, 0, c == player));
      }
      if (!verbose_) {
        return;
      }

      auto pl = players_[player];
      auto op = players_[1 - player];

      op->notify(fmt::format("{} shows you {} cards.", pl->nickname_, size));
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }
    } else if (msg_ == MSG_MISSED_EFFECT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 4;
      CardCode code = read_u32();
      Card card = c_get_card(code);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto str = get_system_string(1622);
        std::string fmt_str = "[%ls]";
        str = str.replace(str.find(fmt_str), fmt_str.length(), card.name_);
        players_[pl]->notify(str);
      }
    } else if (msg_ == MSG_SORT_CARD) {
      // TODO: implement action
      if (!verbose_) {
        dp_ = dl_;
        YGO_SetResponsei(pduel_, -1);
        return;
      }
      auto player = read_u8();
      auto size = compat_read<uint8_t, uint32_t>();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = compat_read<uint8_t, uint32_t>();
        auto seq = compat_read<uint8_t, uint32_t>();
        cards.push_back(get_card(c, loc, seq));
      }
      auto pl = players_[player];
      pl->notify(
          "Sort " + std::to_string(size) +
          " cards by entering numbers separated by spaces (c = cancel):");
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }

      fmt::println("sort card action not implemented");
      YGO_SetResponsei(pduel_, -1);

      // // generate all permutations
      // std::vector<int> perm(size);
      // std::iota(perm.begin(), perm.end(), 0);
      // std::vector<std::vector<int>> perms;
      // do {
      //   auto option = std::accumulate(perm.begin(), perm.end(),
      //   std::string(),
      //                                 [&](std::string &acc, int i) {
      //                                   return acc + std::to_string(i + 1) +
      //                                   " ";
      //                                 });
      //   options_.push_back(option);
      // } while (std::next_permutation(perm.begin(), perm.end()));
      // options_.push_back("c");
      // callback_ = [this](int idx) {
      //   const auto &option = options_[idx];
      //   if (option == "c") {
      //     resp_buf_[0] = 255;
      //     YGO_SetResponseb(pduel_, resp_buf_);
      //     return;
      //   }
      //   std::istringstream iss(option);
      //   int x;
      //   int i = 0;
      //   while (iss >> x) {
      //     resp_buf_[i] = uint8_t(x);
      //     i++;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      // };
    } else if (msg_ == MSG_ADD_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto pl = players_[player];
      PlayerId op_id = 1 - player;
      auto op = players_[op_id];
      // TODO: counter type to string
      pl->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_REMOVE_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto pl = players_[player];
      PlayerId op_id = 1 - player;
      auto op = players_[op_id];
      pl->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_ATTACK_DISABLED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      for (PlayerId pl = 0; pl < 2; pl++) {
        players_[pl]->notify(get_system_string(1621));
      }
    } else if (msg_ == MSG_SHUFFLE_SET_CARD) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      // TODO: implement output
      dp_ = dl_;
    } else if (msg_ == MSG_SHUFFLE_DECK) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto pl = players_[player];
      auto op = players_[1 - player];
      pl->notify("You shuffled your deck.");
      op->notify(pl->nickname_ + " shuffled their deck.");
    } else if (msg_ == MSG_SHUFFLE_EXTRA) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      for (int i = 0; i < count; ++i) {
        read_u32();
      }
      auto pl = players_[player];
      auto op = players_[1 - player];
      pl->notify(fmt::format("You shuffled your extra deck ({}).", count));
      op->notify(fmt::format("{} shuffled their extra deck ({}).", pl->nickname_, count));
    } else if (msg_ == MSG_SHUFFLE_HAND) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      auto player = read_u8();
      dp_ = dl_;

      auto pl = players_[player];
      auto op = players_[1 - player];
      pl->notify("You shuffled your hand.");
      op->notify(pl->nickname_ + " shuffled their hand.");
    } else if (msg_ == MSG_SUMMONED) {
      // B.3.a 3.8 finish: verified empty payload (operations.cpp:2307).
      // ygopro.h doesn't push_event here either.
      dp_ = dl_;
    } else if (msg_ == MSG_SUMMONING) {
      // B.3.a 3.8 finish: payload verified against edo9300
      // operations.cpp:2237-2239. Emit: [u32 code][loc_info 10B].
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_loc_info());
      push_event(EventType::Summon, code, card.controler_,
                 EVENT_LOC_NONE,
                 edo_location_to_event_loc(card.location_));
      if (!verbose_) {
        return;
      }
      const auto &nickname = players_[card.controler_]->nickname_;
      for (auto pl : players_) {
        pl->notify(nickname + " summoning " + card.name_ + " (" +
                   std::to_string(card.attack_) + "/" +
                   std::to_string(card.defense_) + ") in " +
                   card.get_position() + " position.");
      }
    } else if (msg_ == MSG_SPSUMMONED) {
      // B.3.a 3.8 finish: verified empty payload (operations.cpp:3223).
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONED) {
      // B.3.a 3.8 finish: verified empty payload (operations.cpp:2406).
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONING) {
      // B.3.a 3.8 finish: payload verified against edo9300
      // operations.cpp:2371-2373. Emit: [u32 code][loc_info 10B].
      auto code = read_u32();
      auto loc_info = read_loc_info();
      Card card = c_get_card(code);
      card.set_location(loc_info);
      push_event(EventType::Summon, code, card.controler_,
                 EVENT_LOC_NONE,
                 edo_location_to_event_loc(card.location_));
      if (!verbose_) {
        return;
      }
      auto cpl = players_[card.controler_];
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        players_[1 - pl]->notify(cpl->nickname_ + " flip summons " + spec +
                                 " (" + card.name_ + ")");
      }
    } else if (msg_ == MSG_SPSUMMONING) {
      // B.3.a 3.8 finish: payload verified against edo9300
      // operations.cpp:3147-3149 / 3360 / 3621. Emit:
      // [u32 code][loc_info 10B].
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_loc_info());
      push_event(EventType::Summon, code, card.controler_,
                 EVENT_LOC_NONE,
                 edo_location_to_event_loc(card.location_));
      if (!verbose_) {
        return;
      }
      const auto &nickname = players_[card.controler_]->nickname_;
      for (PlayerId p = 0; p < 2; p++) {
        auto pl = players_[p];
        auto pos = card.get_position();
        auto atk = std::to_string(card.attack_);
        auto def = std::to_string(card.defense_);
        std::string name = p == card.controler_ ? "You" : nickname;
        if (card.type_ & TYPE_LINK) {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + ") in " + pos + " position.");
        } else {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + "/" + def + ") in " + pos + " position.");
        }
      }
    } else if (msg_ == MSG_CHAIN_NEGATED) {
      // B.3.a 3.8 finish — chain_stack_ leak investigation: NEGATED
      // doesn't pop the engine's core.current_chain (operations.cpp:36
      // emits the message but pop_back happens inside CHAIN_SOLVED at
      // processor.cpp:4323). No pop here.
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_DISABLED) {
      // Same as NEGATED — no engine-side pop on DISABLED.
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_SOLVED) {
      // B.3.a 3.8 finish: pop one link from chain_stack_ to mirror
      // edo9300's `core.current_chain.pop_back()` at processor.cpp:4323
      // which fires immediately before the MSG_CHAIN_SOLVED message.
      // Without this pop, chain_stack_ would accumulate all chain links
      // for the duration of the chain (cleared only on CHAIN_END),
      // which drifts from ygopro's "what's currently resolving" view.
      if (!chain_stack_.empty()) {
        chain_stack_.pop_back();
      }
      dp_ = dl_;
      revealed_.clear();
    } else if (msg_ == MSG_CHAIN_SOLVING) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_END) {
      // B.3.a Chunk 3.8: clear chain_stack_ on CHAIN_END. edo9300 emits
      // this after the chain fully resolves (processor.cpp:4351).
      chain_stack_.clear();
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINING) {
      // B.3.a Chunk 3.8: payload verified against edo9300
      // processor.cpp:3700-3707. Sequence:
      //   [u32 phandler->data.code]
      //   [loc_info = u8 con, u8 loc, u32 seq, u32 pos]  (10 bytes)
      //   [u8 triggering_controler]
      //   [u8 triggering_location]
      //   [u32 triggering_sequence]   (edopro compat_read picks u32)
      //   [u64 peffect->description]  (edopro compat_read picks u64)
      //   [u32 chain_size + 1]         (edopro compat_read picks u32)
      // Match. Decode runs regardless of verbose so chain_stack_ tracks
      // even in training mode.
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_loc_info());
      auto tc = read_u8();
      auto tl = read_u8();
      auto ts = compat_read<uint8_t, uint32_t>();
      uint32_t desc = compat_read<uint32_t, uint64_t>();
      auto cs = compat_read<uint8_t, uint32_t>();
      (void)tc; (void)tl; (void)ts; (void)desc; (void)cs;
      chain_stack_.push_back(code);
      if (!verbose_) {
        return;
      }
      auto c = card.controler_;
      PlayerId o = 1 - c;
      chaining_player_ = c;
      players_[c]->notify("Activating " + card.get_spec(c) + " (" + card.name_ +
                          ")");
      players_[o]->notify(players_[c]->nickname_ + " activating " +
                          card.get_spec(o) + " (" + card.name_ + ")");
    } else if (msg_ == MSG_DAMAGE) {
      auto player = read_u8();
      auto amount = read_u32();
      _damage(player, amount);
    } else if (msg_ == MSG_RECOVER) {
      auto player = read_u8();
      auto amount = read_u32();
      _recover(player, amount);
    } else if (msg_ == MSG_LPUPDATE) {
      auto player = read_u8();
      auto lp = read_u32();
      if (lp >= lp_[player]) {
        _recover(player, lp - lp_[player]);
      } else {
        _damage(player, lp_[player] - lp);
      }
    } else if (msg_ == MSG_PAY_LPCOST) {
      auto player = read_u8();
      auto cost = read_u32();
      lp_[player] -= cost;
      if (!verbose_) {
        return;
      }
      auto pl = players_[player];
      pl->notify("You pay " + std::to_string(cost) + " LP. Your LP is now " +
                 std::to_string(lp_[player]) + ".");
      players_[1 - player]->notify(
          pl->nickname_ + " pays " + std::to_string(cost) + " LP. " +
          pl->nickname_ + "'s LP is now " + std::to_string(lp_[player]) + ".");
    } else if (msg_ == MSG_ATTACK) {
      // B.3.a 3.8 finish: payload verified against edo9300
      // processor.cpp:2117-2125. Emit: [loc_info attacker 10B][loc_info
      // target 10B] (target is loc_info{} all-zeros for direct attacks).
      auto attacker = read_loc_info();
      PlayerId ac = attacker.controler;
      auto aloc = attacker.location;
      auto aseq = attacker.sequence;
      auto apos = attacker.position;
      auto target = read_loc_info();
      PlayerId tc = target.controler;
      auto tloc = target.location;
      auto tseq = target.sequence;
      auto tpos = target.position;

      // Push Attack event. For direct attacks (target is loc_info{}
      // all-zero per processor.cpp:2123), to_loc reads EVENT_LOC_NONE.
      // Mirrors ygopro.h:5028-5039. Attacker code is looked up via
      // QUERY_CODE on the attacker's location; if the lookup throws
      // (shouldn't for a valid attacker), default to code=0.
      bool direct_attack = (tc == 0 && tloc == 0 && tseq == 0 && tpos == 0);
      CardCode attacker_code = 0;
      try {
        attacker_code = get_card_code(ac, aloc, aseq);
      } catch (...) {
        attacker_code = 0;
      }
      push_event(EventType::Attack, attacker_code, ac,
                 edo_location_to_event_loc(aloc),
                 direct_attack ? EVENT_LOC_NONE
                               : edo_location_to_event_loc(tloc));

      if (!verbose_) {
        return;
      }
      if ((ac == 0) && (aloc == 0) && (aseq == 0) && (apos == 0)) {
        return;
      }

      Card acard = get_card(ac, aloc, aseq);
      auto name = players_[ac]->nickname_;
      if ((tc == 0) && (tloc == 0) && (tseq == 0) && (tpos == 0)) {
        for (PlayerId i = 0; i < 2; i++) {
          players_[i]->notify(name + " prepares to attack with " +
                              acard.get_spec(i) + " (" + acard.name_ + ")");
        }
        return;
      }

      Card tcard = get_card(tc, tloc, tseq);
      for (PlayerId i = 0; i < 2; i++) {
        auto aspec = acard.get_spec(i);
        auto tspec = tcard.get_spec(i);
        auto tcname = tcard.name_;
        if ((tcard.controler_ != i) && (tcard.position_ & POS_FACEDOWN)) {
          tcname = tcard.get_position() + " card";
        }
        players_[i]->notify(name + " prepares to attack " + tspec + " (" +
                            tcname + ") with " + aspec + " (" + acard.name_ +
                            ")");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_START) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("begin damage");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_END) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("end damage");
      }
    } else if (msg_ == MSG_BATTLE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto attacker = read_loc_info();
      auto aa = read_u32();
      auto ad = read_u32();
      auto bd0 = read_u8();
      auto target = read_loc_info();
      auto da = read_u32();
      auto dd = read_u32();
      auto bd1 = read_u8();

      auto ac = attacker.controler;
      auto aloc = attacker.location;
      auto aseq = attacker.sequence;

      auto tc = target.controler;
      auto tloc = target.location;
      auto tseq = target.sequence;
      auto tpos = target.position;

      Card acard = get_card(ac, aloc, aseq);
      Card tcard;
      if (tloc != 0) {
        tcard = get_card(tc, tloc, tseq);
      }
      for (int i = 0; i < 2; i++) {
        auto pl = players_[i];
        std::string attacker_points;
        if (acard.type_ & TYPE_LINK) {
          attacker_points = std::to_string(aa);
        } else {
          attacker_points = std::to_string(aa) + "/" + std::to_string(ad);
        }
        if (tloc != 0) {
          std::string defender_points;
          if (tcard.type_ & TYPE_LINK) {
            defender_points = std::to_string(da);
          } else {
            defender_points = std::to_string(da) + "/" + std::to_string(dd);
          }
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks " +
                     tcard.name_ + " (" + defender_points + ")");
        } else {
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks");
        }
      }
    } else if (msg_ == MSG_WIN) {
      auto player = read_u8();
      auto reason = read_u8();
      auto winner = players_[player];
      auto loser = players_[1 - player];

      _duel_end(player, reason);

      auto l_reason = reason_to_string(reason);
      if (verbose_) {
        winner->notify("You won (" + l_reason + ").");
        loser->notify("You lost (" + l_reason + ").");
      }
    } else if (msg_ == MSG_RETRY) {
      throw std::runtime_error("Retry");
    } else if (msg_ == MSG_SELECT_BATTLECMD) {
      // B.3.a 3.5 port. LegalAction mapping mirrors IDLECMD's pattern
      // but with a different set of command types:
      //   activatable → activate_spec(data, spec), response_ = idx << 16
      //   attackable  → act_spec(Attack, spec),    response_ = (idx<<16)+1
      //   m (to M2)   → phase(Main2),              response_ = 2
      //   e (to EP)   → phase(End),                response_ = 3
      auto player = read_u8();
      auto activatable = read_cardlist_spec(true, true);
      auto attackable = read_cardlist_spec(false, false, true);
      bool to_m2 = read_u8();
      bool to_ep = read_u8();

      auto pl = players_[player];
      if (verbose_) {
        pl->notify("Battle menu:");
      }
      uint32_t fidx = 0;
      for (const auto [code, spec, data] : activatable) {
        // B.3.a step-4: unpack raw desc → effect_idx so obs:actions_ col 6
        // (_set_obs_action_effect) sees the canonical encoding
        // (< CARD_EFFECT_OFFSET → system_string_to_id; ≥ → 2..15 card effect).
        auto [code_d, eff_idx] = unpack_desc(code, data);
        LegalAction la = LegalAction::activate_spec(eff_idx, spec);
        la.msg_ = msg_;
        auto cit = card_ids_.find(code);
        la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        la.response_ = (fidx << 16) + 0;
        legal_actions_.push_back(la);
        if (verbose_) {
          auto [loc, seq, pos] = spec_to_ls(spec);
          auto c = get_card(player, loc, seq);
          pl->notify("v " + spec + ": activate " + c.name_ + " (" +
                     std::to_string(c.attack_) + "/" +
                     std::to_string(c.defense_) + ")");
        }
        fidx++;
      }
      fidx = 0;
      for (const auto [code, spec, data] : attackable) {
        LegalAction la = LegalAction::act_spec(ActionAct::Attack, spec);
        la.msg_ = msg_;
        auto cit = card_ids_.find(code);
        la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        la.response_ = (fidx << 16) + 1;
        legal_actions_.push_back(la);
        if (verbose_) {
          auto [loc, seq, pos] = spec_to_ls(spec);
          auto c = get_card(player, loc, seq);
          if (c.type_ & TYPE_LINK) {
            pl->notify("a " + spec + ": " + c.name_ + " (" +
                       std::to_string(c.attack_) + ") attack");
          } else {
            pl->notify("a " + spec + ": " + c.name_ + " (" +
                       std::to_string(c.attack_) + "/" +
                       std::to_string(c.defense_) + ") attack");
          }
        }
        fidx++;
      }
      if (to_m2) {
        LegalAction la = LegalAction::phase(ActionPhase::Main2);
        la.msg_ = msg_;
        la.response_ = 2;
        legal_actions_.push_back(la);
        if (verbose_) {
          pl->notify("m: Main phase 2.");
        }
      }
      if (to_ep) {
        if (!to_m2) {
          LegalAction la = LegalAction::phase(ActionPhase::End);
          la.msg_ = msg_;
          la.response_ = 3;
          legal_actions_.push_back(la);
          if (verbose_) {
            pl->notify("e: End phase.");
          }
        }
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        // Step-2: response_ pre-computed at handler time — (fidx<<16)
        // for activate, (fidx<<16)+1 for attack, 2 for Main2 phase,
        // 3 for End phase.
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };
    } else if (msg_ == MSG_SELECT_UNSELECT_CARD) {
      auto player = read_u8();
      bool finishable = read_u8();
      bool cancelable = read_u8();
      auto min = compat_read<uint8_t, uint32_t>();
      auto max = compat_read<uint8_t, uint32_t>();
      auto select_size = compat_read<uint8_t, uint32_t>();

      std::vector<std::string> select_specs;
      select_specs.reserve(select_size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          auto loc_info = read_loc_info();
          Card card = c_get_card(code);
          card.set_location(loc_info);
          cards.push_back(card);
        }
        auto pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          pl->notify(spec + ": " + card.name_);
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          dp_ += 4;
          auto loc_info = read_loc_info();
          auto spec = ls_to_spec(loc_info, player);
          select_specs.push_back(spec);
        }
      }

      auto unselect_size = compat_read<uint8_t, uint32_t>();

      // unselect not allowed (no regrets!)
      if (compat_mode_) {
        dp_ += 8 * unselect_size;
      } else {
        dp_ += 14 * unselect_size;
      }

      // B.3.a 3.5 port. Each selectable card becomes a LegalAction
      // with spec_ = the card's spec string. The finish marker (if
      // finishable) becomes LegalAction::finish(). This matches
      // ygopro's single-card-per-action model for UNSELECT_CARD
      // (different from SELECT_CARD's combination approach).
      for (size_t j = 0; j < select_specs.size(); ++j) {
        LegalAction la = LegalAction::from_spec(select_specs[j]);
        la.msg_ = msg_;
        la.response_ = static_cast<uint32_t>(j);
        legal_actions_.push_back(la);
      }
      if (finishable) {
        LegalAction la = LegalAction::finish();
        la.msg_ = msg_;
        la.response_ = static_cast<uint32_t>(-1);
        legal_actions_.push_back(la);
      }

      // cancelable and finishable not needed

      to_play_ = player;
      if (compat_mode_) {
        callback_ = [this](int idx) {
          // Step-2: check .finish_ flag instead of options_[idx]=="f".
          if (legal_actions_[idx].finish_) {
            YGO_SetResponsei(pduel_, -1);
          } else {
            resp_buf_[0] = 1;
            resp_buf_[1] = static_cast<uint8_t>(idx);
            YGO_SetResponseb(pduel_, resp_buf_);
          }
        };
      } else {
        callback_ = [this](int idx) {
          if (legal_actions_[idx].finish_) {
            YGO_SetResponsei(pduel_, -1);
          } else {
            uint32_t ret = 1;
            memcpy(resp_buf_, &ret, sizeof(ret));
            uint32_t v = idx;
            memcpy(resp_buf_ + 4, &v, sizeof(v));
            YGO_SetResponseb(pduel_, resp_buf_, 8);
          }
        };
      }
    } else if (msg_ == MSG_SELECT_CARD) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = compat_read<uint8_t, uint32_t>();
      auto max = compat_read<uint8_t, uint32_t>();
      auto size = compat_read<uint8_t, uint32_t>();

      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          Card card = c_get_card(code);
          card.set_location(read_loc_info());
          cards.push_back(card);
        }
        auto pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          if (card.controler_ != player && card.position_ & POS_FACEDOWN) {
            pl->notify(spec + ": " + card.get_position() + " card");
          } else {
            pl->notify(spec + ": " + card.name_);
          }
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          loc_info info = read_loc_info();
          auto spec = ls_to_spec(info, player);
          specs.push_back(spec);
        }
      }

      if (min > spec_.config["max_multi_select"_]) {
        if (discard_hand_) {
          // random discard
          std::vector<int> comb(size);
          std::iota(comb.begin(), comb.end(), 0);
          std::shuffle(comb.begin(), comb.end(), gen_);
          resp_buf_[0] = min;
          for (int i = 0; i < min; ++i) {
            resp_buf_[i + 1] = comb[i];
          }
          YGO_SetResponseb(pduel_, resp_buf_);
          discard_hand_ = false;
          return;
        }

        show_turn();

        show_deck(player);
        show_history_actions(player);

        show_deck(1-player);
        show_history_actions(1-player);

        fmt::println("player: {}, min: {}, max: {}, size: {}", player, min, max, size);
        std::cout << std::flush;
        throw std::runtime_error(
            fmt::format("Min > {} not implemented for select card",
                        spec_.config["max_multi_select"_]));
      }

      max = std::min(max, uint32_t(spec_.config["max_multi_select"_]));

      // B.3.a 3.5 convergence: sequenced multi-select (mode 0). Per
      // `edopro_serve_model_multiselect_check.md`, deployment's
      // OnSelectCard calls the model once per card pick with
      // OnSelectYesNo between picks as the finish marker —
      // training must match for priority-#1 parity. Legacy
      // combination-enumeration path replaced.
      init_multi_select(static_cast<int>(min), static_cast<int>(max),
                        0, specs, 0);
      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_TRIBUTE) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = compat_read<uint8_t, uint32_t>();
      auto max = compat_read<uint8_t, uint32_t>();
      auto size = compat_read<uint8_t, uint32_t>();

      if (max > 3) {
        throw std::runtime_error("Max > 3 not implemented for select tribute");
      }

      std::vector<int> release_params;
      release_params.reserve(size);
      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = compat_read<uint8_t, uint32_t>();
          auto release_param = read_u8();
          Card card = get_card(controller, loc, seq);
          cards.push_back(card);
          release_params.push_back(release_param);
        }
        auto pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) +
                   " cards to tribute separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          pl->notify(spec + ": " + card.name_);
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = compat_read<uint8_t, uint32_t>();
          auto release_param = read_u8();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          specs.push_back(spec);

          release_params.push_back(release_param);
        }
      }

      bool has_weight =
          std::any_of(release_params.begin(), release_params.end(),
                      [](int i) { return i != 1; });

      if (min != max) {
        throw std::runtime_error(
          fmt::format("min({}) != max({}), not implemented for select tribute", min, max));
      }

      // B.3.a 3.5 convergence: sequenced multi-select. TRIBUTE delegates
      // to OnSelectCard in deployment (GameAI.cs:767-773), so uses the
      // same mode-0 sequenced pattern as SELECT_CARD. The `has_weight`
      // / combinations_with_weight infrastructure isn't needed under
      // sequenced — the engine will reject invalid combinations via
      // MSG_RETRY, and release_params become a gotcha to watch during
      // audit rollouts. For YugiKaiba, weighted tributes don't occur.
      //
      // If audit rollouts show weighted tributes failing validation,
      // we'd need to filter specs at each sub-decision so only release-
      // param-valid cards are offered. Defer until we see it.
      (void)has_weight;
      init_multi_select(static_cast<int>(min), static_cast<int>(max),
                        0, specs, 0);
      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_SUM) {
      uint8_t mode;
      uint8_t player;
      if (compat_mode_) {
        mode = read_u8();
        player = read_u8();
      } else {
        player = read_u8();
        mode = read_u8();
      }
      auto val = read_u32();
      int min = compat_read<uint8_t, uint32_t>();
      int max = compat_read<uint8_t, uint32_t>();
      auto must_select_size = compat_read<uint8_t, uint32_t>();

      if (mode == 0) {
        if (must_select_size != 1) {
          throw std::runtime_error(
              " must select size: " + std::to_string(must_select_size) +
              " not implemented for MSG_SELECT_SUM");
        }
      } else {
        if (min != 0 || max != 0 || must_select_size != 0) {
          std::string err = fmt::format(
              "min: {}, max: {}, must select size: {} not implemented for "
              "MSG_SELECT_SUM, mode: {}",
              min, max, must_select_size, mode);
          throw std::runtime_error(err);
        }
      }

      std::vector<int> must_select_params;
      std::vector<std::string> must_select_specs;
      std::vector<int> select_params;
      std::vector<std::string> select_specs;

      must_select_params.reserve(must_select_size);
      must_select_specs.reserve(must_select_size);

      int expected = val;
      if (verbose_) {
        std::vector<Card> must_select;
        must_select.reserve(must_select_size);
        for (int i = 0; i < must_select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          uint32_t seq;
          if (compat_mode_) {
            seq = read_u8();
          } else {
            seq = read_u32();
            dp_ += 4;
          }
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          must_select.push_back(card);
          must_select_params.push_back(param);
        }
        if (must_select_size > 0) {
          expected -= must_select_params[0] & 0xff;
        }
        auto pl = players_[player];
        pl->notify("Select cards with a total value of " +
                  std::to_string(expected) + ", seperated by spaces.");
        for (const auto &card : must_select) {
          auto spec = card.get_spec(player);
          must_select_specs.push_back(spec);
          pl->notify(card.name_ + " (" + spec +
                    ") must be selected, automatically selected.");
        }
      } else {
        for (int i = 0; i < must_select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          uint32_t seq;
          if (compat_mode_) {
            seq = read_u8();
          } else {
            seq = read_u32();
            dp_ += 4;
          }
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          must_select_specs.push_back(spec);
          must_select_params.push_back(param);
        }
        if (must_select_size > 0) {
          expected -= must_select_params[0] & 0xff;
        }
      }

      uint8_t select_size = compat_read<uint8_t, uint32_t>();
      select_params.reserve(select_size);
      select_specs.reserve(select_size);

      if (verbose_) {
        std::vector<Card> select;
        select.reserve(select_size);
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          uint32_t seq;
          if (compat_mode_) {
            seq = read_u8();
          } else {
            seq = read_u32();
            dp_ += 4;
          }
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          select.push_back(card);
          select_params.push_back(param);
        }
        auto pl = players_[player];
        for (const auto &card : select) {
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          pl->notify(spec + ": " + card.name_);
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          uint32_t seq;
          if (compat_mode_) {
            seq = read_u8();
          } else {
            seq = read_u32();
            dp_ += 4;
          }
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          select_specs.push_back(spec);
          select_params.push_back(param);
        }
      }

      std::vector<std::vector<int>> card_levels;
      for (int i = 0; i < select_size; ++i) {
        std::vector<int> levels;
        int level1 = select_params[i] & 0xff;
        int level2 = (select_params[i] >> 16);
        if (level1 > 0) {
          levels.push_back(level1);
        }
        if (level2 > 0) {
          levels.push_back(level2);
        }
        card_levels.push_back(levels);
      }

      std::vector<std::vector<int>> combs =
          combinations_with_weight2(card_levels, expected, true);

      // B.3.a 3.5 convergence: sequenced multi-select (mode 1+). Combs
      // are pre-computed as before; init_multi_select seeds
      // legal_actions_ with the first-element specs of each admissible
      // comb, and _callback_multi_select_2 prunes combs on each pick.
      // Response emitted by _callback_multi_select_2_finish when a
      // comb empties. Note: SELECT_SUM is NOT model-routed in
      // deployment (edopro_serve_model_multiselect_check.md §4) —
      // WindBot's heuristics handle it. Training still exercises
      // SELECT_SUM via self-play, so sequenced internal consistency
      // matters. ygopro.h:5585 is the reference.
      init_multi_select(min, max, static_cast<int>(must_select_size),
                        select_specs, 1, combs);
      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };

    } else if (msg_ == MSG_SELECT_CHAIN) {
      auto player = read_u8();
      uint32_t size;
      if (compat_mode_) {
        size = read_u8();
      }
      auto spe_count = read_u8();
      bool forced = read_u8();
      dp_ += 8;
      if (!compat_mode_) {
        size = read_u32();
      }
      // auto hint_timing = read_u32();
      // auto other_timing = read_u32();

      std::vector<Card> cards;
      std::vector<uint32_t> descs;
      std::vector<uint32_t> spec_codes;
      for (int i = 0; i < size; ++i) {
        uint8_t flag;
        if (compat_mode_) {
          flag = read_u8();
        }
        CardCode code = read_u32();
        auto loc_info = read_loc_info();
        if (verbose_) {
          Card card = c_get_card(code);
          card.set_location(loc_info);
          cards.push_back(card);
          spec_codes.push_back(card.get_spec_code(player));
        } else {
          spec_codes.push_back(
            ls_to_spec_code(loc_info, player));
        }
        uint32_t desc = compat_read<uint32_t, uint64_t>();
        descs.push_back(desc);
        if (!compat_mode_) {
          flag = read_u8();
        }
      }

      if ((size == 0) && (spe_count == 0)) {
        // non-GUI don't need this
        // if (verbose_) {
        //   fmt::println("keep processing");
        // }
        YGO_SetResponsei(pduel_, -1);
        return;
      }

      auto pl = players_[player];
      auto op = players_[1 - player];
      chaining_player_ = player;
      if (!op->seen_waiting_) {
        if (verbose_) {
          op->notify("Waiting for opponent.");
        }
        op->seen_waiting_ = true;
      }

      std::vector<int> chain_index;
      ankerl::unordered_dense::map<uint32_t, int> chain_counts;
      ankerl::unordered_dense::map<uint32_t, int> chain_orders;
      std::vector<std::string> chain_specs;
      std::vector<std::string> effect_descs;
      for (int i = 0; i < size; i++) {
        chain_index.push_back(i);
        chain_counts[spec_codes[i]] += 1;
      }
      for (int i = 0; i < size; i++) {
        auto spec_code = spec_codes[i];
        auto cs = code_to_spec(spec_code);
        auto chain_count = chain_counts[spec_code];
        if (chain_count > 1) {
          cs.push_back('a' + chain_orders[spec_code]);
        }
        chain_orders[spec_code]++;
        chain_specs.push_back(cs);
        if (verbose_) {
          const auto &card = cards[i];
          effect_descs.push_back(card.get_effect_description(descs[i], true));
        }
      }

      if (verbose_) {
        if (forced) {
          pl->notify("Select chain:");
        } else {
          pl->notify("Select chain (c to cancel):");
        }
        for (int i = 0; i < size; i++) {
          const auto &effect_desc = effect_descs[i];
          if (effect_desc.empty()) {
            pl->notify(chain_specs[i] + ": " + cards[i].name_);
          } else {
            pl->notify(chain_specs[i] + " (" + cards[i].name_ +
                       "): " + effect_desc);
          }
        }
      }

      // B.3.a 3.5 port. Payload parity already audited during Chunk 3.8
      // chain-stack work (MSG_CHAINING uses same loc_info layout). LegalAction
      // mapping: each chain candidate becomes activate_spec(eff_idx, spec).
      // The forced flag gates the cancel option.
      for (size_t i = 0; i < chain_specs.size(); ++i) {
        const auto &spec = chain_specs[i];
        auto [code_u, eff_idx] = unpack_desc(0, descs[i]);
        LegalAction la = LegalAction::activate_spec(eff_idx, spec);
        if (code_u != 0) {
          auto cit = card_ids_.find(code_u);
          la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        }
        la.msg_ = msg_;
        la.response_ = static_cast<uint32_t>(i);
        legal_actions_.push_back(la);
      }
      if (!forced) {
        LegalAction la = LegalAction::cancel();
        la.msg_ = msg_;
        la.response_ = static_cast<uint32_t>(-1);
        legal_actions_.push_back(la);
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        // Step-2: response_ pre-computed as i (pick) or -1 (cancel).
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };
    } else if (msg_ == MSG_SELECT_YESNO) {
      // B.3.a 3.5 port. Payload: [u8 player][u64 desc] (non-compat).
      // Parity: mirrors edo9300 MSG_SELECT_YESNO emit (same shape as
      // MSG_SELECT_EFFECTYN but without the card loc_info). LegalAction
      // mapping follows ygopro.h:5683-5728: "yes" → activate_spec, "no"
      // → cancel. response_ preserves the idx=0→1, idx=1→0 inversion.
      auto player = read_u8();
      auto desc = compat_read<uint32_t, uint64_t>();
      auto [code, eff_idx] = unpack_desc(0, desc);
      {
        LegalAction la_yes = LegalAction::activate_spec(eff_idx, "");
        if (code != 0) {
          auto cit = card_ids_.find(code);
          la_yes.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        }
        la_yes.msg_ = msg_;
        la_yes.response_ = 1;
        legal_actions_.push_back(la_yes);

        LegalAction la_no = LegalAction::cancel();
        la_no.msg_ = msg_;
        la_no.response_ = 0;
        legal_actions_.push_back(la_no);
      }
      if (verbose_) {
        auto pl = players_[player];
        pl->notify("TODO: MSG_SELECT_YESNO desc");
        pl->notify("Please enter y or n.");
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };
    } else if (msg_ == MSG_SELECT_EFFECTYN) {
      // B.3.a 3.5 port. Payload (non-compat): [u8 player][u32 code]
      // [loc_info 10B][u64 desc] = 1 + 4 + 10 + 8 = 23 bytes. LegalAction
      // mapping: activate_spec(eff_idx, spec) as "yes", cancel() as "no"
      // (ygopro.h:5729-5790). Unlike YESNO, EFFECTYN carries a card
      // spec_ field so the policy can localize the effect source.
      auto player = read_u8();
      CardCode code = read_u32();
      loc_info loc = read_loc_info();
      auto desc = compat_read<uint32_t, uint64_t>();
      std::string spec = ls_to_spec(loc, player);
      auto [code_d, eff_idx] = unpack_desc(code, desc);
      if (desc == 0) {
        code_d = code;
      }
      {
        LegalAction la_yes = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          auto cit = card_ids_.find(code_d);
          la_yes.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        }
        la_yes.msg_ = msg_;
        la_yes.response_ = 1;
        legal_actions_.push_back(la_yes);

        LegalAction la_no = LegalAction::cancel();
        // Cancel for EFFECTYN carries the card spec — the obs encoder
        // writes spec for BOTH yes and no (MSG_SELECT_EFFECTYN branch in
        // _set_obs_action). Matches legacy options_ = "n spec".
        la_no.spec_ = spec;
        la_no.msg_ = msg_;
        la_no.response_ = 0;
        legal_actions_.push_back(la_no);
      }
      if (verbose_) {
        Card card = c_get_card(code);
        card.set_location(loc);
        auto pl = players_[player];
        auto name = card.name_;
        std::string s;
        if (desc == 0) {
          s = "From " + card.get_spec(player) + ", activate " + name + "?";
        } else if (desc < 2048) {
          s = get_system_string(desc);
          std::string fmt_str = "[%ls]";
          auto pos = find_substrs(s, fmt_str);
          if (pos.size() == 0) {
            // nothing to replace
          } else if (pos.size() == 1) {
            auto p = pos[0];
            s = s.substr(0, p) + name + s.substr(p + fmt_str.size());
          } else if (pos.size() == 2) {
            auto p1 = pos[0];
            auto p2 = pos[1];
            s = s.substr(0, p1) + card.get_spec(player) +
                s.substr(p1 + fmt_str.size(), p2 - p1 - fmt_str.size()) + name +
                s.substr(p2 + fmt_str.size());
          } else {
            throw std::runtime_error("Unknown effectyn desc " +
                                     std::to_string(desc) + " of " + name);
          }
        } else if (desc < 10000u) {
          s = get_system_string(desc);
        } else {
          CardCode code_s = (desc >> 4) & 0x0fffffff;
          uint32_t offset = desc & 0xf;
          if (cards_.find(code_s) != cards_.end()) {
            auto &card_ = c_get_card(code_s);
            s = card_.strings_[offset];
            if (s.empty()) {
              s = "???";
            }
          } else {
            throw std::runtime_error("Unknown effectyn desc " +
                                     std::to_string(desc) + " of " + name);
          }
        }
        pl->notify(s);
        pl->notify("Please enter y or n.");
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };
    } else if (msg_ == MSG_SELECT_OPTION) {
      // B.3.a Chunk 3.5/3.6 — first LegalAction-migrated handler.
      //
      // Payload parity verified against edo9300 processor.cpp (search
      // for `MSG_SELECT_OPTION`; emitter writes [u8 playerid][u8 size]
      // [size × u64 description]). edopro's compat_read<uint32_t, uint64_t>
      // reads u64 in non-compat mode. Match.
      //
      // Action model transition: `options_` (legacy strings "1".."N")
      // AND `legal_actions_` (LegalAction::activate_spec) are populated
      // in parallel. Callback still reads `options_[idx]` for the
      // notification path and `idx` itself for `YGO_SetResponsei`; the
      // LegalAction carries the decoded (code, effect_idx, cid_) so
      // the forthcoming obs:actions_ 12-col port can emit the right
      // cid/effect fields for this handler without reparsing strings.
      //
      // Invariant required by the transition: index i in options_ MUST
      // correspond to index i in legal_actions_. callback_(idx) is
      // safe as long as both remain in lock-step.
      auto player = read_u8();
      auto size = read_u8();
      to_play_ = player;
      if (verbose_) {
        players_[player]->notify("Select an option:");
      }
      for (int i = 0; i < size; ++i) {
        auto desc = compat_read<uint32_t, uint64_t>();
        auto [code, eff_idx] = unpack_desc(0, desc);
        if (desc == 0) {
          throw std::runtime_error(fmt::format(
              "Unknown desc {} in select_option", desc));
        }
        LegalAction la = LegalAction::activate_spec(eff_idx, "");
        if (code != 0) {
          auto cit = card_ids_.find(code);
          la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        }
        la.msg_ = msg_;
        la.response_ = static_cast<uint32_t>(i);
        legal_actions_.push_back(la);

        if (verbose_) {
          // The pre-migration code had a `TODO: MSG_SELECT_OPTION desc`
          // placeholder for the effect description. Preserved as-is;
          // richer effect descriptions are verbose-only cosmetic.
          players_[player]->notify(std::to_string(i + 1) +
                                   ": TODO: MSG_SELECT_OPTION desc");
        }
      }
      callback_ = [this](int idx) {
        // B.3.a 3.5/3.6 step 2: read response from legal_actions_, not
        // options_. response_ was pre-computed to idx at handler time.
        if (verbose_) {
          players_[to_play_]->notify(
              fmt::format("You selected option {}.", idx + 1));
          players_[1 - to_play_]->notify(fmt::format(
              "{} selected option {}.",
              players_[to_play_]->nickname_, idx + 1));
        }
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };
    } else if (msg_ == MSG_SELECT_IDLECMD) {
      // B.3.a 3.5 port. Payload parses via read_cardlist_spec helpers.
      // LegalAction mapping:
      //   summonable   → act_spec(Summon, spec),    response_ = idx << 16
      //   spsummon     → act_spec(SpSummon, spec),  response_ = (idx << 16)+1
      //   repos        → act_spec(Repo, spec),      response_ = (idx << 16)+2
      //   idle_mset    → act_spec(MSet, spec),      response_ = (idx << 16)+3
      //   idle_set     → act_spec(Set, spec),       response_ = (idx << 16)+4
      //   idle_activate→ activate_spec(?, spec),    response_ = (idx << 16)+5
      //   b (to BP)    → phase(Battle),             response_ = 6
      //   e (to EP)    → phase(End),                response_ = 7
      // The idx used in the callback's (idx << 16) is the per-command-
      // family index (not the options_ global idx), so response_ stores
      // the per-family index.
      int32_t player = read_u8();
      auto summonable_ = read_cardlist_spec();
      auto spsummon_ = read_cardlist_spec();
      auto repos_ = read_cardlist_spec(false);
      auto idle_mset_ = read_cardlist_spec();
      auto idle_set_ = read_cardlist_spec();
      auto idle_activate_ = read_cardlist_spec(true, true);
      bool to_bp_ = read_u8();
      bool to_ep_ = read_u8();
      read_u8(); // can_shuffle

      int offset = 0;

      auto pl = players_[player];
      if (verbose_) {
        pl->notify("Select a card and action to perform.");
      }
      uint32_t fidx = 0;
      for (const auto &[code, spec, data] : summonable_) {
        LegalAction la = LegalAction::act_spec(ActionAct::Summon, spec);
        la.msg_ = msg_;
        auto cit = card_ids_.find(code);
        la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        la.response_ = (fidx << 16) + 0;
        legal_actions_.push_back(la);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify("s " + spec + ": Summon " + name +
                     " in face-up attack position.");
        }
        fidx++;
      }
      offset += summonable_.size();
      int spsummon_offset = offset;
      fidx = 0;
      for (const auto &[code, spec, data] : spsummon_) {
        LegalAction la = LegalAction::act_spec(ActionAct::SpSummon, spec);
        la.msg_ = msg_;
        auto cit = card_ids_.find(code);
        la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        la.response_ = (fidx << 16) + 1;
        legal_actions_.push_back(la);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify("c " + spec + ": Special summon " + name + ".");
        }
        fidx++;
      }
      offset += spsummon_.size();
      int repos_offset = offset;
      fidx = 0;
      for (const auto &[code, spec, data] : repos_) {
        LegalAction la = LegalAction::act_spec(ActionAct::Repo, spec);
        la.msg_ = msg_;
        auto cit = card_ids_.find(code);
        la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        la.response_ = (fidx << 16) + 2;
        legal_actions_.push_back(la);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify("r " + spec + ": Reposition " + name + ".");
        }
        fidx++;
      }
      offset += repos_.size();
      int mset_offset = offset;
      fidx = 0;
      for (const auto &[code, spec, data] : idle_mset_) {
        LegalAction la = LegalAction::act_spec(ActionAct::MSet, spec);
        la.msg_ = msg_;
        auto cit = card_ids_.find(code);
        la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        la.response_ = (fidx << 16) + 3;
        legal_actions_.push_back(la);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify("m " + spec + ": Summon " + name +
                     " in face-down defense position.");
        }
        fidx++;
      }
      offset += idle_mset_.size();
      int set_offset = offset;
      fidx = 0;
      for (const auto &[code, spec, data] : idle_set_) {
        LegalAction la = LegalAction::act_spec(ActionAct::Set, spec);
        la.msg_ = msg_;
        auto cit = card_ids_.find(code);
        la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        la.response_ = (fidx << 16) + 4;
        legal_actions_.push_back(la);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify("t " + spec + ": Set " + name + ".");
        }
        fidx++;
      }
      offset += idle_set_.size();
      int activate_offset = offset;
      ankerl::unordered_dense::map<std::string, int> idle_activate_count;
      for (const auto &[code, spec, data] : idle_activate_) {
        idle_activate_count[spec] += 1;
      }
      ankerl::unordered_dense::map<std::string, int> activate_count;
      fidx = 0;
      for (const auto &[code, spec, data] : idle_activate_) {
        std::string option = "v " + spec;
        int count = idle_activate_count[spec];
        activate_count[spec]++;
        if (count > 1) {
          option.push_back('a' + activate_count[spec] - 1);
        }
        // B.3.a step-4: unpack raw desc so la.effect_ is in canonical form
        // (see BATTLECMD note above).
        auto [code_d, eff_idx] = unpack_desc(code, data);
        LegalAction la = LegalAction::activate_spec(eff_idx, spec);
        la.msg_ = msg_;
        auto cit = card_ids_.find(code);
        la.cid_ = (cit != card_ids_.end()) ? cit->second : 0;
        la.response_ = (fidx << 16) + 5;
        // Per-spec activation ordinal mirrors the legacy options_ alpha
        // suffix: first activatable entry for this spec → 0, second → 1,
        // etc. Matches the obs encoder's `cmd_act2id['v'] + offset` path.
        la.act_offset_ = count > 1
            ? static_cast<uint8_t>(activate_count[spec] - 1)
            : 0;
        legal_actions_.push_back(la);
        if (verbose_) {
          pl->notify(option + ": " +
                     c_get_card(code).get_effect_description(data));
        }
        fidx++;
      }

      if (to_bp_) {
        LegalAction la = LegalAction::phase(ActionPhase::Battle);
        la.msg_ = msg_;
        la.response_ = 6;
        legal_actions_.push_back(la);
        if (verbose_) {
          pl->notify("b: Enter the battle phase.");
        }
      }
      if (to_ep_) {
        if (!to_bp_) {
          LegalAction la = LegalAction::phase(ActionPhase::End);
          la.msg_ = msg_;
          la.response_ = 7;
          legal_actions_.push_back(la);
          if (verbose_) {
            pl->notify("e: End phase.");
          }
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        // Step-2: response_ pre-computed per family at handler time
        // (shifted encoding (fidx<<16)|cmd_code for card commands, or
        // 6 / 7 for phase transitions).
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };
    } else if (msg_ == MSG_SELECT_PLACE || msg_ == MSG_SELECT_DISFIELD) {
      // B.3.a 3.5 port. Payload: [u8 player][u8 count][u32 flag]. Two
      // message ids (PLACE and DISFIELD) share identical parsing and
      // response shape — only the verbose notification wording differs.
      // LegalAction mapping: spec_ carries the edopro spec string
      // (e.g. "m3", "os5"). The ActionPlace enum mapping from
      // ygopro.h:5981-6044 uses a separate flag_to_usable_places()
      // helper that edopro doesn't have; populating place_ is deferred
      // to step 2/4 — the spec_ field alone is sufficient for the
      // callback's 3-byte response after the refactor.
      auto player = read_u8();
      auto count = read_u8();
      if (count == 0) {
        count = 1;
      }
      auto flag = read_u32();
      auto specs = flag_to_usable_cardspecs(flag);
      for (const auto &spec : specs) {
        LegalAction la = LegalAction::from_spec(spec);
        la.msg_ = msg_;
        // B.3.a step-4: populate la.place_ from cmd_place2id. The 1-based
        // id in cmd_place2id (m1=1..os8=30) matches ActionPlace enum
        // values exactly (MZone1=1..OpSZone8=30), so a direct cast is safe.
        auto pit = cmd_place2id.find(spec);
        if (pit != cmd_place2id.end()) {
          la.place_ = static_cast<ActionPlace>(pit->second);
        }
        // response_ left 0 — the 3-byte YGO_SetResponseb payload is
        // reconstructed from spec_ in the callback.
        legal_actions_.push_back(la);
      }
      if (verbose_) {
        std::string specs_str = specs[0];
        for (size_t i = 1; i < specs.size(); ++i) {
          specs_str += ", " + specs[i];
        }
        const char *label =
            msg_ == MSG_SELECT_PLACE ? "place" : "disfield";
        if (count == 1) {
          players_[player]->notify(fmt::format(
              "Select {} for card, one of {}.", label, specs_str));
        } else if (msg_ == MSG_SELECT_PLACE) {
          players_[player]->notify(fmt::format(
              "Select {} {}s for card, from {}.", count, label, specs_str));
        } else {
          throw std::runtime_error("Select disfield count " +
                                   std::to_string(count) + " not implemented");
        }
      }
      to_play_ = player;
      callback_ = [this, player](int idx) {
        // Step-2: read spec from LegalAction (same string content as
        // options_[idx]; response_ is 0 because the 3-byte response
        // is reconstructed from the spec via spec_to_ls).
        std::string spec = legal_actions_[idx].spec_;
        auto plr = player;
        if (!spec.empty() && spec[0] == 'o') {
          plr = 1 - player;
          spec = spec.substr(1);
        }
        auto [loc, seq, pos] = spec_to_ls(spec);
        resp_buf_[0] = plr;
        resp_buf_[1] = loc;
        resp_buf_[2] = seq;
        YGO_SetResponseb(pduel_, resp_buf_, 3);
      };
    } else if (msg_ == MSG_ANNOUNCE_NUMBER) {
      // B.3.a 3.5 port. Payload: [u8 player][u8 count][count × u64 number].
      // Each number becomes a LegalAction::number(n). response_ = i
      // (the index of the chosen number in the count-sized list), per
      // YGO_SetResponsei(idx) callback.
      auto player = read_u8();
      int count = read_u8();
      std::vector<int> numbers;
      for (int i = 0; i < count; ++i) {
        int number = compat_read<uint32_t, uint64_t>();
        if (number <= 0 || number > 12) {
          throw std::runtime_error("Number " + std::to_string(number) +
                                   " not implemented for announce number");
        }
        numbers.push_back(number);
        LegalAction la = LegalAction::number(static_cast<uint8_t>(number));
        la.msg_ = msg_;
        la.response_ = static_cast<uint32_t>(i);
        legal_actions_.push_back(la);
      }
      if (verbose_) {
        auto pl = players_[player];
        std::string str = "Select a number, one of: [";
        for (int i = 0; i < count; ++i) {
          str += std::to_string(numbers[i]);
          if (i < count - 1) {
            str += ", ";
          }
        }
        str += "]";
        pl->notify(str);
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };
    } else if (msg_ == MSG_ANNOUNCE_ATTRIB) {
      auto player = read_u8();
      int count = read_u8();
      auto flag = read_u32();

      int n_attrs = 7;

      std::vector<uint8_t> attrs;
      for (int i = 0; i < n_attrs; i++) {
        if (flag & (1 << i)) {
          attrs.push_back(i + 1);
        }
      }

      if (count != 1) {
        throw std::runtime_error("Announce attrib count " +
                                 std::to_string(count) + " not implemented");
      }

      if (verbose_) {
        auto pl = players_[player];
        pl->notify("Select " + std::to_string(count) +
                   " attributes separated by spaces:");
        for (int i = 0; i < attrs.size(); i++) {
          pl->notify(std::to_string(attrs[i]) + ": " +
                     attribute2str.at(1 << (attrs[i] - 1)));
        }
      }

      // B.3.a 3.5 port. Payload: [u8 player][u8 count][u32 flag]. The
      // wrapper restricts count == 1, so each LegalAction carries one
      // attribute bit (ATTRIBUTE_EARTH=1, ATTRIBUTE_WATER=2, ...). The
      // response_ is the POS_* value `1 << (attr_idx-1)` that the
      // callback computes from the option string.
      auto combs = combinations(attrs.size(), count);
      for (const auto &comb : combs) {
        uint32_t resp = 0;
        uint8_t primary_attr_val = 0;
        for (int j = 0; j < count; ++j) {
          int attr_idx = attrs[comb[j]];
          if (j == 0) primary_attr_val = static_cast<uint8_t>(1 << (attr_idx - 1));
          resp |= 1u << (attr_idx - 1);
        }
        LegalAction la =
            LegalAction::attribute(static_cast<int>(primary_attr_val));
        la.msg_ = msg_;
        la.response_ = resp;
        legal_actions_.push_back(la);
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };

    } else if (msg_ == MSG_SELECT_POSITION) {
      // B.3.a 3.5 port. Payload: [u8 player][u32 code][u8 valid_pos].
      // LegalAction mapping: each allowed position becomes a LegalAction
      // with position_ = POS_* bit, response_ = 1 << (pos_idx) where
      // pos_idx is the 0-3 bit position within valid_pos. The
      // callback's `1 << pos` encoding is preserved via response_.
      auto player = read_u8();
      auto code = read_u32();
      auto valid_pos = read_u8();
      auto cit = card_ids_.find(code);
      CardId cid = (cit != card_ids_.end()) ? cit->second : 0;

      if (verbose_) {
        auto pl = players_[player];
        auto card = c_get_card(code);
        pl->notify("Select position for " + card.name_ + ":");
      }

      int i = 1;
      uint8_t pos_bit_idx = 0;
      for (auto pos : {POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK,
                       POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE}) {
        if (valid_pos & pos) {
          LegalAction la;
          la.position_ = pos;
          la.cid_ = cid;
          la.msg_ = msg_;
          la.response_ = static_cast<uint32_t>(1 << pos_bit_idx);
          legal_actions_.push_back(la);

          if (verbose_) {
            auto pl = players_[player];
            pl->notify(fmt::format("{}: {}", i, position_to_string(pos)));
          }
        }
        i++;
        pos_bit_idx++;
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, static_cast<int32_t>(
            legal_actions_[idx].response_));
      };
    } else {
      show_deck(0);
      show_deck(1);
      throw std::runtime_error(
        fmt::format("Unknown message {}, length {}, dp {}",
        msg_, dl_, dp_));
    }
  }

  int GetSuitableReturn(uint32_t maxseq, uint32_t size) {
    using nl8 = std::numeric_limits<uint8_t>;
    using nl16 = std::numeric_limits<uint16_t>;
    using nl32 = std::numeric_limits<uint32_t>;
    if(maxseq < nl8::max()) {
      if(maxseq >= size * nl8::digits)
        return 2;
    } else if(maxseq < nl16::max()) {
      if(maxseq >= size * nl16::digits)
        return 1;
    }
    else if(maxseq < nl32::max()) {
      if(maxseq >= size * nl32::digits)
        return 0;
    }
    return 3;
  }

  void _damage(uint8_t player, uint32_t amount) {
    lp_[player] -= amount;
    if (verbose_) {
      auto lp = players_[player];
      lp->notify(fmt::format("Your lp decreased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp decreased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _recover(uint8_t player, uint32_t amount) {
    lp_[player] += amount;
    if (verbose_) {
      auto lp = players_[player];
      lp->notify(fmt::format("Your lp increased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp increased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _duel_end(uint8_t player, uint8_t reason) {
    winner_ = player;
    win_reason_ = reason;

    std::unique_lock<std::shared_timed_mutex> ulock(duel_mtx);
    YGO_EndDuel(pduel_);
    ulock.unlock();

    duel_started_ = false;
  }
};

using EDOProEnvPool = AsyncEnvPool<EDOProEnv>;

// B.3.a step-5: synthetic test harness for the obs:actions_ 12-col encoder.
// Constructs a controlled set of LegalActions, runs _set_obs_action on each
// into an (N, 12) uint8 buffer, and returns the raw bytes as a flat vector
// (row-major). The Python test reshapes to (N, 12) and asserts known
// field values at expected positions. Covers all handler families:
//   row 0: MSG_SELECT_IDLECMD activate (spec+cid, card-effect offset)
//   row 1: MSG_SELECT_IDLECMD phase=Battle ("b" → to battle phase)
//   row 2: MSG_SELECT_CARD finish marker
//   row 3: MSG_SELECT_CARD pick (spec_index=5)
//   row 4: MSG_SELECT_CHAIN activate_spec (spec+cid, system-string effect)
//   row 5: MSG_SELECT_CHAIN cancel
//   row 6: MSG_SELECT_EFFECTYN yes (activate)
//   row 7: MSG_SELECT_EFFECTYN no (cancel)
//   row 8: MSG_SELECT_POSITION (face-up defense)
//   row 9: MSG_SELECT_OPTION (option 3)
//   row 10: MSG_SELECT_PLACE (m3 → ActionPlace::MZone3)
//   row 11: MSG_SELECT_BATTLECMD attack
//   row 12: MSG_ANNOUNCE_NUMBER (number=7)
//   row 13: MSG_ANNOUNCE_ATTRIB (ATTRIBUTE_FIRE bit)
inline std::vector<uint8_t> test_action_encoder() {
  std::vector<LegalAction> las;

  // row 0 — IDLECMD Activate with card effect (eff_idx ≥ CARD_EFFECT_OFFSET)
  {
    LegalAction la = LegalAction::activate_spec(CARD_EFFECT_OFFSET + 3, "h2");
    la.msg_ = MSG_SELECT_IDLECMD;
    la.cid_ = 0x0205;        // high byte 0x02, low byte 0x05
    la.spec_index_ = 7;       // 7th card in obs:cards_ (1-based)
    las.push_back(la);
  }
  // row 1 — IDLECMD phase=Battle
  {
    LegalAction la = LegalAction::phase(ActionPhase::Battle);
    la.msg_ = MSG_SELECT_IDLECMD;
    las.push_back(la);
  }
  // row 2 — SELECT_CARD finish
  {
    LegalAction la = LegalAction::finish();
    la.msg_ = MSG_SELECT_CARD;
    las.push_back(la);
  }
  // row 3 — SELECT_CARD pick
  {
    LegalAction la = LegalAction::from_spec("h1");
    la.msg_ = MSG_SELECT_CARD;
    la.cid_ = 0x0108;
    la.spec_index_ = 5;
    las.push_back(la);
  }
  // row 4 — CHAIN activate with system-string effect
  {
    LegalAction la = LegalAction::activate_spec(30 /*system_string id*/, "s2");
    la.msg_ = MSG_SELECT_CHAIN;
    la.cid_ = 0x0301;
    la.spec_index_ = 11;
    las.push_back(la);
  }
  // row 5 — CHAIN cancel
  {
    LegalAction la = LegalAction::cancel();
    la.msg_ = MSG_SELECT_CHAIN;
    las.push_back(la);
  }
  // row 6 — EFFECTYN yes
  {
    LegalAction la = LegalAction::activate_spec(0 /*default*/, "m1");
    la.msg_ = MSG_SELECT_EFFECTYN;
    la.cid_ = 0x0410;
    la.spec_index_ = 3;
    las.push_back(la);
  }
  // row 7 — EFFECTYN no (cancel)
  {
    LegalAction la = LegalAction::cancel();
    la.msg_ = MSG_SELECT_EFFECTYN;
    la.spec_ = "m1";
    la.cid_ = 0x0410;
    la.spec_index_ = 3;
    las.push_back(la);
  }
  // row 8 — POSITION (face-up defense = POS_FACEUP_DEFENSE = 4)
  {
    LegalAction la;
    la.msg_ = MSG_SELECT_POSITION;
    la.cid_ = 0x0502;
    la.position_ = POS_FACEUP_DEFENSE;
    las.push_back(la);
  }
  // row 9 — OPTION 3
  {
    LegalAction la = LegalAction::activate_spec(1190 /*system_string*/, "");
    la.msg_ = MSG_SELECT_OPTION;
    la.response_ = 2;  // unused by encoder — encoder reads .act_ + .effect_
    las.push_back(la);
  }
  // row 10 — PLACE m3 → ActionPlace::MZone3
  {
    LegalAction la = LegalAction::from_spec("m3");
    la.msg_ = MSG_SELECT_PLACE;
    la.place_ = ActionPlace::MZone3;
    las.push_back(la);
  }
  // row 11 — BATTLECMD attack
  {
    LegalAction la = LegalAction::act_spec(ActionAct::Attack, "m1");
    la.msg_ = MSG_SELECT_BATTLECMD;
    la.cid_ = 0x0601;
    la.spec_index_ = 4;
    las.push_back(la);
  }
  // row 12 — ANNOUNCE_NUMBER 7
  {
    LegalAction la = LegalAction::number(7);
    la.msg_ = MSG_ANNOUNCE_NUMBER;
    las.push_back(la);
  }
  // row 13 — ANNOUNCE_ATTRIB fire bit
  {
    LegalAction la = LegalAction::attribute(ATTRIBUTE_FIRE);
    la.msg_ = MSG_ANNOUNCE_ATTRIB;
    las.push_back(la);
  }

  constexpr int kNCols = 12;
  const int n = static_cast<int>(las.size());
  // Build a TArray-compatible buffer. EDOProEnv's static helpers take a
  // TArray<uint8_t>&; here we synthesize a minimal one via Array on a
  // heap-backed contiguous buffer.
  Array arr(ShapeSpec(sizeof(uint8_t), {n, kNCols}));
  TArray<uint8_t> feat(arr);
  for (int i = 0; i < n; ++i) {
    EDOProEnv::_set_obs_action(feat, i, las[i]);
  }

  std::vector<uint8_t> bytes(static_cast<size_t>(n) * kNCols);
  std::memcpy(bytes.data(), arr.Data(), bytes.size());
  return bytes;
}

} // namespace edopro

#endif // YGOENV_EDOPro_EDOPro_H_
