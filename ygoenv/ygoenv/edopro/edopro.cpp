#include "ygoenv/edopro/edopro.h"
#include "ygoenv/core/py_envpool.h"

using EDOProEnvSpec = PyEnvSpec<edopro::EDOProEnvSpec>;
using EDOProEnvPool = PyEnvPool<edopro::EDOProEnvPool>;

PYBIND11_MODULE(edopro_ygoenv, m) {
  REGISTER(m, EDOProEnvSpec, EDOProEnvPool)

  m.def("init_module", &edopro::init_module);
  // B.3.a Chunk 3.7: load the pre-baked card embedding + annotation store
  // used by the state encoder. Must be called before the first env step
  // for obs:state_ to be meaningful.
  m.def("init_embeddings_store", &edopro::init_embeddings_store,
        "Load pre-baked card embeddings + annotations for the state encoder.");
  // B.3.a synthetic test for YGO_NewCard owner/playerid semantics.
  // Exercised only by tests/test_b3a_new_card_semantics.py — never by
  // training or serving paths. Left in the module so the fix is
  // regression-protected: if anyone reverts YGO_NewCard to the old
  // info.team=playerid mapping, the test fails immediately.
  m.def("_test_new_card_semantics", &edopro::test_new_card_semantics,
        "Create a duel, OCG_DuelNewCard with asymmetric owner/playerid, "
        "query QUERY_OWNER. Returns observed owner seat.");
  // B.3.a step-5: synthetic test for the 12-col obs:actions_ encoder.
  // Returns the encoded byte matrix for a known set of LegalActions so
  // the Python test can assert field values at expected positions.
  m.def("_test_action_encoder", []() {
    auto bytes = edopro::test_action_encoder();
    return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }, "Run the 12-col obs:actions_ encoder on a fixed LegalAction set and "
     "return the raw byte matrix.");
  // B.5 synthetic test — dump Card struct fields for a given code so the
  // Python parity test can diff them against direct sqlite3 reads of
  // cards.cdb. Must be called after init_module.
  py::class_<edopro::CardSnapshot>(m, "_CardSnapshot")
    .def_readonly("code", &edopro::CardSnapshot::code)
    .def_readonly("alias", &edopro::CardSnapshot::alias)
    .def_readonly("setcodes", &edopro::CardSnapshot::setcodes)
    .def_readonly("type", &edopro::CardSnapshot::type)
    .def_readonly("level", &edopro::CardSnapshot::level)
    .def_readonly("lscale", &edopro::CardSnapshot::lscale)
    .def_readonly("rscale", &edopro::CardSnapshot::rscale)
    .def_readonly("attack", &edopro::CardSnapshot::attack)
    .def_readonly("defense", &edopro::CardSnapshot::defense)
    .def_readonly("race", &edopro::CardSnapshot::race)
    .def_readonly("attribute", &edopro::CardSnapshot::attribute)
    .def_readonly("link_marker", &edopro::CardSnapshot::link_marker)
    .def_readonly("name", &edopro::CardSnapshot::name)
    .def_readonly("desc", &edopro::CardSnapshot::desc);
  m.def("_test_card_struct", &edopro::test_card_struct,
        "Return the Card struct fields for `code` (from the wrapper's "
        "cards_ cache populated by init_module). Used by the B.5 "
        "card-feature parity test.");
}
