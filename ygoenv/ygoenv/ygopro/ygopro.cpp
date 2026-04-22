#include "ygoenv/ygopro/ygopro.h"
#include "ygoenv/core/py_envpool.h"

using YGOProEnvSpec = PyEnvSpec<ygopro::YGOProEnvSpec>;
using YGOProEnvPool = PyEnvPool<ygopro::YGOProEnvPool>;

PYBIND11_MODULE(ygopro_ygoenv, m) {
  REGISTER(m, YGOProEnvSpec, YGOProEnvPool)

  m.def("init_module", &ygopro::init_module);
  m.def("init_embeddings_store", &ygopro::init_embeddings_store,
        "Load the pre-baked card embedding + annotation blob (see "
        "ExodAI_ml/bake_c_encoder_data.py).");
}
