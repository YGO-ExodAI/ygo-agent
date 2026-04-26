/*
 * Copyright 2021 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef YGOENV_CORE_PY_ENVPOOL_H_
#define YGOENV_CORE_PY_ENVPOOL_H_

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ygoenv/core/envpool.h"

namespace py = pybind11;

/**
 * Convert Array to py::array, with py::capsule
 */
template <typename dtype>
struct ArrayToNumpyHelper {
  static py::array Convert(const Array& a) {
    auto* ptr = new std::shared_ptr<char>(a.SharedPtr());
    auto capsule = py::capsule(ptr, [](void* ptr) {
      delete reinterpret_cast<std::shared_ptr<char>*>(ptr);
    });
    return py::array(a.Shape(), reinterpret_cast<dtype*>(a.Data()), capsule);
  }
};

template <typename dtype>
struct ArrayToNumpyHelper<Container<dtype>> {
  using UniquePtr = Container<dtype>;
  static py::array Convert(const Array& a) {
    auto* ptr_arr = reinterpret_cast<UniquePtr*>(a.Data());
    auto* ptr =
        new std::unique_ptr<py::object[]>(new py::object[a.size]);  // NOLINT
    auto capsule = py::capsule(ptr, [](void* ptr) {
      delete reinterpret_cast<std::unique_ptr<py::object[]>*>(ptr);  // NOLINT
    });
    for (std::size_t i = 0; i < a.size; ++i) {
      auto* inner_ptr = new UniquePtr(std::move(ptr_arr[i]));
      (ptr_arr + i)->~UniquePtr();
      auto capsule = py::capsule(inner_ptr, [](void* inner_ptr) {
        delete reinterpret_cast<UniquePtr*>(inner_ptr);
      });
      if (*inner_ptr == nullptr) {
        (*ptr)[i] = py::none();
      } else {
        (*ptr)[i] =
            py::array((*inner_ptr)->Shape(),
                      reinterpret_cast<dtype*>((*inner_ptr)->Data()), capsule);
      }
    }
    return {py::dtype("object"), a.Shape(),
            reinterpret_cast<py::object*>(ptr->get()), capsule};
  }
};

template <typename dtype>
Array NumpyToArray(const py::array& arr) {
  using ArrayT = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
  ArrayT arr_t(arr);
  ShapeSpec spec(arr_t.itemsize(),
                 std::vector<int>(arr_t.shape(), arr_t.shape() + arr_t.ndim()));
  return {spec, reinterpret_cast<char*>(arr_t.mutable_data())};
}

template <typename dtype>
Array NumpyToArrayIncRef(const py::array& arr) {
  using ArrayT = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
  auto* arr_ptr = new ArrayT(arr);
  ShapeSpec spec(
      arr_ptr->itemsize(),
      std::vector<int>(arr_ptr->shape(), arr_ptr->shape() + arr_ptr->ndim()));
  return Array(spec, reinterpret_cast<char*>(arr_ptr->mutable_data()),
               [arr_ptr](char* p) {
                 py::gil_scoped_acquire acquire;
                 delete arr_ptr;
               });
}

template <typename Spec>
struct SpecTupleHelper {
  static decltype(auto) Make(const Spec& spec) {
    return std::make_tuple(py::dtype::of<typename Spec::dtype>(), spec.shape,
                           spec.bounds, spec.elementwise_bounds);
  }
};

/**
 * For Container type, it is converted a numpy array of numpy array.
 * The spec itself describes the shape of the outer array, the inner_spec
 * contains the spec of the inner array.
 * Therefore the shape returned to python side has the format
 * (outer_shape, inner_shape).
 */
template <typename dtype>
struct SpecTupleHelper<Spec<Container<dtype>>> {
  static decltype(auto) Make(const Spec<Container<dtype>>& spec) {
    return std::make_tuple(py::dtype::of<dtype>(),
                           std::make_tuple(spec.shape, spec.inner_spec.shape),
                           spec.inner_spec.bounds,
                           spec.inner_spec.elementwise_bounds);
  }
};

template <typename... Spec>
decltype(auto) ExportSpecs(const std::tuple<Spec...>& specs) {
  return std::apply(
      [&](auto&&... spec) {
        return std::make_tuple(SpecTupleHelper<Spec>::Make(spec)...);
      },
      specs);
}

template <typename EnvSpec>
class PyEnvSpec : public EnvSpec {
 public:
  using StateSpecT =
      decltype(ExportSpecs(std::declval<typename EnvSpec::StateSpec>()));
  using ActionSpecT =
      decltype(ExportSpecs(std::declval<typename EnvSpec::ActionSpec>()));

  StateSpecT py_state_spec;
  ActionSpecT py_action_spec;
  typename EnvSpec::ConfigValues py_config_values;
  static std::vector<std::string> py_config_keys;
  static std::vector<std::string> py_state_keys;
  static std::vector<std::string> py_action_keys;
  static typename EnvSpec::ConfigValues py_default_config_values;

  explicit PyEnvSpec(const typename EnvSpec::ConfigValues& conf)
      : EnvSpec(conf),
        py_state_spec(ExportSpecs(EnvSpec::state_spec)),
        py_action_spec(ExportSpecs(EnvSpec::action_spec)),
        py_config_values(EnvSpec::config.AllValues()) {}
};
template <typename EnvSpec>
std::vector<std::string> PyEnvSpec<EnvSpec>::py_config_keys =
    EnvSpec::Config::AllKeys();
template <typename EnvSpec>
std::vector<std::string> PyEnvSpec<EnvSpec>::py_state_keys =
    EnvSpec::StateSpec::AllKeys();
template <typename EnvSpec>
std::vector<std::string> PyEnvSpec<EnvSpec>::py_action_keys =
    EnvSpec::ActionSpec::AllKeys();
template <typename EnvSpec>
typename EnvSpec::ConfigValues PyEnvSpec<EnvSpec>::py_default_config_values =
    EnvSpec::kDefaultConfig.AllValues();

/**
 * Bind specs to arrs, and return py::array in ret
 */
template <typename... Spec>
void ToNumpy(const std::vector<Array>& arrs, const std::tuple<Spec...>& specs,
             std::vector<py::array>* ret) {
  std::size_t index = 0;
  std::apply(
      [&](auto&&... spec) {
        (ret->emplace_back(
             ArrayToNumpyHelper<typename Spec::dtype>::Convert(arrs[index++])),
         ...);
      },
      specs);
}

template <typename... Spec>
void ToArray(const std::vector<py::array>& py_arrs,
             const std::tuple<Spec...>& specs, std::vector<Array>* ret) {
  std::size_t index = 0;
  std::apply(
      [&](auto&&... spec) {
        (ret->emplace_back(
             NumpyToArrayIncRef<typename Spec::dtype>(py_arrs[index++])),
         ...);
      },
      specs);
}

/**
 * Templated subclass of EnvPool,
 * to be overrided by the real EnvPool.
 */
template <typename EnvPool>
class PyEnvPool : public EnvPool {
 public:
  using PySpec = PyEnvSpec<typename EnvPool::Spec>;

  PySpec py_spec;
  static std::vector<std::string> py_state_keys;
  static std::vector<std::string> py_action_keys;

  explicit PyEnvPool(const PySpec& py_spec)
      : EnvPool(py_spec), py_spec(py_spec) {}

  /**
   * py api
   */
  void PySend(const std::vector<py::array>& action) {
    std::vector<Array> arr;
    arr.reserve(action.size());
    ToArray(action, py_spec.action_spec, &arr);
    py::gil_scoped_release release;
    EnvPool::Send(arr);  // delegate to the c++ api
  }

  /**
   * py api
   */
  std::vector<py::array> PyRecv() {
    std::vector<Array> arr;
    {
      py::gil_scoped_release release;
      arr = EnvPool::Recv();
      DCHECK_EQ(arr.size(), std::tuple_size_v<typename EnvPool::State::Keys>);
    }
    std::vector<py::array> ret;
    ret.reserve(EnvPool::State::kSize);
    ToNumpy(arr, py_spec.state_spec, &ret);
    return ret;
  }

  /**
   * py api
   */
  void PyReset(const py::array& env_ids) {
    // PyArray arr = PyArray::From<int>(env_ids);
    auto arr = NumpyToArrayIncRef<int>(env_ids);
    py::gil_scoped_release release;
    EnvPool::Reset(arr);
  }

  // Phase P1 Primitive 1, Chunk 7: per-env save/load passthrough.
  //
  // PySaveState reaches into AsyncEnvPool::envs_[env_id] and forwards
  // to the env's SaveState(). Returns the protobuf blob as py::bytes.
  // PyLoadState forwards a blob to envs_[env_id]->LoadState().
  //
  // These bypass the Send/Recv message-passing path because save/load
  // are infrequent control-plane operations, not per-step actions.
  // The env's own SaveState/LoadState handle the core_mutex bracketing
  // — no separate locking needed at this layer beyond GIL release.
  //
  // Caller responsibility: don't call save/load on an env_id that has
  // a step in flight. Typical usage drains via Recv first.
  py::bytes PySaveState(int env_id) {
    if (env_id < 0 || static_cast<std::size_t>(env_id) >= this->envs_.size()) {
      throw std::runtime_error("PySaveState: env_id out of range");
    }
    std::string blob;
    {
      py::gil_scoped_release release;
      blob = this->envs_[env_id]->SaveState();
    }
    return py::bytes(blob);
  }

  void PyLoadState(int env_id, const py::bytes& blob) {
    if (env_id < 0 || static_cast<std::size_t>(env_id) >= this->envs_.size()) {
      throw std::runtime_error("PyLoadState: env_id out of range");
    }
    std::string blob_str = blob;  // implicit py::bytes -> std::string
    py::gil_scoped_release release;
    this->envs_[env_id]->LoadState(blob_str);
  }

  // Phase P1 Primitive 1, Chunk 7 (B2): republish the env's current obs
  // into the state buffer queue without taking a step. Required between
  // _load_state and the next _recv() — without it, the recv returns
  // stale obs from the pre-load game state and rollout_from_state's
  // obs_sync_fn falls back to the destructive reset() path.
  //
  // Caller responsibility (mirrors save/load): don't call on an env_id
  // that has a step in flight. Order is -1 (async slot allocation), so
  // single-env use cases don't need to coordinate ordering with peers.
  void PyPublishObs(int env_id) {
    if (env_id < 0 || static_cast<std::size_t>(env_id) >= this->envs_.size()) {
      throw std::runtime_error("PyPublishObs: env_id out of range");
    }
    py::gil_scoped_release release;
    this->envs_[env_id]->EnvPublishObs(this->state_buffer_queue_.get(), -1);
    // Keep stepping_env_num_ in sync with the slot we just published.
    // Without this, the next Recv() takes the partial-batch drain branch
    // (additional_wait > 0), which overshoots StateBuffer::done_count_
    // and bumps alloc_count_ ahead — desynchronizing the alloc/done
    // pointers so the *next* envs.step() reads from an empty queue slot
    // and returns Truncate(0). See src/docs/p5/index-error-characterization.md.
    if (this->is_sync_) {
      this->stepping_env_num_++;
    }
  }

  // Returns the konami codes of all cards present in the env's
  // current duel state. Used by Chunk 7's script-corpus hash to
  // know which c<id>.lua files contributed to the saved state.
  std::vector<uint32_t> PyGetStateCardCodes(int env_id) {
    if (env_id < 0 || static_cast<std::size_t>(env_id) >= this->envs_.size()) {
      throw std::runtime_error("PyGetStateCardCodes: env_id out of range");
    }
    std::vector<uint32_t> codes;
    {
      py::gil_scoped_release release;
      codes = this->envs_[env_id]->GetStateCardCodes();
    }
    return codes;
  }

  // Phase P1 Primitive 1, Chunk 9b followup (Issue 1, option B2):
  // expose env-adapter pending-message state for the JSON sidecar.
  //
  // PyGetPendingMessageBytes — wire-format bytes of the most-recent
  //   handle_message() frame (the one that built the current
  //   legal_actions_). Returned as py::bytes so the Python sidecar
  //   layer can b64-encode and persist it. Empty if no message has
  //   been parsed yet.
  // PyGetFieldReturns — engine field.returns ProgressiveBuffer.data
  //   snapshot. Returned as py::bytes for symmetry. Not in the proto.
  // PySetPendingMessageState — atomic restore of both fields after
  //   LoadState. Re-runs handle_message() internally to repopulate
  //   legal_actions_/msg_/to_play_/callback_ from the saved bytes.
  py::bytes PyGetPendingMessageBytes(int env_id) {
    if (env_id < 0 || static_cast<std::size_t>(env_id) >= this->envs_.size()) {
      throw std::runtime_error("PyGetPendingMessageBytes: env_id out of range");
    }
    std::vector<uint8_t> bytes;
    {
      py::gil_scoped_release release;
      bytes = this->envs_[env_id]->GetPendingMessageBytes();
    }
    return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }

  py::bytes PyGetFieldReturns(int env_id) {
    if (env_id < 0 || static_cast<std::size_t>(env_id) >= this->envs_.size()) {
      throw std::runtime_error("PyGetFieldReturns: env_id out of range");
    }
    std::vector<uint8_t> bytes;
    {
      py::gil_scoped_release release;
      bytes = this->envs_[env_id]->GetFieldReturns();
    }
    return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }

  void PySetPendingMessageState(int env_id,
                                const py::bytes& msg_bytes,
                                const py::bytes& field_returns) {
    if (env_id < 0 || static_cast<std::size_t>(env_id) >= this->envs_.size()) {
      throw std::runtime_error("PySetPendingMessageState: env_id out of range");
    }
    std::string msg_str = msg_bytes;
    std::string returns_str = field_returns;
    std::vector<uint8_t> msg_vec(msg_str.begin(), msg_str.end());
    std::vector<uint8_t> ret_vec(returns_str.begin(), returns_str.end());
    py::gil_scoped_release release;
    this->envs_[env_id]->SetPendingMessageState(msg_vec, ret_vec);
  }
};

template <typename EnvPool>
std::vector<std::string> PyEnvPool<EnvPool>::py_state_keys =
    PyEnvPool<EnvPool>::PySpec::py_state_keys;
template <typename EnvPool>
std::vector<std::string> PyEnvPool<EnvPool>::py_action_keys =
    PyEnvPool<EnvPool>::PySpec::py_action_keys;

py::object abc_meta = py::module::import("abc").attr("ABCMeta");

/**
 * Call this macro in the translation unit of each envpool instance
 * It will register the envpool instance to the registry.
 * The static bool status is local to the translation unit.
 */
#define REGISTER(MODULE, SPEC, ENVPOOL)                              \
  py::class_<SPEC>(MODULE, "_" #SPEC, py::metaclass(abc_meta))       \
      .def(py::init<const typename SPEC::ConfigValues&>())           \
      .def_readonly("_config_values", &SPEC::py_config_values)       \
      .def_readonly("_state_spec", &SPEC::py_state_spec)             \
      .def_readonly("_action_spec", &SPEC::py_action_spec)           \
      .def_readonly_static("_state_keys", &SPEC::py_state_keys)      \
      .def_readonly_static("_action_keys", &SPEC::py_action_keys)    \
      .def_readonly_static("_config_keys", &SPEC::py_config_keys)    \
      .def_readonly_static("_default_config_values",                 \
                           &SPEC::py_default_config_values);         \
  py::class_<ENVPOOL>(MODULE, "_" #ENVPOOL, py::metaclass(abc_meta)) \
      .def(py::init<const SPEC&>())                                  \
      .def_readonly("_spec", &ENVPOOL::py_spec)                      \
      .def("_recv", &ENVPOOL::PyRecv)                                \
      .def("_send", &ENVPOOL::PySend)                                \
      .def("_reset", &ENVPOOL::PyReset)                              \
      .def("_save_state", &ENVPOOL::PySaveState)                     \
      .def("_load_state", &ENVPOOL::PyLoadState)                     \
      .def("_publish_obs", &ENVPOOL::PyPublishObs)                   \
      .def("_get_state_card_codes", &ENVPOOL::PyGetStateCardCodes)   \
      .def("_get_pending_message_bytes",                             \
           &ENVPOOL::PyGetPendingMessageBytes)                       \
      .def("_get_field_returns", &ENVPOOL::PyGetFieldReturns)        \
      .def("_set_pending_message_state",                             \
           &ENVPOOL::PySetPendingMessageState)                       \
      .def_readonly_static("_state_keys", &ENVPOOL::py_state_keys)   \
      .def_readonly_static("_action_keys",                           \
                           &ENVPOOL::py_action_keys);                \

#endif  // YGOENV_CORE_PY_ENVPOOL_H_
