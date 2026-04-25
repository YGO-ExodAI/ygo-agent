// card_embedding_store.h — C++ loader for the pre-baked embeddings +
// annotations blob produced by ExodAI_ml/bake_c_encoder_data.py.
//
// File layout (see the bake script for the canonical spec):
//
//   Header (64 bytes, little-endian, zero-padded):
//     magic              8 bytes  'EAIENC01'
//     num_cards          u32
//     emb_dim            u32       = 256
//     ann_dim            u32       = 465
//     annotations_sha256 32 bytes  (hash of annotations.json content)
//     reserved           padding to 64
//
//   Code table (num_cards × 4 bytes, sorted ascending):
//     konami_code uint32
//
//   Data table (num_cards records, each (emb_dim + ann_dim) × 4 bytes):
//     embedding  float32 × 256
//     annotation float32 × 465
//
// At init time we read the whole file into a single heap allocation and
// build an `unordered_dense::map<CardCode, uint32>` from code to record
// index. All subsequent lookups are O(1) and return raw pointers into
// the loaded buffer (zero-copy, stable for the lifetime of the process).
//
// Alt-art alias resolution happens at bake time — alt-art codes and
// their canonical codes share the same record bytes.

#ifndef CARD_EMBEDDING_STORE_H_
#define CARD_EMBEDDING_STORE_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ankerl/unordered_dense.h>
#include <fmt/core.h>

namespace exodai {

class CardEmbeddingStore {
 public:
  static constexpr uint32_t kEmbDim = 256;
  static constexpr uint32_t kAnnDim = 465;
  static constexpr uint32_t kHeaderSize = 64;
  static constexpr const char* kMagic = "EAIENC01";  // 8 bytes

  using CardCode = uint32_t;

  // Load the blob from `path`. Throws if the file is missing, the magic
  // is wrong, or the dimensions don't match the compile-time constants.
  // Second load from the same path is a no-op — idempotent.
  static const CardEmbeddingStore& load(const std::string& path) {
    if (instance_ != nullptr) {
      return *instance_;
    }
    instance_ = new CardEmbeddingStore(path);
    return *instance_;
  }

  // Accessor for the process-wide singleton after load().
  static const CardEmbeddingStore& get() {
    if (instance_ == nullptr) {
      throw std::runtime_error(
          "CardEmbeddingStore::get() called before load()");
    }
    return *instance_;
  }

  static bool is_loaded() { return instance_ != nullptr; }

  // Number of cards in the blob.
  uint32_t num_cards() const { return num_cards_; }

  // SHA256 of annotations.json at bake time (32 bytes, hex-encoded).
  const std::array<uint8_t, 32>& annotations_sha256() const {
    return ann_sha256_;
  }

  // O(1) pointer-to-data lookup. Returns nullptr if `code` isn't in the
  // blob — caller must fall back to zero_embedding()/zero_annotation().
  const float* embedding(CardCode code) const {
    auto it = code_to_index_.find(code);
    if (it == code_to_index_.end()) {
      return nullptr;
    }
    return record_embedding(it->second);
  }

  const float* annotation(CardCode code) const {
    auto it = code_to_index_.find(code);
    if (it == code_to_index_.end()) {
      return nullptr;
    }
    return record_annotation(it->second);
  }

  // Pointer to a persistent all-zero float array. Use as a fallback for
  // unknown cards so callers can always write out[i] = ptr[i] without
  // null checks.
  const float* zero_embedding() const { return zero_emb_.data(); }
  const float* zero_annotation() const { return zero_ann_.data(); }

  // Mean-of-non-zero-embeddings — mirrors
  // EmbeddingService.get_mean() semantics:
  //
  //   1. If codes is empty, write zeros and return.
  //   2. Look up each code; cards with zero embeddings are skipped
  //      (they don't contribute to either the numerator or denominator).
  //   3. If no cards have non-zero embeddings, write zeros and return.
  //   4. Otherwise, write the element-wise mean of all non-zero
  //      embeddings.
  //
  // Accumulation order is plain left-to-right, matching numpy's
  // behavior for 1D sums. Writes kEmbDim floats into `out`.
  void mean_embedding(const std::vector<CardCode>& codes, float* out) const {
    std::memset(out, 0, kEmbDim * sizeof(float));
    if (codes.empty()) return;

    int contributing = 0;
    for (CardCode code : codes) {
      const float* emb = embedding(code);
      if (emb == nullptr) {
        continue;
      }
      // Skip zero-vectors (EmbeddingService.get_mean ignores them so that
      // cards with missing data don't drag the mean toward zero).
      bool any_nonzero = false;
      for (uint32_t i = 0; i < kEmbDim; ++i) {
        if (emb[i] != 0.0f) { any_nonzero = true; break; }
      }
      if (!any_nonzero) {
        continue;
      }
      for (uint32_t i = 0; i < kEmbDim; ++i) {
        out[i] += emb[i];
      }
      ++contributing;
    }
    if (contributing == 0) {
      std::memset(out, 0, kEmbDim * sizeof(float));
      return;
    }
    const float inv = 1.0f / static_cast<float>(contributing);
    for (uint32_t i = 0; i < kEmbDim; ++i) {
      out[i] *= inv;
    }
  }

 private:
  explicit CardEmbeddingStore(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      throw std::runtime_error(
          fmt::format("CardEmbeddingStore: cannot open {}", path));
    }
    f.seekg(0, std::ios::end);
    const std::streamsize file_size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (file_size < static_cast<std::streamsize>(kHeaderSize)) {
      throw std::runtime_error(
          fmt::format("CardEmbeddingStore: {} is too small ({} bytes)",
                      path, file_size));
    }

    blob_.resize(static_cast<size_t>(file_size));
    f.read(reinterpret_cast<char*>(blob_.data()), file_size);
    if (!f) {
      throw std::runtime_error(
          fmt::format("CardEmbeddingStore: failed to read {}", path));
    }

    // Parse header.
    if (std::memcmp(blob_.data(), kMagic, 8) != 0) {
      throw std::runtime_error(
          fmt::format("CardEmbeddingStore: bad magic in {}", path));
    }
    num_cards_ = read_u32(blob_.data() + 8);
    uint32_t emb_dim = read_u32(blob_.data() + 12);
    uint32_t ann_dim = read_u32(blob_.data() + 16);
    if (emb_dim != kEmbDim || ann_dim != kAnnDim) {
      throw std::runtime_error(fmt::format(
          "CardEmbeddingStore: dim mismatch in {} (emb={} ann={}, expected "
          "{}/{})",
          path, emb_dim, ann_dim, kEmbDim, kAnnDim));
    }
    std::memcpy(ann_sha256_.data(), blob_.data() + 20, 32);

    // Verify the file is large enough for the declared record count.
    const size_t code_table_bytes = num_cards_ * sizeof(uint32_t);
    const size_t record_bytes = (kEmbDim + kAnnDim) * sizeof(float);
    const size_t data_bytes = num_cards_ * record_bytes;
    const size_t expected_size = kHeaderSize + code_table_bytes + data_bytes;
    if (blob_.size() < expected_size) {
      throw std::runtime_error(fmt::format(
          "CardEmbeddingStore: {} truncated (got {} bytes, expected {})",
          path, blob_.size(), expected_size));
    }

    code_table_offset_ = kHeaderSize;
    data_table_offset_ = kHeaderSize + code_table_bytes;

    // Build the code → record-index hashmap. This is the one allocation
    // we do beyond the blob itself; after this, lookups are pure
    // pointer arithmetic.
    code_to_index_.reserve(num_cards_);
    const uint8_t* code_ptr = blob_.data() + code_table_offset_;
    for (uint32_t i = 0; i < num_cards_; ++i) {
      const uint32_t code = read_u32(code_ptr + i * sizeof(uint32_t));
      code_to_index_.emplace(code, i);
    }
  }

  // Single-process singleton. The training loop calls load() once at
  // init; everyone else calls get().
  static CardEmbeddingStore* instance_;

  static uint32_t read_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
  }

  const float* record_embedding(uint32_t index) const {
    const size_t record_bytes = (kEmbDim + kAnnDim) * sizeof(float);
    return reinterpret_cast<const float*>(
        blob_.data() + data_table_offset_ + index * record_bytes);
  }

  const float* record_annotation(uint32_t index) const {
    return record_embedding(index) + kEmbDim;
  }

  std::vector<uint8_t> blob_;
  uint32_t num_cards_ = 0;
  size_t code_table_offset_ = 0;
  size_t data_table_offset_ = 0;
  ankerl::unordered_dense::map<CardCode, uint32_t> code_to_index_;
  std::array<uint8_t, 32> ann_sha256_{};
  std::array<float, kEmbDim> zero_emb_{};
  std::array<float, kAnnDim> zero_ann_{};
};

// Definition of the singleton pointer — one translation unit owns it.
inline CardEmbeddingStore* CardEmbeddingStore::instance_ = nullptr;

}  // namespace exodai

#endif  // CARD_EMBEDDING_STORE_H_
