/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_
#define _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_

#include "lib/charset/ob_charset.h"

#include <cstdint>

namespace oceanbase
{
namespace storage
{
typedef int32_t ObFTWordCode;
typedef int32_t ObFTWordBase;
typedef uint32_t ObFTWordStateIndex;

/**
 * @class ObFTSingleWord
 * @brief sturct to store a single character of a charset;
 */
struct ObFTSingleWord
{
public:
  ObFTSingleWord() : word(""), word_len(0) {}
  ObFTSingleWord(const ObFTSingleWord &other) = default;
  ObFTSingleWord &operator=(const ObFTSingleWord &other) = default;

  // These accessors are called once per trie step and once per hash map probe,
  // i.e. several times per input character. Out of line they cost a call plus a
  // spilled/returned ObString for a 1..4 byte comparison, so keep them here.
  int32_t set_word(const char *word, int32_t word_len)
  {
    int ret = OB_SUCCESS;
    if (OB_ISNULL(word)) {
      ret = OB_INVALID_ARGUMENT;
    } else if (word_len > ObCharset::MAX_MB_LEN) {
      ret = OB_SIZE_OVERFLOW;
    } else {
      memcpy(this->word, word, word_len);
      this->word_len = static_cast<uint8_t>(word_len);
    }
    return ret;
  }
  ObString get_word() const { return ObString(word_len, word); }
  bool operator==(const ObFTSingleWord &other) const
  {
    return (this == &other)
           || (word_len == other.word_len && 0 == memcmp(word, other.word, word_len));
  }

public:
  char word[common::ObCharset::MAX_MB_LEN];
  uint8_t word_len;
} __attribute__((packed));

enum class ObFTDictType : uint32_t
{
  DICT_TYPE_INVALID = 0,
  DICT_IK_MAIN = 1,
  DICT_IK_QUAN = 2,
  DICT_IK_STOP = 3,
};

class ObFTDictDesc
{
public:
  ObFTDictDesc(const ObString &name,
               const ObFTDictType type,
               const ObCharsetType charset,
               const ObCollationType coll_type)
      : name_(name), type_(type), charset_(charset), coll_type_(coll_type)
  {
  }

public:
  ObString name_;
  ObFTDictType type_;
  ObCharsetType charset_;
  ObCollationType coll_type_;
};

} //  namespace storage
} //  namespace oceanbase

#endif // _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_
