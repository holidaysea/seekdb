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

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_embed.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/json/ob_json_helper.h"
#include "share/rc/ob_server_runtime.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{

namespace
{
// Scans one JSON number token starting at `pos`.
// Returns true when the token is a plain decimal number (no strings, no nesting).
bool scan_json_number(const char *text, const int64_t length, int64_t &pos)
{
  bool ok = true;
  int64_t digits = 0;
  if (OB_UNLIKELY(pos >= length) || nullptr == text) {
    ok = false;
  } else {
    if ('-' == text[pos]) {
      ++pos;
    }
    while (pos < length && text[pos] >= '0' && text[pos] <= '9') {
      ++pos;
      ++digits;
    }
    if (pos < length && '.' == text[pos]) {
      ++pos;
      while (pos < length && text[pos] >= '0' && text[pos] <= '9') {
        ++pos;
        ++digits;
      }
    }
    if (0 == digits) {
      ok = false;
    } else if (pos < length && ('e' == text[pos] || 'E' == text[pos])) {
      ++pos;
      if (pos < length && ('+' == text[pos] || '-' == text[pos])) {
        ++pos;
      }
      int64_t exp_digits = 0;
      while (pos < length && text[pos] >= '0' && text[pos] <= '9') {
        ++pos;
        ++exp_digits;
      }
      if (0 == exp_digits) {
        ok = false;
      }
    }
  }
  return ok;
}

// Validates the embedding payload returned by the model endpoint.
// The payload must be a non empty JSON array of plain numbers whose length
// matches the requested dimension (when one was given).
int validate_dense_embedding(const ObString &embedding_str, const int64_t expected_dimension)
{
  int ret = OB_SUCCESS;
  const char *text = embedding_str.ptr();
  const int64_t length = embedding_str.length();
  int64_t pos = 0;
  int64_t element_count = 0;
  if (OB_UNLIKELY(embedding_str.empty()) || OB_ISNULL(text)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_USER_ERROR(OB_ERR_UNEXPECTED, "ai_embed, model returned an empty embedding");
  } else {
    for (; pos < length && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'); ++pos) {
    }
    if (pos >= length || '[' != text[pos]) {
      ret = OB_ERR_UNEXPECTED;
      LOG_USER_ERROR(OB_ERR_UNEXPECTED, "ai_embed, embedding result must be a json array");
    } else {
      ++pos;
      bool expect_value = true;
      bool closed = false;
      while (OB_SUCC(ret) && pos < length) {
        const char ch = text[pos];
        if (' ' == ch || '\t' == ch || '\n' == ch || '\r' == ch || ',' == ch) {
          ++pos;
        } else if (']' == ch) {
          ++pos;
          closed = true;
          break;
        } else if (!expect_value || !scan_json_number(text, length, pos)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_USER_ERROR(OB_ERR_UNEXPECTED, "ai_embed, embedding vector must contain numbers only");
        } else {
          ++element_count;
          expect_value = false;
        }
      }
      for (; OB_SUCC(ret) && pos < length; ++pos) {
        const char ch = text[pos];
        if (' ' != ch && '\t' != ch && '\n' != ch && '\r' != ch) {
          ret = OB_ERR_UNEXPECTED;
          LOG_USER_ERROR(OB_ERR_UNEXPECTED, "ai_embed, model returned an invalid embedding json");
        }
      }
      if (OB_FAIL(ret)) {
      } else if (!closed) {
        ret = OB_ERR_UNEXPECTED;
        LOG_USER_ERROR(OB_ERR_UNEXPECTED, "ai_embed, model returned an invalid embedding json");
      } else if (0 == element_count) {
        ret = OB_ERR_UNEXPECTED;
        LOG_USER_ERROR(OB_ERR_UNEXPECTED, "ai_embed, embedding vector must not be empty");
      } else if (expected_dimension > 0 && element_count != expected_dimension) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT,
                       "ai_embed, embedding vector size does not match the requested dimension");
      }
    }
  }
  return ret;
}
} // namespace
ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                        T_FUN_SYS_AI_EMBED, 
                        N_AI_EMBED, 
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL, 
                        NOT_ROW_DIMENSION) 
{
}

ObExprAIEmbed::~ObExprAIEmbed() 
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const 
{
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    types_stack[CONTENT_IDX].set_calc_type(ObVarcharType);
    types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (param_num == 3) {
      if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (arg_model_id->is_null() || arg_content->is_null()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or content is null");
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObString model_id = arg_model_id->get_string();
    ObString content = arg_content->get_string();
    if (model_id.empty() || content.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      res.set_null();
    }
    int64_t dim = 0;
    ObJsonInt *dim_json = nullptr;
    ObJsonObject *config = nullptr;
    if (OB_FAIL(ret)) {
    } else if (OB_NOT_NULL(arg_dim)) {
      dim = arg_dim->get_int();
      if (dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
        res.set_null();
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, dim, dim_json))) {
      } else if (OB_FAIL(config->add("dimensions", dim_json))) {
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                   model_id, temp_allocator, resolved_endpoint))) {
    } else {
      ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
      ObString result;
      if (OB_FAIL(model.call_dense_embedding(content, config, result))) {
      } else if (OB_FAIL(validate_dense_embedding(result, dim))) {
      } else if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, res, result))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                           const ObRawExpr &raw_expr,
                           ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   ObObj const_data;
  //   bool got_data = false;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //     LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //   } else {
  //     rt_expr.extra_info_ = info;
  //   }
  // }

  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
