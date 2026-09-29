// tick_engine.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <map>
#include <string>
#include <vector>
#include <memory>

// 引入刚刚保存的算子库 (请确保文件名一致)
#include "tick_engine_ops.hpp"

namespace py = pybind11;

// ======================================================================
// 核心流式引擎：零拷贝驱动计算图 (支持多特征宽表输出)
// ======================================================================
py::tuple run_tick_engine(
    py::array_t<int64_t> timestamps_ms,
    std::map<std::string, std::shared_ptr<FeatureNode>> feature_nodes_map,
    std::map<std::string, py::array_t<float>> tick_data_arrays,
    std::vector<std::shared_ptr<OpNode>> topo_ordered_nodes,
    // [Refactor]: Map output nodes for multi-feature extraction
    std::map<std::string, std::shared_ptr<OpNode>> final_output_nodes,
    std::vector<std::shared_ptr<OpNode>> state_nodes)
{
      int64_t N = timestamps_ms.size();
      const int64_t *ts_ptr = timestamps_ms.data();

      // 提取所有输入的指针，实现绝对 O(1) 的零拷贝访问
      std::map<std::string, const float *> data_ptrs;
      for (auto const &kv : tick_data_arrays)
      {
            data_ptrs[kv.first] = kv.second.data();
      }

      // 初始化两个输出：时间戳和特征值字典
      std::vector<int64_t> out_timestamps;

      // [Memory]: Initialize contiguous memory vectors for output features
      std::map<std::string, std::vector<float>> out_features_map;
      for (auto const &kv : final_output_nodes)
      {
            out_features_map[kv.first] = std::vector<float>();
      }

      if (N == 0)
      {
            py::dict empty_dict;
            return py::make_tuple(py::cast(out_timestamps), empty_dict);
      }

      // 强制对齐到整分钟 (毫秒)
      int64_t current_min_bound = (ts_ptr[0] / 60000) * 60000 + 60000;

      // 提前把 string map 查找降维成连续的内存指针数组，O(1) 极速访问
      std::vector<std::pair<std::shared_ptr<FeatureNode>, const float*>> fast_inject_list;
      fast_inject_list.reserve(feature_nodes_map.size());
      for (auto const &kv : feature_nodes_map)
      {
            fast_inject_list.push_back({kv.second, data_ptrs[kv.first]});
      }

      for (int64_t i = 0; i < N; ++i)
      {
            int64_t ts = ts_ptr[i];

            // 1. 严格的串行：检测 -> 收割 -> 清空 -> 推进
            if (ts >= current_min_bound)
            {
                  out_timestamps.push_back(current_min_bound); // 记录收割时的时间戳

                  // [Execution]: Harvest feature values at minute boundaries
                  for (auto const &kv : final_output_nodes)
                  {
                        out_features_map[kv.first].push_back(kv.second->get_value());
                  }

                  // 清理状态算子，准备迎接新的一分钟
                  for (auto &node : state_nodes)
                        node->clear();

                  current_min_bound = (ts / 60000) * 60000 + 60000;
            }

            // 2. 极速将数据注入叶子节点
            // for (auto const &kv : feature_nodes_map)
            // {
            //       kv.second->inject(data_ptrs[kv.first][i]);
            // }
            for (auto const &item : fast_inject_list)
            {
                  item.first->inject(item.second[i]); 
                  // 🚀 快在这里：item.second 直接就是底层连续数组的指针，没有字典查找！
            }

            // 3. 按照 Python 给好的拓扑顺序，极速递推整个 DAG
            for (auto &node : topo_ordered_nodes)
            {
                  node->update();
            }
      }

      // 最后一分钟的数据收尾
      out_timestamps.push_back(current_min_bound);
      for (auto const &kv : final_output_nodes)
      {
            out_features_map[kv.first].push_back(kv.second->get_value());
      }

      // [PyBind11]: Convert C++ map to Python dict zero-copy return
      py::dict py_out_features;
      for (auto const &kv : out_features_map)
      {
            py_out_features[py::str(kv.first)] = py::cast(kv.second);
      }

      return py::make_tuple(py::cast(out_timestamps), py_out_features);
}

// ======================================================================
// PyBind11 模块导出与魔术方法重载 (完整全量算子版)
// ======================================================================
PYBIND11_MODULE(tick_engine, m)
{
      // ---------------------------------------------------------
      // 1. 基础基类与魔术方法
      // ---------------------------------------------------------
      py::class_<OpNode, std::shared_ptr<OpNode>>(m, "OpNode")
          // 原有的 OpNode 连线
          .def("set_input", [](std::shared_ptr<OpNode> self, std::shared_ptr<OpNode> in)
               {
            if (self->inputs_.empty()) self->inputs_.push_back(in); else self->inputs_[0] = in; })
          // [Operator Overload]: Support float scalar to ConstNode auto-casting
          .def("set_input", [](std::shared_ptr<OpNode> self, float val)
               {
            auto in = std::make_shared<ConstNode>(val);
            if (self->inputs_.empty()) self->inputs_.push_back(in); else self->inputs_[0] = in; })
          // 原有的双输入连线
          .def("set_inputs", [](std::shared_ptr<OpNode> self, std::shared_ptr<OpNode> in1, std::shared_ptr<OpNode> in2)
               {
            if (self->inputs_.size() < 2) self->inputs_.resize(2);
            self->inputs_[0] = in1; self->inputs_[1] = in2; })
          
          .def("set_inputs", [](std::shared_ptr<OpNode> self, std::shared_ptr<OpNode> in1, float val2)
               {
            if (self->inputs_.size() < 2) self->inputs_.resize(2);
            self->inputs_[0] = in1; self->inputs_[1] = std::make_shared<ConstNode>(val2); })
          .def("set_inputs", [](std::shared_ptr<OpNode> self, float val1, std::shared_ptr<OpNode> in2)
               {
            if (self->inputs_.size() < 2) self->inputs_.resize(2);
            self->inputs_[0] = std::make_shared<ConstNode>(val1); self->inputs_[1] = in2; })
          .def("get_inputs", [](std::shared_ptr<OpNode> n)
               { return n->inputs_; })

          // 数学重载
          .def("__add__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpAdd>(a, b); })
          .def("__add__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpAdd>(a, std::make_shared<ConstNode>(b)); })
          .def("__sub__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpSub>(a, b); })
          .def("__sub__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpSub>(a, std::make_shared<ConstNode>(b)); })
          .def("__truediv__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpDiv>(a, b); })
          .def("__truediv__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpDiv>(a, std::make_shared<ConstNode>(b)); })
          .def("__mul__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpMul>(a, b); })
          .def("__mul__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpMul>(a, std::make_shared<ConstNode>(b)); })

          // 逻辑重载
          .def("__gt__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpMoreThan>(a, b); })
          .def("__gt__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpMoreThan>(a, std::make_shared<ConstNode>(b)); })
          .def("__lt__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpLessThan>(a, b); })
          .def("__lt__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpLessThan>(a, std::make_shared<ConstNode>(b)); })
          .def("__eq__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpEqual>(a, b); })
          .def("__eq__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpEqual>(a, std::make_shared<ConstNode>(b)); })
          .def("__and__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpAnd>(a, b); })
          .def("__or__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpOr>(a, b); })
          .def("__ge__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpGreaterEqual>(a, b); })
          .def("__ge__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpGreaterEqual>(a, std::make_shared<ConstNode>(b)); })

          .def("__le__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpLessEqual>(a, b); })
          .def("__le__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpLessEqual>(a, std::make_shared<ConstNode>(b)); })

          .def("__ne__", [](std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpNotEqual>(a, b); })
          .def("__ne__", [](std::shared_ptr<OpNode> a, float b) -> std::shared_ptr<OpNode>
               { return std::make_shared<OpNotEqual>(a, std::make_shared<ConstNode>(b)); });

      py::class_<FeatureNode, OpNode, std::shared_ptr<FeatureNode>>(m, "FeatureNode")
          .def(py::init<>());
      py::class_<ConstNode, OpNode, std::shared_ptr<ConstNode>>(m, "ConstNode").def(py::init<float>());

      py::class_<OpTsResetable, OpNode, std::shared_ptr<OpTsResetable>>(m, "TsResetable")
          .def(py::init<>())
          .def("set_inputs", &OpTsResetable::set_inputs);

      // ---------------------------------------------------------
      // 2. 无状态一元算子 (直接传入 OpNode 使用)
      // ---------------------------------------------------------
      m.def("Abs", [](std::shared_ptr<OpNode> x) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpAbs>(x); });
      m.def("Log", [](std::shared_ptr<OpNode> x) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpLog>(x); });
      m.def("Sqrt", [](std::shared_ptr<OpNode> x) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpSqrt>(x); });
      m.def("Sign", [](std::shared_ptr<OpNode> x) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpSign>(x); });
      m.def("IndicatorMoreThanZero", [](std::shared_ptr<OpNode> x) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpIndicatorMoreThanZero>(x); });
      m.def("DistortLog", [](std::shared_ptr<OpNode> x) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpDistortLog>(x); });
      m.def("DistortSqrt", [](std::shared_ptr<OpNode> x) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpDistortSqrt>(x); });
      m.def("IsNa", [](std::shared_ptr<OpNode> x) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpIsNa>(x); });

      // 无状态二元算子
      m.def("FillNa", [](std::shared_ptr<OpNode> x, std::shared_ptr<OpNode> fill) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpFillNa>(x, fill); });

      m.def("FillNa", [](std::shared_ptr<OpNode> x, float fill) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpFillNa>(x, std::make_shared<ConstNode>(fill)); });

      // 三元分支
      m.def("IfElse", [](std::shared_ptr<OpNode> c, std::shared_ptr<OpNode> t, std::shared_ptr<OpNode> f) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpIfElse>(c, t, f); });
      m.def("IfElse", [](std::shared_ptr<OpNode> c, std::shared_ptr<OpNode> t, float f) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpIfElse>(c, t, std::make_shared<ConstNode>(f)); });
      m.def("IfElse", [](std::shared_ptr<OpNode> c, float t, float f) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpIfElse>(c, std::make_shared<ConstNode>(t), std::make_shared<ConstNode>(f)); });
      m.def("IfElse", [](std::shared_ptr<OpNode> cond, float t, std::shared_ptr<OpNode> f) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpIfElse>(cond, std::make_shared<ConstNode>(t), f); });

      // ---------------------------------------------------------
      // 3. 有状态时序算子 (工厂函数模式)
      // 核心思想：先用一个 dummy ConstNode 占位构造，随后在 Python 用 set_input 替换它！
      // ---------------------------------------------------------
      auto dummy = []()
      { return std::make_shared<ConstNode>(NAN); };

      // 基础位移
      m.def("TsDelay", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsDelay>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsDelta", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsDelta>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsPct", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsPct>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsFirst", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsFirst>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsLast", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsLast>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      // 高阶矩 (O(1) 极速)
      m.def("TsSum", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsSum>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsMean", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsMean>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsStd", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsStd>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsZscore", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsZscore>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsSkewness", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsSkewness>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsKurtosis", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsKurtosis>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsCv", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsCv>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsIr", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsIr>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      // 衰减类
      m.def("EWMA", [dummy](float halflife) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpEWMA>(dummy(), halflife); }, py::arg("halflife"));
      m.def("TsDecay", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsDecay>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      // 单调队列极值
      m.def("TsMax", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsMax>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsMin", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsMin>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsMaxDiff", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsMaxDiff>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsMinDiff", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsMinDiff>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsArgMax", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsArgMax>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsArgMin", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsArgMin>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      // 分位数与 Lazy 算子
      m.def("TsQuantile", [dummy](int w, float q, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsQuantile>(dummy(), w, q, min_p); }, py::arg("window") = 0, py::arg("q") = 0.5, py::arg("min_periods") = 0);
      m.def("TsMedian", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsMedian>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsRank", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsRank>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsMad", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsMad>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);
      m.def("TsEntropy", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsEntropy>(dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      // ---------------------------------------------------------
      // 4. 二元状态算子 (需要两个输入)
      // ---------------------------------------------------------
      m.def("TsCov", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsCov>(dummy(), dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      m.def("TsCorr", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsCorr>(dummy(), dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      m.def("TsFacNeutralCoef", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsFacNeutralCoef>(dummy(), dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      m.def("TsFacNeutralRes", [dummy](int w, int min_p) -> std::shared_ptr<OpNode>
            { return std::make_shared<OpTsFacNeutralRes>(dummy(), dummy(), w, min_p); }, py::arg("window") = 0, py::arg("min_periods") = 0);

      // 导出 NAN 常量与主函数
      m.attr("NAN") = NAN;
      m.def("run_tick_engine", &run_tick_engine);
}