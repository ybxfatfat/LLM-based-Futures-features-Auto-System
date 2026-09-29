#pragma once
#include <vector>
#include <deque>
#include <cmath>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <set>
#include <iostream>

// ======================================================================
// 基础节点抽象 & RingBuffer
// ======================================================================
class OpNode
{
public:
    float val_ = NAN;
    std::vector<std::shared_ptr<OpNode>> inputs_;

    OpNode(std::vector<std::shared_ptr<OpNode>> inputs = {}) : inputs_(inputs) {}
    virtual ~OpNode() = default;

    virtual void update() = 0;
    virtual void clear() { val_ = NAN; }
    inline float get_value() const { return val_; }
};

class ConstNode : public OpNode
{
public:
    ConstNode(float v) { val_ = v; }
    void update() override {}
};

class FeatureNode : public OpNode
{
public:
    void inject(float v) { val_ = v; }
    void update() override {}
};

template <typename T>
class RingBuffer
{
public:
    std::vector<T> buffer;
    int head = 0, count = 0, window = 0;

    RingBuffer(int w) : window(w)
    {
        if (w > 0)
            buffer.resize(w);
    }

    T push(T val)
    {
        if (window == 0)
        {
            buffer.push_back(val);
            count++;
            return NAN;
        }
        else
        {
            T old_val = (count >= window) ? buffer[head] : NAN;
            if (count < window)
                count++;
            buffer[head] = val;
            head = (head + 1) % window;
            return old_val;
        }
    }

    void clear()
    {
        if (window == 0)
            buffer.clear();
        head = 0;
        count = 0;
    }

    std::vector<T> get_flat_data() const
    {
        if (window == 0)
            return buffer;
        std::vector<T> res;
        res.reserve(count);
        for (int i = 0; i < count; ++i)
        {
            res.push_back(buffer[(head - count + i + window) % window]);
        }
        return res;
    }
};

// ======================================================================
// 1. 无状态数学算子 (Stateless Ops) - 1:1 映射你提供的所有算子
// ======================================================================
class OpAbs : public OpNode
{
public:
    OpAbs(std::shared_ptr<OpNode> x) : OpNode({x}) {}
    void update() override { val_ = std::abs(inputs_[0]->get_value()); }
};

class OpLog : public OpNode
{
public:
    OpLog(std::shared_ptr<OpNode> x) : OpNode({x}) {}
    void update() override { val_ = std::log(std::fmax(inputs_[0]->get_value(), 1e-8f)); }
};

class OpSqrt : public OpNode
{
public:
    OpSqrt(std::shared_ptr<OpNode> x) : OpNode({x}) {}
    void update() override { val_ = std::sqrt(std::fmax(inputs_[0]->get_value(), 0.0f)); }
};

class OpSign : public OpNode
{
public:
    OpSign(std::shared_ptr<OpNode> x) : OpNode({x}) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        if (std::isnan(x))
            val_ = NAN;
        else
            val_ = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
};

class OpIndicatorMoreThanZero : public OpNode
{
public:
    OpIndicatorMoreThanZero(std::shared_ptr<OpNode> x) : OpNode({x}) {}
    void update() override { val_ = (inputs_[0]->get_value() > 0.0f) ? 1.0f : 0.0f; }
};

class OpDistortLog : public OpNode
{
public:
    OpDistortLog(std::shared_ptr<OpNode> x) : OpNode({x}) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        float sign = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        val_ = sign * std::log1p(std::abs(x));
    }
};

class OpDistortSqrt : public OpNode
{
public:
    OpDistortSqrt(std::shared_ptr<OpNode> x) : OpNode({x}) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        float sign = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        val_ = sign * std::sqrt(std::abs(x));
    }
};

// 二元对齐算子
class OpAdd : public OpNode
{
public:
    OpAdd(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override { val_ = inputs_[0]->get_value() + inputs_[1]->get_value(); }
};

class OpSub : public OpNode
{
public:
    OpSub(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override { val_ = inputs_[0]->get_value() - inputs_[1]->get_value(); }
};

class OpMul : public OpNode
{
public:
    OpMul(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override { val_ = inputs_[0]->get_value() * inputs_[1]->get_value(); }
};

class OpDiv : public OpNode
{
public:
    OpDiv(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override
    {
        float x = inputs_[0]->get_value(), y = inputs_[1]->get_value();
        val_ = (y != 0.0f) ? (x / y) : 0.0f;
    }
};

class OpMoreThan : public OpNode
{
public:
    OpMoreThan(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override { val_ = (inputs_[0]->get_value() > inputs_[1]->get_value()) ? 1.0f : 0.0f; }
};

class OpLessThan : public OpNode
{
public:
    OpLessThan(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override { val_ = (inputs_[0]->get_value() < inputs_[1]->get_value()) ? 1.0f : 0.0f; }
};

class OpEqual : public OpNode
{
public:
    OpEqual(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override
    {
        float x = inputs_[0]->get_value(), y = inputs_[1]->get_value();
        val_ = (std::abs(x - y) <= 1e-5f) ? 1.0f : 0.0f;
    }
};

// ======================================================================
// 扩展比较算子: >= (GE), <= (LE), != (NEQ)
// ======================================================================
class OpGreaterEqual : public OpNode
{
public:
    OpGreaterEqual(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override { val_ = (inputs_[0]->get_value() >= inputs_[1]->get_value()) ? 1.0f : 0.0f; }
};

class OpLessEqual : public OpNode
{
public:
    OpLessEqual(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override { val_ = (inputs_[0]->get_value() <= inputs_[1]->get_value()) ? 1.0f : 0.0f; }
};

class OpNotEqual : public OpNode
{
public:
    OpNotEqual(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override
    {
        float x = inputs_[0]->get_value(), y = inputs_[1]->get_value();
        val_ = (std::abs(x - y) > 1e-5f) ? 1.0f : 0.0f;
    }
};

// ======================================================================
// 逻辑与 / 逻辑或 算子 (支持按位运算符号 & 和 | )
// ======================================================================
class OpAnd : public OpNode
{
public:
    OpAnd(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override
    {
        // 只要两个都大于 0 就认为是 True
        val_ = (inputs_[0]->get_value() > 0.0f && inputs_[1]->get_value() > 0.0f) ? 1.0f : 0.0f;
    }
};

class OpOr : public OpNode
{
public:
    OpOr(std::shared_ptr<OpNode> a, std::shared_ptr<OpNode> b) : OpNode({a, b}) {}
    void update() override
    {
        val_ = (inputs_[0]->get_value() > 0.0f || inputs_[1]->get_value() > 0.0f) ? 1.0f : 0.0f;
    }
};

class OpIfElse : public OpNode
{
public:
    OpIfElse(std::shared_ptr<OpNode> c, std::shared_ptr<OpNode> t, std::shared_ptr<OpNode> f) : OpNode({c, t, f}) {}
    void update() override { val_ = (inputs_[0]->get_value() > 0.0f) ? inputs_[1]->get_value() : inputs_[2]->get_value(); }
};

class OpFillNa : public OpNode
{
public:
    OpFillNa(std::shared_ptr<OpNode> x, std::shared_ptr<OpNode> fill) : OpNode({x, fill}) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        val_ = std::isnan(x) ? inputs_[1]->get_value() : x;
    }
};

class OpIsNa : public OpNode
{
public:
    OpIsNa(std::shared_ptr<OpNode> x) : OpNode({x}) {}
    void update() override { val_ = std::isnan(inputs_[0]->get_value()) ? 1.0f : 0.0f; }
};

// ======================================================================
// 2. 基础位移算子 (Shifting & Pct)
// ======================================================================
class OpTsDelay : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;

public:
    OpTsDelay(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);
        val_ = (rb_.count >= window_) ? old : NAN;
    }
    void clear() override
    {
        rb_.clear();
        val_ = NAN;
    }
};

class OpTsDelta : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;

public:
    OpTsDelta(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);
        if (rb_.count >= window_ && !std::isnan(x) && !std::isnan(old))
            val_ = x - old;
        else
            val_ = NAN;
    }
    void clear() override
    {
        rb_.clear();
        val_ = NAN;
    }
};

class OpTsPct : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;

public:
    OpTsPct(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);
        if (rb_.count >= window_ && x > 0 && old > 0)
            val_ = std::log(x / old);
        else
            val_ = NAN;
    }
    void clear() override
    {
        rb_.clear();
        val_ = NAN;
    }
};

class OpTsFirst : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;

public:
    OpTsFirst(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override
    {
        rb_.push(inputs_[0]->get_value());
        if (rb_.count >= min_periods_)
        {
            // 返回最老的一个元素。如果是 bucket 模式(w=0)，返回第0个。
            val_ = (window_ == 0) ? rb_.buffer[0] : rb_.buffer[rb_.head];
        }
        else
        {
            val_ = NAN;
        }
    }
    void clear() override
    {
        rb_.clear();
        val_ = NAN;
    }
};

class OpTsLast : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;

public:
    OpTsLast(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        rb_.push(x);
        val_ = (rb_.count >= min_periods_) ? x : NAN;
    }
    void clear() override
    {
        rb_.clear();
        val_ = NAN;
    }
};

// ======================================================================
// 3. 线性统计算子 (O(1) 极速在线估计，严防精度丢失)
// ======================================================================
class TsMoments : public OpNode
{
protected:
    int window_, min_periods_;
    RingBuffer<float> rb_;
    double sum1_ = 0, sum2_ = 0, sum3_ = 0, sum4_ = 0;
    int valid_cnt_ = 0;

public:
    TsMoments(std::shared_ptr<OpNode> x, int w, int min_p)
        : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}

    // 只提供 set_input 支持空构造后接入连线
    TsMoments() : window_(0), min_periods_(0), rb_(0) {}
    void set_input(std::shared_ptr<OpNode> in) { inputs_.push_back(in); }

    void update() override
    {
        if (inputs_.empty())
            return;
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);

        if (!std::isnan(x))
        {
            double d = x, d2 = d * d;
            sum1_ += d;
            sum2_ += d2;
            sum3_ += d2 * d;
            sum4_ += d2 * d2;
            valid_cnt_++;
        }
        if (!std::isnan(old))
        {
            double d = old, d2 = d * d;
            sum1_ -= d;
            sum2_ -= d2;
            sum3_ -= d2 * d;
            sum4_ -= d2 * d2;
            valid_cnt_--;
        }
        compute_target();
    }
    virtual void compute_target() = 0;

    void clear() override
    {
        rb_.clear();
        sum1_ = 0;
        sum2_ = 0;
        sum3_ = 0;
        sum4_ = 0;
        valid_cnt_ = 0;
        val_ = NAN;
    }
};

class OpTsSum : public TsMoments
{
public:
    using TsMoments::TsMoments;
    void compute_target() override { val_ = (valid_cnt_ >= min_periods_) ? (float)sum1_ : NAN; }
};

class OpTsMean : public TsMoments
{
public:
    using TsMoments::TsMoments;
    void compute_target() override { val_ = (valid_cnt_ >= min_periods_) ? (float)(sum1_ / valid_cnt_) : NAN; }
};

class OpTsStd : public TsMoments
{
public:
    using TsMoments::TsMoments;
    void compute_target() override
    {
        if (valid_cnt_ >= min_periods_ && valid_cnt_ > 1)
        {
            double mean = sum1_ / valid_cnt_;
            double var = (sum2_ - valid_cnt_ * mean * mean) / valid_cnt_;
            val_ = (var > 0) ? (float)std::sqrt(var) : 0.0f;
        }
        else
            val_ = NAN;
    }
};

class OpTsZscore : public TsMoments
{
public:
    using TsMoments::TsMoments;
    void compute_target() override
    {
        if (valid_cnt_ >= min_periods_ && valid_cnt_ > 1)
        {
            double mean = sum1_ / valid_cnt_;
            double var = (sum2_ - valid_cnt_ * mean * mean) / valid_cnt_;
            double std_val = (var > 0) ? std::sqrt(var) : 0.0;
            if (std_val > 0)
                val_ = (float)((inputs_[0]->get_value() - mean) / std_val);
            else
                val_ = 1.0f;
        }
        else
            val_ = NAN;
    }
};

class OpTsSkewness : public TsMoments
{
public:
    OpTsSkewness(std::shared_ptr<OpNode> x, int w, int min_p) : TsMoments(x, w, min_p) {}
    void compute_target() override
    {
        if (valid_cnt_ >= std::max(min_periods_, 3))
        {
            double m = sum1_ / valid_cnt_;
            double m2 = (sum2_ - 2 * m * sum1_ + valid_cnt_ * m * m) / valid_cnt_;
            double m3 = (sum3_ - 3 * m * sum2_ + 3 * m * m * sum1_ - valid_cnt_ * m * m * m) / valid_cnt_;
            double skew = m3 / (std::pow(m2, 1.5) + 1e-12);
            double adj = std::sqrt(valid_cnt_ * (valid_cnt_ - 1)) / (valid_cnt_ - 2);
            val_ = (float)(skew * adj);
        }
        else
            val_ = NAN;
    }
};

class OpTsKurtosis : public TsMoments
{
public:
    OpTsKurtosis(std::shared_ptr<OpNode> x, int w, int min_p) : TsMoments(x, w, min_p) {}
    void compute_target() override
    {
        if (valid_cnt_ >= std::max(min_periods_, 4))
        {
            double m = sum1_ / valid_cnt_;
            double m2 = (sum2_ - 2 * m * sum1_ + valid_cnt_ * m * m) / valid_cnt_;
            double m4 = (sum4_ - 4 * m * sum3_ + 6 * m * m * sum2_ - 4 * m * m * m * sum1_ + valid_cnt_ * m * m * m * m) / valid_cnt_;
            double kurt = m4 / (m2 * m2 + 1e-12);
            double adj = (double)(valid_cnt_ - 1) / ((valid_cnt_ - 2) * (valid_cnt_ - 3) + 1e-12);
            val_ = (float)(adj * ((valid_cnt_ + 1) * kurt - 3 * (valid_cnt_ - 1)));
        }
        else
            val_ = NAN;
    }
};

class OpTsCv : public TsMoments
{ // Std / Mean
public:
    using TsMoments::TsMoments;
    void compute_target() override
    {
        if (valid_cnt_ >= min_periods_ && valid_cnt_ > 1)
        {
            double mean = sum1_ / valid_cnt_;
            double var = (sum2_ - valid_cnt_ * mean * mean) / valid_cnt_;
            double std_val = (var > 0) ? std::sqrt(var) : 0.0;
            val_ = (mean != 0) ? (float)(std_val / mean) : 0.0f;
        }
        else
            val_ = NAN;
    }
};

class OpTsIr : public TsMoments
{ // Mean / Std
public:
    using TsMoments::TsMoments;
    void compute_target() override
    {
        if (valid_cnt_ >= min_periods_ && valid_cnt_ > 1)
        {
            double mean = sum1_ / valid_cnt_;
            double var = (sum2_ - valid_cnt_ * mean * mean) / valid_cnt_;
            double std_val = (var > 0) ? std::sqrt(var) : 0.0;
            val_ = (std_val != 0) ? (float)(mean / std_val) : 0.0f;
        }
        else
            val_ = NAN;
    }
};

// ======================================================================
// 4. 衰减类算子 (Decay & EWMA)
// ======================================================================
class OpEWMA : public OpNode
{
private:
    float alpha_;
    bool initialized_ = false;

public:
    OpEWMA(std::shared_ptr<OpNode> x, float halflife) : OpNode({x})
    {
        alpha_ = 1.0f - std::exp(-std::log(2.0f) / halflife);
    }
    void update() override
    {
        float x = inputs_[0]->get_value();
        if (std::isnan(x))
        {
            val_ = NAN;
        }
        else
        {
            if (!initialized_)
            {
                val_ = x;
                initialized_ = true;
            }
            else
            {
                val_ = alpha_ * x + (1.0f - alpha_) * val_;
            }
        }
    }
    void clear() override
    {
        initialized_ = false;
        val_ = NAN;
    }
};

class OpTsDecay : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;
    double num_ = 0.0, den_ = 0.0;
    int valid_cnt_ = 0;

public:
    // O(1) 滑动加权衰减计算: sum(x_i * i) / sum(i)
    OpTsDecay(std::shared_ptr<OpNode> x, int w, int min_p)
        : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}

    void update() override
    {
        if (window_ == 0)
        { // W=0 的 Bucket 模式，只能使用 O(N) Lazy eval 计算 decay
            rb_.push(inputs_[0]->get_value());
            auto data = rb_.get_flat_data();
            double n = 0, d = 0;
            int cnt = 0;
            for (size_t i = 0; i < data.size(); ++i)
            {
                if (!std::isnan(data[i]))
                {
                    n += data[i] * (i + 1);
                    d += (i + 1);
                    cnt++;
                }
            }
            val_ = (cnt >= min_periods_ && d > 0) ? (float)(n / d) : NAN;
            return;
        }

        // W>0 的 O(1) 滑动模式
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);

        // 减去所有窗口内的累加基础值 (因为所有旧元素的权重都相对前移了1，权重降低)
        double current_sum = 0.0;
        for (int i = 0; i < window_; i++)
        {
            float v = rb_.buffer[i];
            if (!std::isnan(v))
                current_sum += v;
        }

        if (!std::isnan(old))
        {
            num_ -= (old * 1.0); // 最老的元素权重为1被丢弃
            den_ -= 1.0;
            valid_cnt_--;
        }

        // 新加入的元素权重最高 (为 window_)
        if (!std::isnan(x))
        {
            num_ = num_ - current_sum + x + (x * window_);
            den_ = den_ - valid_cnt_ + window_; // 权重分母逻辑同理
            valid_cnt_++;
        }
        else
        {
            num_ = num_ - current_sum;
            den_ = den_ - valid_cnt_;
        }

        val_ = (valid_cnt_ >= min_periods_ && den_ > 0) ? (float)(num_ / den_) : NAN;
    }

    void clear() override
    {
        rb_.clear();
        num_ = 0;
        den_ = 0;
        valid_cnt_ = 0;
        val_ = NAN;
    }
};

// ======================================================================
// 5. 单调队列算子 (Max, Min, ArgMax, ArgMin)
// ======================================================================
class OpTsMax : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;
    std::deque<std::pair<int, float>> mq_;
    int tick_ = 0, valid_cnt_ = 0;

public:
    OpTsMax(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);

        if (!std::isnan(x))
            valid_cnt_++;
        if (!std::isnan(old))
            valid_cnt_--;

        if (window_ > 0 && !mq_.empty() && mq_.front().first <= tick_ - window_)
            mq_.pop_front();
        if (!std::isnan(x))
        {
            while (!mq_.empty() && mq_.back().second <= x)
                mq_.pop_back();
            mq_.push_back({tick_, x});
        }
        tick_++;
        val_ = (valid_cnt_ >= min_periods_ && !mq_.empty()) ? mq_.front().second : NAN;
    }
    void clear() override
    {
        rb_.clear();
        mq_.clear();
        tick_ = 0;
        valid_cnt_ = 0;
        val_ = NAN;
    }
};

class OpTsMin : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;
    std::deque<std::pair<int, float>> mq_;
    int tick_ = 0, valid_cnt_ = 0;

public:
    OpTsMin(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);
        if (!std::isnan(x))
            valid_cnt_++;
        if (!std::isnan(old))
            valid_cnt_--;
        if (window_ > 0 && !mq_.empty() && mq_.front().first <= tick_ - window_)
            mq_.pop_front();
        if (!std::isnan(x))
        {
            while (!mq_.empty() && mq_.back().second >= x)
                mq_.pop_back();
            mq_.push_back({tick_, x});
        }
        tick_++;
        val_ = (valid_cnt_ >= min_periods_ && !mq_.empty()) ? mq_.front().second : NAN;
    }
    void clear() override
    {
        rb_.clear();
        mq_.clear();
        tick_ = 0;
        valid_cnt_ = 0;
        val_ = NAN;
    }
};

// MaxDiff = curr_x - Max
class OpTsMaxDiff : public OpTsMax
{
public:
    using OpTsMax::OpTsMax;
    void update() override
    {
        OpTsMax::update();
        if (!std::isnan(val_))
            val_ = inputs_[0]->get_value() - val_;
    }
};

// MinDiff = curr_x - Min
class OpTsMinDiff : public OpTsMin
{
public:
    using OpTsMin::OpTsMin;
    void update() override
    {
        OpTsMin::update();
        if (!std::isnan(val_))
            val_ = inputs_[0]->get_value() - val_;
    }
};

// ======================================================================
// TsArgMax：使用单调递减队列 (Monotonic Decreasing Deque)
// 时间复杂度：Amortized O(1)
// 极值对齐：完美对齐 PyTorch argmax 的 "First occurrence on tie" 逻辑
// ======================================================================
class OpTsArgMax : public OpNode
{
private:
    int window_;
    int min_periods_;
    RingBuffer<float> rb_;

    // 双端队列存的是对：{绝对时间戳 tick, 真实的特征值}
    std::deque<std::pair<int, float>> mq_;
    int tick_ = 0;
    int valid_cnt_ = 0;

public:
    OpTsArgMax(std::shared_ptr<OpNode> x, int w, int min_p)
        : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}

    void update() override
    {
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);

        // 1. 维护有效计数器
        if (!std::isnan(x))
            valid_cnt_++;
        if (!std::isnan(old))
            valid_cnt_--;

        // 2. 淘汰过期数据 (队头的时间戳 <= 当前时间 - 窗口大小)
        if (window_ > 0 && !mq_.empty() && mq_.front().first <= tick_ - window_)
        {
            mq_.pop_front();
        }

        // 3. 维护单调递减队列
        if (!std::isnan(x))
        {
            // 【极其关键的细节】：这里必须用严格小于 (<)
            // 如果队尾元素等于当前 x，我们*不*把它弹出去！
            // 这样能保证队列头部始终是“最老”的最大值，完全对齐 PyTorch 的 argmax 默认行为！
            while (!mq_.empty() && mq_.back().second < x)
            {
                mq_.pop_back();
            }
            mq_.push_back({tick_, x});
        }

        // 4. 计算相对索引特征
        // 只有在固定窗口 (W > 1) 的模式下，算这个相对距离才有物理意义
        if (valid_cnt_ >= min_periods_ && !mq_.empty() && window_ > 1)
        {
            // 计算当前窗口内“最老的那个合法 tick”的绝对位置
            int oldest_valid_tick = tick_ - window_ + 1;

            // 当前最大值的索引相对位置 (0 表示在最左侧/最老， W-1 表示在最右侧/最新)
            float idx = (float)(mq_.front().first - oldest_valid_tick);

            // 输出格式：(W - 1 - idx) / (W - 1)
            // 如果最新值为最大值，idx = W-1，输出 0.0
            // 如果最老值为最大值，idx = 0，  输出 1.0
            val_ = (window_ - 1.0f - idx) / (window_ - 1.0f);
        }
        else
        {
            val_ = NAN;
        }

        tick_++;
    }

    void clear() override
    {
        rb_.clear();
        mq_.clear();
        tick_ = 0;
        valid_cnt_ = 0;
        val_ = NAN;
    }
};

// ======================================================================
// TsArgMin：使用单调递增队列 (Monotonic Increasing Deque)
// 时间复杂度：Amortized O(1)
// 极值对齐：完美对齐 PyTorch argmin 的 "First occurrence on tie" 逻辑
// ======================================================================
class OpTsArgMin : public OpNode
{
private:
    int window_;
    int min_periods_;
    RingBuffer<float> rb_;

    // 双端队列存的是对：{绝对时间戳 tick, 真实的特征值}
    std::deque<std::pair<int, float>> mq_;
    int tick_ = 0;
    int valid_cnt_ = 0;

public:
    OpTsArgMin(std::shared_ptr<OpNode> x, int w, int min_p)
        : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}

    void update() override
    {
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);

        // 1. 维护有效计数器
        if (!std::isnan(x))
            valid_cnt_++;
        if (!std::isnan(old))
            valid_cnt_--;

        // 2. 淘汰过期数据 (队头的时间戳 <= 当前时间 - 窗口大小)
        if (window_ > 0 && !mq_.empty() && mq_.front().first <= tick_ - window_)
        {
            mq_.pop_front();
        }

        // 3. 维护单调递增队列
        if (!std::isnan(x))
        {
            // 【极其关键的细节】：这里必须用严格大于 (>)
            // 遇到相等的值时，保留旧值，确保取得是“最老的最小值”
            while (!mq_.empty() && mq_.back().second > x)
            {
                mq_.pop_back();
            }
            mq_.push_back({tick_, x});
        }

        // 4. 计算相对索引特征
        if (valid_cnt_ >= min_periods_ && !mq_.empty() && window_ > 1)
        {
            int oldest_valid_tick = tick_ - window_ + 1;

            float idx = (float)(mq_.front().first - oldest_valid_tick);

            // 输出格式：(W - 1 - idx) / (W - 1)
            val_ = (window_ - 1.0f - idx) / (window_ - 1.0f);
        }
        else
        {
            val_ = NAN;
        }

        tick_++;
    }

    void clear() override
    {
        rb_.clear();
        mq_.clear();
        tick_ = 0;
        valid_cnt_ = 0;
        val_ = NAN;
    }
};
// ======================================================================
// 6. O(log W) 极速二叉树算子：分位数与中位数 (双 Multiset 架构)
// 【核心解答】：这是对于 Window > 0 最完美的 O(log W) 流式解法！
// ======================================================================
class OpTsQuantile : public OpNode
{
private:
    int window_, min_periods_;
    float q_;
    RingBuffer<float> rb_;

    // 两个 multiset 平衡树
    std::multiset<float> left_, right_;

    void balance()
    {
        if (left_.empty() && right_.empty())
            return;
        int total = left_.size() + right_.size();
        int target_left = std::max(1, (int)std::round((total - 1) * q_)) + 1; // 修正左侧大小

        while (left_.size() < (size_t)target_left && !right_.empty())
        {
            left_.insert(*right_.begin());
            right_.erase(right_.begin());
        }
        while (left_.size() > (size_t)target_left)
        {
            auto it = left_.end();
            --it;
            right_.insert(*it);
            left_.erase(it);
        }
    }

public:
    OpTsQuantile(std::shared_ptr<OpNode> x, int w, float q, int min_p)
        : OpNode({x}), window_(w), min_periods_(min_p), q_(q), rb_(w) {}

    void update() override
    {
        float x = inputs_[0]->get_value();
        float old = rb_.push(x);

        if (window_ == 0)
        {
            // 切片桶模式：O(1) 直接存数组，因为 get_value 只有在分钟边界才调 1 次
            return;
        }

        // 滑动窗口模式：O(log W) 的红黑树增删
        if (!std::isnan(old))
        {
            auto it = left_.find(old);
            if (it != left_.end())
                left_.erase(it);
            else
                right_.erase(right_.find(old));
        }
        if (!std::isnan(x))
        {
            if (left_.empty() || x <= *left_.rbegin())
                left_.insert(x);
            else
                right_.insert(x);
        }
        balance();

        int cnt = left_.size() + right_.size();
        if (cnt >= min_periods_)
        {
            auto it = left_.end();
            --it;
            val_ = *it;
        }
        else
        {
            val_ = NAN;
        }
    }

    float get_value() const
    {
        if (window_ > 0)
            return val_;
        // 桶聚合时，O(N) 极速计算 (符合你期望的 nth_element 最佳形态)
        std::vector<float> data = rb_.get_flat_data();
        std::vector<float> v;
        for (float f : data)
            if (!std::isnan(f))
                v.push_back(f);
        if (v.size() < (size_t)min_periods_)
            return NAN;
        int t = std::round((v.size() - 1) * q_);
        std::nth_element(v.begin(), v.begin() + t, v.end());
        return v[t];
    }

    void clear() override
    {
        rb_.clear();
        left_.clear();
        right_.clear();
        val_ = NAN;
    }
};

class OpTsMedian : public OpTsQuantile
{
public:
    OpTsMedian(std::shared_ptr<OpNode> x, int w, int min_p) : OpTsQuantile(x, w, 0.5f, min_p) {}
};

// ======================================================================
// 7. Lazy Eval 算子 (适合 Rank, Mad, Entropy 等需要通盘扫描的操作)
// ======================================================================
class OpTsRank : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;

public:
    OpTsRank(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override
    {
        float x = inputs_[0]->get_value();
        rb_.push(x);
        // O(N) 计算: 虽然每次跑是 O(N)，但通常 W 非常小。若 W>0 且要极端优化，可上 Fenwick 树 (树状数组)
        if (window_ > 0)
            val_ = this->get_value();
    }
    float get_value() const
    {
        std::vector<float> data = rb_.get_flat_data();
        int valid = 0, greater = 0;
        float curr = data.empty() ? NAN : data.back();
        if (std::isnan(curr))
            return NAN;
        for (float v : data)
        {
            if (!std::isnan(v))
            {
                valid++;
                if (curr > v)
                    greater++;
            }
        }
        return (valid >= min_periods_) ? (float)greater / (valid - 1.0f + 1e-8f) : NAN;
    }
    void clear() override
    {
        rb_.clear();
        val_ = NAN;
    }
};

class OpTsMad : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;

public:
    OpTsMad(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override { rb_.push(inputs_[0]->get_value()); }
    float get_value() const
    {
        std::vector<float> v;
        for (float f : rb_.get_flat_data())
            if (!std::isnan(f))
                v.push_back(f);
        if (v.size() < (size_t)min_periods_)
            return NAN;
        std::vector<float> temp = v;
        int t = v.size() / 2;
        std::nth_element(temp.begin(), temp.begin() + t, temp.end());
        float med = temp[t];
        float mad_sum = 0;
        for (float f : v)
            mad_sum += std::abs(f - med);
        return mad_sum / v.size();
    }
    void clear() override
    {
        rb_.clear();
        val_ = NAN;
    }
};

class OpTsEntropy : public OpNode
{
private:
    int window_, min_periods_;
    RingBuffer<float> rb_;

public:
    OpTsEntropy(std::shared_ptr<OpNode> x, int w, int min_p) : OpNode({x}), window_(w), min_periods_(min_p), rb_(w) {}
    void update() override { rb_.push(inputs_[0]->get_value()); }
    float get_value() const
    {
        std::vector<float> v;
        double sum = 0;
        for (float f : rb_.get_flat_data())
            if (!std::isnan(f))
            {
                v.push_back(f);
                sum += f;
            }
        if (v.size() < (size_t)min_periods_ || sum <= 0)
            return NAN;
        double ent = 0;
        for (float f : v)
        {
            double p = f / sum;
            if (p > 0)
                ent -= p * std::log(p);
        }
        return (float)ent;
    }
    void clear() override
    {
        rb_.clear();
        val_ = NAN;
    }
};

// ======================================================================
// 8. 二元状态算子 (Binary Cov, Corr, Beta, Residual)
// ======================================================================
class TsBinaryMoments : public OpNode
{
protected:
    int window_, min_periods_;
    RingBuffer<float> rb_x_, rb_y_;
    double sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0;
    int valid_cnt_ = 0;

public:
    TsBinaryMoments(std::shared_ptr<OpNode> x, std::shared_ptr<OpNode> y, int w, int min_p)
        : OpNode({x, y}), window_(w), min_periods_(min_p), rb_x_(w), rb_y_(w) {}

    void update() override
    {
        float x = inputs_[0]->get_value();
        float y = inputs_[1]->get_value();

        float old_x = rb_x_.push(x);
        float old_y = rb_y_.push(y);

        bool curr_valid = !std::isnan(x) && !std::isnan(y);
        bool old_valid = !std::isnan(old_x) && !std::isnan(old_y);

        if (curr_valid)
        {
            double dx = x, dy = y;
            sum_x += dx;
            sum_y += dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
            sum_xy += dx * dy;
            valid_cnt_++;
        }
        if (old_valid)
        {
            double dx = old_x, dy = old_y;
            sum_x -= dx;
            sum_y -= dy;
            sum_x2 -= dx * dx;
            sum_y2 -= dy * dy;
            sum_xy -= dx * dy;
            valid_cnt_--;
        }
        compute_target(x, y);
    }

    virtual void compute_target(float curr_x, float curr_y) = 0;

    void clear() override
    {
        rb_x_.clear();
        rb_y_.clear();
        sum_x = 0;
        sum_y = 0;
        sum_x2 = 0;
        sum_y2 = 0;
        sum_xy = 0;
        valid_cnt_ = 0;
        val_ = NAN;
    }
};

class OpTsCov : public TsBinaryMoments
{
public:
    using TsBinaryMoments::TsBinaryMoments;
    void compute_target(float x, float y) override
    {
        if (valid_cnt_ >= min_periods_ && valid_cnt_ > 1)
        {
            val_ = (float)((sum_xy - (sum_x * sum_y) / valid_cnt_) / (valid_cnt_ - 1));
        }
        else
            val_ = NAN;
    }
};

class OpTsCorr : public TsBinaryMoments
{
public:
    using TsBinaryMoments::TsBinaryMoments;
    void compute_target(float x, float y) override
    {
        if (valid_cnt_ >= min_periods_ && valid_cnt_ > 1)
        {
            double cov = sum_xy - (sum_x * sum_y) / valid_cnt_;
            double vx = sum_x2 - (sum_x * sum_x) / valid_cnt_;
            double vy = sum_y2 - (sum_y * sum_y) / valid_cnt_;
            if (vx > 0 && vy > 0)
            {
                double corr = cov / std::sqrt(vx * vy);
                val_ = (float)std::max(-1.0, std::min(1.0, corr));
            }
            else
                val_ = NAN;
        }
        else
            val_ = NAN;
    }
};

class OpTsFacNeutralCoef : public TsBinaryMoments
{ // Beta
public:
    using TsBinaryMoments::TsBinaryMoments;
    void compute_target(float x, float y) override
    {
        if (valid_cnt_ >= min_periods_ && valid_cnt_ > 1)
        {
            double cov = (sum_xy - (sum_x * sum_y) / valid_cnt_) / valid_cnt_;
            double vx = (sum_x2 - (sum_x * sum_x) / valid_cnt_) / valid_cnt_;
            val_ = (vx > 0) ? (float)(cov / vx) : NAN;
        }
        else
            val_ = NAN;
    }
};

class OpTsFacNeutralRes : public TsBinaryMoments
{ // Residual: y - (beta * x + alpha)
public:
    using TsBinaryMoments::TsBinaryMoments;
    void compute_target(float curr_x, float curr_y) override
    {
        if (std::isnan(curr_x) || std::isnan(curr_y))
        {
            val_ = NAN;
            return; // 极度关键：当前没数据，残差必须为 NaN
        }
        if (valid_cnt_ >= min_periods_ && valid_cnt_ > 1)
        {
            double cov = (sum_xy - (sum_x * sum_y) / valid_cnt_) / valid_cnt_;
            double vx = (sum_x2 - (sum_x * sum_x) / valid_cnt_) / valid_cnt_;
            double beta = (vx > 0) ? (cov / vx) : 0.0;
            double mean_x = sum_x / valid_cnt_;
            double mean_y = sum_y / valid_cnt_;
            val_ = (float)((curr_y - mean_y) - beta * (curr_x - mean_x));
        }
        else
            val_ = NAN;
    }
};

// ======================================================================
// 终极游程包装器：使任何状态算子获得条件重置能力 (TsResetable)
// ======================================================================
class OpTsResetable : public OpNode
{
private:
    std::shared_ptr<OpNode> inner_op_;

public:
    // in1: 被包装的真实状态算子 (如 OpTsMax)
    // in2: 重置条件信号
    OpTsResetable(std::shared_ptr<OpNode> inner_op, std::shared_ptr<OpNode> reset_cond)
        : OpNode({inner_op, reset_cond}), inner_op_(inner_op) {}

    // Python 端初始化用
    OpTsResetable() : OpNode() {}

    // in1: 被包装的算子, in2: 重置条件
    void set_inputs(std::shared_ptr<OpNode> op, std::shared_ptr<OpNode> cond)
    {
        inputs_.resize(2);
        inputs_[0] = op;
        inputs_[1] = cond;
        inner_op_ = op;
    }

    void update() override
    {
        if (inputs_.size() < 2 || !inner_op_)
            return;

        float reset_cond = inputs_[1]->get_value();

        // 如果重置条件成立 (>0)，瞬间强行清空内部包裹的算子的状态！
        if (!std::isnan(reset_cond) && reset_cond > 0.0f)
        {
            inner_op_->clear();
        }

        // 不管清没清，都让内部算子按照正常逻辑往前推进一步
        // 注意：底层图引擎在遍历 topo_order 时已经调过 inner_op_->update() 了
        // 为了防止重复 update，这个包装器只负责清空和转交值，不再调 update。

        val_ = inner_op_->get_value();
    }

    void clear() override
    {
        if (inner_op_)
            inner_op_->clear();
        val_ = NAN;
    }
};