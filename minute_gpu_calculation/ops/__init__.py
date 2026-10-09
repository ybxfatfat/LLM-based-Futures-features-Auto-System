"""Physical GPU operator registry used by :mod:`batch_executor`.

Formula names are kept as strings in the DAG, so this module is the single
place that binds those public names to executable operator instances.
"""

from .ops_cs import CsRank, CsZscore
from .ops_math import (
    Abs,
    Add,
    DistortLog,
    DistortSqrt,
    Div,
    Equal,
    FillNa,
    IfElse,
    IndicatorMoreThanZero,
    IsNa,
    LessThan,
    Log,
    Minus,
    MoreThan,
    Prod,
    Sign,
    Sqrt,
)
from .ops_ts import (
    EWMA,
    TsArgMax,
    TsArgMin,
    TsCorr,
    TsCov,
    TsCv,
    TsDecay,
    TsDelay,
    TsDelta,
    TsEntropy,
    TsFacNeutralCoef,
    TsFacNeutralRes,
    TsFirst,
    TsForwardReturn,
    TsFutureDelay,
    TsIr,
    TsKurtosis,
    TsLast,
    TsMad,
    TsMax,
    TsMaxDiff,
    TsMean,
    TsMedian,
    TsMin,
    TsMinDiff,
    TsPct,
    TsQuantile,
    TsRank,
    TsSkewness,
    TsStd,
    TsSum,
    TsZscore,
)


_minus = Minus()
_prod = Prod()
_div = Div()
_ts_zscore = TsZscore()

OP_REGISTRY = {
    # Element-wise mathematics and conditionals.
    "Abs": Abs(),
    "Log": Log(),
    "Sqrt": Sqrt(),
    "Sign": Sign(),
    "IndicatorMoreThanZero": IndicatorMoreThanZero(),
    "DistortLog": DistortLog(),
    "DistortSqrt": DistortSqrt(),
    "Add": Add(),
    "Minus": _minus,
    "Sub": _minus,
    "Prod": _prod,
    "Mul": _prod,
    "Div": _div,
    "SafeDiv": _div,
    "MoreThan": MoreThan(),
    "LessThan": LessThan(),
    "Equal": Equal(),
    "IfElse": IfElse(),
    "FillNa": FillNa(),
    "IsNa": IsNa(),
    # Time-series operators.
    "TsDelay": TsDelay(),
    "TsDelta": TsDelta(),
    "TsPct": TsPct(),
    "TsFirst": TsFirst(),
    "TsLast": TsLast(),
    "TsSum": TsSum(),
    "TsMean": TsMean(),
    "TsStd": TsStd(),
    "TsZscore": _ts_zscore,
    "TsZScore": _ts_zscore,
    "TsMax": TsMax(),
    "TsMin": TsMin(),
    "TsMaxDiff": TsMaxDiff(),
    "TsMinDiff": TsMinDiff(),
    "TsArgMax": TsArgMax(),
    "TsArgMin": TsArgMin(),
    "TsMedian": TsMedian(),
    "TsQuantile": TsQuantile(),
    "TsMad": TsMad(),
    "TsRank": TsRank(),
    "TsSkewness": TsSkewness(),
    "TsKurtosis": TsKurtosis(),
    "TsEntropy": TsEntropy(),
    "TsCv": TsCv(),
    "TsIr": TsIr(),
    "EWMA": EWMA(),
    "TsDecay": TsDecay(),
    "TsCov": TsCov(),
    "TsCorr": TsCorr(),
    "TsFacNeutralCoef": TsFacNeutralCoef(),
    "TsFacNeutralRes": TsFacNeutralRes(),
    "TsFutureDelay": TsFutureDelay(),
    "TsForwardReturn": TsForwardReturn(),
    # Cross-sectional operators.
    "CsZscore": CsZscore(),
    "CsRank": CsRank(),
}

__all__ = ["OP_REGISTRY"]
