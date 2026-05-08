"""
data.pipeline.splits — pluggable split strategies.

Each module exports:
    SPLIT_NAME : str
    apply_split(df, boundaries=None, **kwargs) -> tuple[pd.DataFrame, dict]

The returned `boundaries` dict must be accepted verbatim by a second call to
apply_split on the disability DataFrame so both datasets receive identical
fold assignments.
"""
