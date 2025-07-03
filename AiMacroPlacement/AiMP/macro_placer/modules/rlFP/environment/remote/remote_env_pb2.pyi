from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Action(_message.Message):
    __slots__ = ["action"]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: int
    def __init__(self, action: _Optional[int] = ...) -> None: ...

class Number(_message.Message):
    __slots__ = ["num"]
    NUM_FIELD_NUMBER: _ClassVar[int]
    num: int
    def __init__(self, num: _Optional[int] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ["flag", "info"]
    FLAG_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    flag: bool
    info: str
    def __init__(self, flag: bool = ..., info: _Optional[str] = ...) -> None: ...

class Ret(_message.Message):
    __slots__ = ["action_mask", "done", "info", "macro_idx_to_place", "np_obs", "reward", "sparse_adj_i", "sparse_adj_j", "sparse_adj_weight"]
    ACTION_MASK_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    MACRO_IDX_TO_PLACE_FIELD_NUMBER: _ClassVar[int]
    NP_OBS_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    SPARSE_ADJ_I_FIELD_NUMBER: _ClassVar[int]
    SPARSE_ADJ_J_FIELD_NUMBER: _ClassVar[int]
    SPARSE_ADJ_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    action_mask: bytes
    done: bytes
    info: str
    macro_idx_to_place: bytes
    np_obs: bytes
    reward: bytes
    sparse_adj_i: bytes
    sparse_adj_j: bytes
    sparse_adj_weight: bytes
    def __init__(self, macro_idx_to_place: _Optional[bytes] = ..., np_obs: _Optional[bytes] = ..., sparse_adj_i: _Optional[bytes] = ..., sparse_adj_j: _Optional[bytes] = ..., sparse_adj_weight: _Optional[bytes] = ..., action_mask: _Optional[bytes] = ..., reward: _Optional[bytes] = ..., done: _Optional[bytes] = ..., info: _Optional[str] = ...) -> None: ...
