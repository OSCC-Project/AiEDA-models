# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: remote_env.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10remote_env.proto\x12\x13\x65valuate_dreamplace\"\x18\n\x06\x41\x63tion\x12\x0e\n\x06\x61\x63tion\x18\x01 \x01(\x05\"\xe5\x01\n\x03Ret\x12\x1a\n\x12macro_idx_to_place\x18\x01 \x01(\x0c\x12\x0e\n\x06np_obs\x18\x02 \x01(\x0c\x12\x14\n\x0csparse_adj_i\x18\x03 \x01(\x0c\x12\x14\n\x0csparse_adj_j\x18\x04 \x01(\x0c\x12\x19\n\x11sparse_adj_weight\x18\x05 \x01(\x0c\x12\x13\n\x0b\x61\x63tion_mask\x18\x06 \x01(\x0c\x12\x13\n\x06reward\x18\x07 \x01(\x0cH\x00\x88\x01\x01\x12\x11\n\x04\x64one\x18\x08 \x01(\x0cH\x01\x88\x01\x01\x12\x11\n\x04info\x18\t \x01(\tH\x02\x88\x01\x01\x42\t\n\x07_rewardB\x07\n\x05_doneB\x07\n\x05_info\"4\n\x08Response\x12\x0c\n\x04\x66lag\x18\x01 \x01(\x08\x12\x11\n\x04info\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\x07\n\x05_info\"\x15\n\x06Number\x12\x0b\n\x03num\x18\x01 \x01(\x05\x32\x80\x04\n\tRemoteEnv\x12@\n\x05reset\x12\x1b.evaluate_dreamplace.Action\x1a\x18.evaluate_dreamplace.Ret\"\x00\x12?\n\x04step\x12\x1b.evaluate_dreamplace.Action\x1a\x18.evaluate_dreamplace.Ret\"\x00\x12L\n\x0e\x65pisode_length\x12\x1b.evaluate_dreamplace.Number\x1a\x1b.evaluate_dreamplace.Number\"\x00\x12H\n\nmacro_nums\x12\x1b.evaluate_dreamplace.Number\x1a\x1b.evaluate_dreamplace.Number\"\x00\x12G\n\tnode_nums\x12\x1b.evaluate_dreamplace.Number\x1a\x1b.evaluate_dreamplace.Number\"\x00\x12G\n\tedge_nums\x12\x1b.evaluate_dreamplace.Number\x1a\x1b.evaluate_dreamplace.Number\"\x00\x12\x46\n\x0bset_log_dir\x12\x1b.evaluate_dreamplace.Number\x1a\x18.evaluate_dreamplace.Ret\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'remote_env_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _ACTION._serialized_start=41
  _ACTION._serialized_end=65
  _RET._serialized_start=68
  _RET._serialized_end=297
  _RESPONSE._serialized_start=299
  _RESPONSE._serialized_end=351
  _NUMBER._serialized_start=353
  _NUMBER._serialized_end=374
  _REMOTEENV._serialized_start=377
  _REMOTEENV._serialized_end=889
# @@protoc_insertion_point(module_scope)
