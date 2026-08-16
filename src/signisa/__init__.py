"""Signisa: known-target ASL sign verification.

SHARD_SCHEMA lives at package root so the torch-free tensor-build script and the
torch-using loader in signisa.data share one definition.
"""

SHARD_SCHEMA = 2  # 2 = ragged native-length sequences; 1 = fixed 160-frame stacks
