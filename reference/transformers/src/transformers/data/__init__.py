# Minimal subset of upstream transformers.data (v4.56.0).
#
# The vendored transformers checkout intentionally omits the datasets/,
# metrics/ and processors/ subpackages. Only data_collator is needed by
# transformers.trainer. Keep this file in sync with upstream if you ever
# expand the subset.

from .data_collator import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForMultipleChoice,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForSOP,
    DataCollatorForTokenClassification,
    DataCollatorForWholeWordMask,
    DataCollatorWithFlattening,
    DataCollatorWithPadding,
    DefaultDataCollator,
    default_data_collator,
)

__all__ = [
    "DataCollator",
    "DataCollatorForLanguageModeling",
    "DataCollatorForMultipleChoice",
    "DataCollatorForPermutationLanguageModeling",
    "DataCollatorForSeq2Seq",
    "DataCollatorForSOP",
    "DataCollatorForTokenClassification",
    "DataCollatorForWholeWordMask",
    "DataCollatorWithFlattening",
    "DataCollatorWithPadding",
    "DefaultDataCollator",
    "default_data_collator",
]
