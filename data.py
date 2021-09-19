# Adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py

import logging
import os
import re
import warnings
from argparse import Namespace

import pandas as pd

from torch.utils.data.dataloader import DataLoader

from pytorch_lightning import LightningDataModule

from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    # DataCollatorForTokenClassification,
    DataCollatorForWholeWordMask,
    # TextDatasetForNextSentencePrediction,
)

from datasets import (
    load_dataset,
    Dataset,
    DatasetBuilder,
    Features,
    Value,
)

from utils import TextDatasetForNextSentencePrediction

# logger = logging.getLogger(__name__)
logger = logging.getLogger('trainer.data')

# NOTE: for local dataset testing
global hparams

hparams = Namespace(
    cache_dir='.cache',
    config_name='',
    data_dir='.',
    dataset_name="silicone",
    eval_batch_size=32,
    train_batch_size=32,
    fast_dev_run=True,
    model_name_or_path='bert-base-uncased',
    preprocessing_num_workers=4,
    num_workers=4,
    output_dir='output',
    subset_name='swda',
    tokenizer_name=None,
    line_by_line=True,
    pad_to_max_length=True,
    overwrite_cache=True,
    max_seq_length=512,
    mlm_probability=0.15,
)


class SWDA(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def setup(self, stage):
        dataset = load_dataset('silicone', 'swda', cache_dir=self.cache_dir, script_version="master")
        tokenizer =  BertTokenizerFast.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "Utterance"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=self.hparams.mlm_probability
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class GoEmotions(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def setup(self, stage):
        dataset = load_dataset('go_emotions', cache_dir=self.cache_dir, script_version="master")
        tokenizer =  BertTokenizerFast.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=self.hparams.mlm_probability
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class DealNoDeal(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def setup(self, stage):
        dataset = load_dataset('deal_or_no_dialog', 'dialogues', cache_dir=self.cache_dir, script_version="master")
        tokenizer =  BertTokenizerFast.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "dialogue"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=self.hparams.mlm_probability
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class AmazonReviews(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def setup(self, stage):
        dataset = load_dataset('amazon_reviews_multi', 'en', cache_dir=self.cache_dir, script_version="master")
        tokenizer =  BertTokenizerFast.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "review_body"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=self.hparams.mlm_probability
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle =  True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class AGNews(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def setup(self, stage):
        dataset = load_dataset('ag_news', cache_dir=self.cache_dir, script_version="master")

        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                'ag_news',
                split=f"train[:{20}%]",
                cache_dir=self.cache_dir,
            )

        tokenizer =  BertTokenizerFast.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=self.hparams.mlm_probability
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


# train on DNC debate monologue, validate on the DNC debate dialogue
class MonologueDialogue(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        dataset = load_dataset(
                    path='csv',
                    delimiter='\t',
                    data_files={'train': 'data/dnc_debate_monologues_for_MLM.tsv',
                                'validation': 'data/dnc_debate_dialogues_for_MLM.tsv',
                                }
                    )
        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "Text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.hparams.mlm_probability,
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


# train on DNC debate dialogue, validate on the DNC debate monologue
class DialogueMonologue(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        dataset = load_dataset(
                    path='csv',
                    delimiter='\t',
                    data_files={'train': 'data/dnc_debate_dialogues_for_MLM.tsv',
                                'validation': 'data/dnc_debate_monologues_for_MLM.tsv',
                                }
                    )
        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "Text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.hparams.mlm_probability,
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


# train on DNC debate monologue, validate on presidential dialogue; avoid for now
class DNCMonologuePresDialogue(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        dataset = load_dataset(
                    path='csv',
                    delimiter='\t',
                    data_files={'train': 'data/dnc_monologues_for_pres.tsv',
                                'validation': 'data/presidential_debate_dialogues.tsv',
                                }
                    )

        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "Text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.hparams.mlm_probability,
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


# train on DNC debate monologue, validate on presidential monologue; avoid for now
class DNCMonologuePresMonologue(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        dataset = load_dataset(
                    path='csv',
                    delimiter='\t',
                    data_files={'train': 'data/dnc_monologues_for_pres.tsv',
                                'validation': 'data/presidential_debate_monologues.tsv',
                                }
                    )

        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "Text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=True,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.hparams.mlm_probability,
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator


    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.eval_batch_size,
                          shuffle =  False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class DNCForNSP(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        padding = "max_length" if self.hparams.pad_to_max_length else False
        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        train = TextDatasetForNextSentencePrediction(
                    tokenizer,
                    file_path='data/dnc_debate_train_for_NSP.tsv',
                    block_size=self.hparams.max_seq_length,
                    overwrite_cache=self.hparams.overwrite_cache,
                    )

        val = TextDatasetForNextSentencePrediction(
                    tokenizer,
                    file_path='data/dnc_debate_val_for_NSP.tsv',
                    block_size=self.hparams.max_seq_length,
                    overwrite_cache=self.hparams.overwrite_cache,
                    )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            max_length=self.hparams.max_seq_length,
        )

        self.train = train
        self.val = val
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.hparams.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.hparams.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class PresForNSP(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        padding = "max_length" if self.hparams.pad_to_max_length else False
        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        train = TextDatasetForNextSentencePrediction(
                    tokenizer,
                    file_path='data/pres_debate_train_for_NSP.tsv',
                    block_size=self.hparams.max_seq_length,
                    overwrite_cache=self.hparams.overwrite_cache,
                    )

        val = TextDatasetForNextSentencePrediction(
                    tokenizer,
                    file_path='data/pres_debate_val_for_NSP.tsv',
                    block_size=self.hparams.max_seq_length,
                    overwrite_cache=self.hparams.overwrite_cache,
                    )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            max_length=self.hparams.max_seq_length,
        )

        self.train = train
        self.val = val
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.hparams.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.hparams.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class DNCForClassification(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.hparams.update(hparams)

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

    def prepare_data(self):
        pass

    def setup(self, stage):
        _train = pd.read_csv('data/dnc_debate_train_for_SC.tsv', sep = '\t', usecols = ['label','text'])
        _validation = pd.read_csv('data/dnc_debate_val_for_SC.tsv', sep = '\t', usecols = ['label','text'])

        print(_train.text.map(len).min())
        print(_validation.text.map(len).min())

        features = Features(
                    {
                        'label': Value(dtype='int32', id=None),
                        'text': Value(dtype='string', id=None),
                    }
                    )

        train = Dataset.from_pandas(_train, features=features, split='train')
        validation = Dataset.from_pandas(_validation, features=features, split='validation')

        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
            is_fast=False,
        )

        text = "text"

        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if self.hparams.pad_to_max_length else False

        def encode(examples):
            # Remove empty lines
            # examples[text] = [line for line in examples[text]
            #                     if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples[text],
                padding=True,  # padding
                truncation=True,
                max_length=self.hparams.max_seq_length,
                return_special_tokens_mask=False,
            )

        tokenize = {}

        tokenize["train"] = train.map(
            encode,
            batched=True,
            num_proc=self.hparams.num_workers,
            remove_columns=["text"],
            load_from_cache_file=not self.hparams.overwrite_cache,
        )

        tokenize["validation"] = validation.map(
            encode,
            batched=True,
            num_proc=self.hparams.num_workers,
            remove_columns=["text"],
            load_from_cache_file=not self.hparams.overwrite_cache,
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            max_length=self.hparams.max_seq_length,
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator


    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.hparams.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.hparams.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class PresForClassification(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.hparams.update(hparams)

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

    def prepare_data(self):
        pass

    def setup(self, stage):
        _train = pd.read_csv('data/pres_debate_train_for_SC.tsv', sep = '\t', usecols = ['label','text'])
        _validation = pd.read_csv('data/pres_debate_val_for_SC.tsv', sep = '\t', usecols = ['label','text'])

        features = Features(
                    {
                        'label': Value(dtype='int32', id=None),
                        'text': Value(dtype='string', id=None),
                    }
                    )

        train = Dataset.from_pandas(_train, features=features, split='train')
        validation = Dataset.from_pandas(_validation, features=features, split='validation')

        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        text = "text"

        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if self.hparams.pad_to_max_length else False

        def encode(examples):
            # Remove empty lines
            # examples[text] = [line for line in examples[text]
            #                     if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples[text],
                padding=padding,
                truncation=True,
                max_length=self.hparams.max_seq_length,
                return_special_tokens_mask=False,
            )

        tokenize = {}

        tokenize["train"] = train.map(
            encode,
            batched=True,
            num_proc=self.hparams.num_workers,
            remove_columns=["text"],
            load_from_cache_file=not self.hparams.overwrite_cache,
        )

        tokenize["validation"] = validation.map(
            encode,
            batched=True,
            num_proc=self.hparams.num_workers,
            remove_columns=["text"],
            load_from_cache_file=not self.hparams.overwrite_cache,
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            max_length=self.hparams.max_seq_length,
        )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size = self.hparams.train_batch_size,
                          shuffle = True,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size = self.hparams.eval_batch_size,
                          shuffle = False,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


if __name__ == "__main__":
    # swda = SWDA(hparams)
    # swda.setup(None)
    # print(swda.train[0])
    # print(swda.val[0])
    # loader = swda.val_dataloader()
    # batch = next(iter(loader))

    # mono_dia = DNCMonologuePresMonologue(hparams)
    # mono_dia.setup(None)
    # print(mono_dia.train[0])
    # print(mono_dia.val[0])
    # loader = mono_dia.val_dataloader()
    # batch = next(iter(loader))

    data = DNCForClassification(hparams)
    data.setup(None)
    # print(mono_dia.train[0])
    # print(mono_dia.val[0])
    # loader = data.val_dataloader()
    # batch = next(iter(loader))
    # print(batch)
    # print(batch["input_ids"].shape)
    # print(batch["token_type_ids"].shape)
    # print(batch["attention_mask"].shape)
    # print(batch["labels"].shape)