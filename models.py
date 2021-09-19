import logging
import os
import csv
from argparse import Namespace
from pathlib import Path

from sklearn.metrics import matthews_corrcoef

import torch 
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from transformers import (
    AutoConfig,
    AutoModelForNextSentencePrediction,
    AutoModelForSequenceClassification,
    BertModel, 
    BertForMaskedLM, 
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    # AdamW, 
)

from lightning_base import BaseTransformer

logger = logging.getLogger('trainer.models')

# I think hparams does not need to be an explicit argument; confirm

class MaskedLM(BaseTransformer):
    """An instance of the `BertForMaskedLM` model."""
    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        model = BertForMaskedLM(config)

        super().__init__(hparams, config=config, model=model)

        # model.resize_token_embeddings(len(self.tokenizer))

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]
        
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # the mean training loss for all the examples in the batch.
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()

        loss = loss_mean.detach().cpu()
        perplexity = torch.exp(loss_mean.clone().detach()).cpu()


        self.logger.experiment.add_scalar("Loss/Train",
                                            loss,
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Train",
                                            perplexity,
                                            self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]
        
        return {"val_loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        # the mean validation loss for all the examples in the batch.
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        results = {
            **{"val_loss": val_loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(val_loss_mean.clone().detach()).cpu()}
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("Loss/Validation",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Validation",
                                            logs["perplexity"],
                                            self.current_epoch,
        )

        # val_folder = os.path.join(self.hparams.output_dir, "results")
        # # Path(val_folder).mkdir(parents=True, exist_ok=True)
        # filename = self.hparams.model + "_" + self.hparams.dataset_name + ".txt"
        # val_results = os.path.join(val_folder, filename)

        # loss = logs["val_loss"].detach().cpu().numpy()
        # perplexity = logs["perplexity"].detach().cpu().numpy()
        # record = [loss, perplexity] 

        # if self.current_epoch == (self.hparams.max_epochs - 1):
        #     # with open(val_results, 'a', newline='') as f:
        #     #     writer = csv.writer(f, delimiter=',')
        #     #     writer.writerow(record)
        #     #     f.close()
        #     with open(val_results, "a") as writer:
        #         writer.write("***** Seed - {} *****".format(self.hparams.seed))
        #         for key in sorted(logs.keys()):
        #             writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))
        
    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`       
        self.logger.experiment.add_scalar("Loss/Test",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Test",
                                            logs["perplexity"],
                                            self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--line_by_line', 
            action='store_true', 
            default=True,
            help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
        )
        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=True,
            help="Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        parser.add_argument(
            '--mlm_probability', 
            type=float, 
            default=0.15,
            # help= ,
        )
        return parser


class NextSentencePrediction(BaseTransformer):
    """An instance of the `BertForNextSentencePrediction` model."""
    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        model = BertForNextSentencePrediction(config)

        super().__init__(hparams, config=config, model=model)

        # model.resize_token_embeddings(len(self.tokenizer))

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # the mean training loss for all the examples in the batch.
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()

        loss = loss_mean.detach().cpu()
        perplexity = torch.exp(loss_mean.clone().detach()).cpu()

        self.logger.experiment.add_scalar("Loss/Train",
                                            loss,
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Train",
                                            perplexity,
                                            self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]
        
        return {"val_loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        # the mean validation loss for all the examples in the batch.
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        results = {
            **{"val_loss": val_loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(val_loss_mean.clone().detach()).cpu()}
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("Loss/Validation",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Validation",
                                            logs["perplexity"],
                                            self.current_epoch,
        )

        val_folder = os.path.join(self.hparams.output_dir, "results")
        Path(val_folder).mkdir(parents=True, exist_ok=True)
        filename = self.hparams.model + "_" + self.hparams.dataset_name + ".txt"
        val_results = os.path.join(val_folder, filename)

        record = [logs["val_loss"], logs["perplexity"]]
        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(val_results, "a") as writer:
                writer.writerow(record)
    
    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`       
        self.logger.experiment.add_scalar("Loss/Test",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Test",
                                            logs["perplexity"],
                                            self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=True,
            help="Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        return parser


class MLM_NSP(BaseTransformer):
    """An instance of the `BertForNextSentencePrediction` for running on a model
     already fine-tuned on the `BertForMaskedLM` task."""
    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        # load HF `pretrained` save of the MLM
        model = AutoModelForNextSentencePrediction.from_pretrained(hparams.mlm_path, config=config)

        super().__init__(hparams, config=config, model=model)

        # model.resize_token_embeddings(len(self.tokenizer))

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # the mean training loss for all the examples in the batch.
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()

        loss = loss_mean.detach().cpu(),  
        perplexity = torch.exp(loss_mean.clone().detach()).cpu()

        self.logger.experiment.add_scalar("Loss/Train",
                                            loss,
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Train",
                                            perplexity,
                                            self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]
        
        return {"val_loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        # the mean validation loss for all the examples in the batch.
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        results = {
            **{"val_loss": val_loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(val_loss_mean.clone().detach()).cpu()}
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("Loss/Validation",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Validation",
                                            logs["perplexity"],
                                            self.current_epoch,
        )

        val_folder = os.path.join(self.hparams.output_dir, "results")
        Path(val_folder).mkdir(parents=True, exist_ok=True)
        filename = self.hparams.model + "_" + self.hparams.dataset_name + ".txt"
        val_results = os.path.join(val_folder, filename)

        record = [logs["val_loss"], logs["perplexity"]]
        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(val_results, "a") as writer:
                writer.writerow(record)
    
    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`       
        self.logger.experiment.add_scalar("Loss/Test",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Test",
                                            logs["perplexity"],
                                            self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=True,
            help="Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        parser.add_argument(
            "--mlm_path",
            default="./output/best_tfmr",
            type=str,
            help="Path to the Hf `pretrained` save of the MLM",
        )
        return parser


class ProbingTask(BaseTransformer):
    """An instance of the `BertForSequenceClassification` model."""
    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.lr = hparams.lr
        self.num_labels=2
        config = AutoConfig.from_pretrained(hparams.model_name_or_path, num_labels=self.num_labels)
        model = BertForSequenceClassification(config=config)

        super().__init__(hparams, config=config, model=model)

        self.loss = CrossEntropyLoss()

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"]
        }
        labels = batch["labels"]

        # forward pass
        outputs = self(**inputs)
        logits = outputs[0]
        loss = self.loss(logits, labels)

        return {"loss": loss}

    def training_epoch_end(self,outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()

        loss = loss_mean.detach().cpu()
        perplexity = torch.exp(loss_mean.detach().cpu())

        self.logger.experiment.add_scalar("Loss/Train",
                                            loss,
                                            self.global_step,
        )  
        self.logger.experiment.add_scalar("Perplexity/Train",
                                            perplexity,
                                            self.global_step,
        )

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"]
        }
        labels = batch["labels"]

        # forward pass
        outputs = self(**inputs)
        logits = outputs[0]
        loss = self.loss(logits, labels)

        return {"loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()

        loss = loss_mean.detach().cpu()
        perplexity = torch.exp(loss_mean.detach().cpu())
        
        ret = {"log": {"loss": loss, "perplexity": perplexity}}
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("Loss/Validation",
                                            logs["loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Validation",
                                            logs["perplexity"],
                                            self.current_epoch,
        )
        
        # val_folder = os.path.join(self.hparams.output_dir, self.hparams.model + "_" + self.hparams.dataset_name)
        # val_results = os.path.join(val_folder, "val_results.txt")
        # Path(val_folder).mkdir(parents=True, exist_ok=True)

        # if self.current_epoch == (self.hparams.max_epochs - 1):
        #     with open(val_results, "a") as writer:
        #         for key in sorted(logs.keys()):
        #             writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("Loss/Validation",
                                            logs["loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Validation",
                                            logs["perplexity"],
                                            self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.lr,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--line_by_line', 
            action='store_true', 
            default=True,
            help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
        )
        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=True,
            help="Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        parser.add_argument(
            '--mlm', 
            action='store_true', 
            default=False,
            # help= ,
        )
        parser.add_argument(
            '--mlm_probability', 
            type=float, 
            default=0.15,
            # help= ,
        )
        # add ckpt_path
        # parser.add_argument(
        #     '--ckpt_path', 
        #     type=str, 
        #     default="",
        #     # help= ,
        # )

        return parser


class FrozenBertModel(BertModel):
    """An instance of the `BertModel` with all transformer layers frozen."""
    def __init__(self, config):
        super().__init__(config)
        modules = [*self.encoder.layer[:12]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False


class FrozenBertForMaskedLM(BertForMaskedLM):
    """An instance of the `BertForMaskedLM` with self.bert overloaded with FrozenBertModel."""
    def __init__(self, config):
        super().__init__(config)
        self.bert = FrozenBertModel(config)


class AdaptiveFineTuning(BaseTransformer):
    """An implementation of BERT configured to evaluate the impact of MLM on the embedding layer."""
    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        model = FrozenBertForMaskedLM(self.config)

        super().__init__(hparams, config=config, model=model)

        model.resize_token_embeddings(len(self.tokenizer))

        # if hparams.do_train or hparams.fast_dev_run:
        #     train = load_dataset(hparams.dataset_name, hparams.subset_name, split='train', cache_dir=hparams.cache_dir, script_version="master")
        #     self.dataset_size = len(train)
        # if hparams.do_predict:
        #     test = load_dataset(hparams.dataset_name, hparams.subset_name, split='test', cache_dir=hparams.cache_dir, script_version="master")
        #     self.dataset_size = len(test)

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]

        # self.logger.experiment.add_scalar("Perplexity/Train",
        #                                     perplexity,
        #                                     self.global_step,
        # )
        
        return {"loss": loss}

    def training_epoch_end(self,outputs):
        # the mean training loss for all the examples in the batch.
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu()

        results = {
            **{"loss": loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(loss_mean.clone().detach()).detach().cpu()}
        }

        self.logger.experiment.add_scalar("Loss/Train",
                                            results["loss"],
                                            self.current_epoch,
        )

        self.logger.experiment.add_scalar("Perplexity/Train",
                                            results["perplexity"],
                                            self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]

        return {"val_loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        # the mean validation loss for all the examples in the batch.
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()

        results = {
            **{"val_loss": val_loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(val_loss_mean.clone().detach()).detach().cpu()}
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.logger.experiment.add_scalar("Loss/Validation",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Validation",
                                            logs["perplexity"],
                                            self.current_epoch,
        )

        val_folder = os.path.join(self.hparams.output_dir, "results")
        Path(val_folder).mkdir(parents=True, exist_ok=True)
        filename = self.hparams.model + "_" + self.hparams.dataset_name + ".txt"
        val_results = os.path.join(val_folder, filename)

        record = [logs["val_loss"], logs["perplexity"]]
        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(val_results, "a") as writer:
                writer.writerow(record)

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.logger.experiment.add_scalar("Loss/Test",
                                            logs["val_loss"],
                                            self.current_epoch,
        )

        self.logger.experiment.add_scalar("Perplexity/Test",
                                            logs["perplexity"],
                                            self.current_epoch,
        )
        # self.log("Loss/Validation", logs["val_loss"], logger=True)
        # self.log("Perplexity/Validation", logs["perplexity"], logger=True)


    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,  # Learning rate set to 0.13182567385564073
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--line_by_line', 
            action='store_true', 
            default=False,
            help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
        )
        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=False,
            help="Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        parser.add_argument(
            '--mlm_probability', 
            type=float, 
            default=0.15,
            # help= ,
        )
        return parser


# if __name__ == "__main__":
    # validate that transform layers are frozen
    # config = AutoConfig.from_pretrained('bert-base-uncased')
    # model = FrozenBertModel(config)
    # modules = [model.embeddings, model.encoder.layer, model.pooler]
    # print(model)
    # for module in modules:
    #     for param in module.parameters():
    #         print(param.requires_grad)

    # model_pt = FrozenBertForPreTraining(config)
    # modules_pt = [model_pt.bert.embeddings, model_pt.bert.encoder.layer, model_pt.bert.pooler, model_pt.cls]
    # print(model_pt)
    # for module in modules_pt:
    #     for param in module.parameters():
    #         print(param.requires_grad)
