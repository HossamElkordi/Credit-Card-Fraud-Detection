import os
import torch
import random
import argparse
import numpy as np
from Scripts.data import Data, TransactionData
from Scripts.utils import random_split_dataset
from Scripts.tab_former_dl import CustomTrainer
from Scripts.tabformer_models import TabFormerBertLM
from Scripts.datacollator import TransDataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling, TrainingArguments, ProgressCallback

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./Data')
    parser.add_argument("--model_dir", type=str, default='./Models')
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--nbins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--field_hs", type=int, default=768)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument("--return_labels", action='store_true')
    parser.add_argument("--skip_user", action='store_true')
    parser.add_argument("--flatten", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    return parser.parse_args()


def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    dataset = Data(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        nbins=args.nbins,
        return_labels=args.return_labels,
        skip_user=args.skip_user,
        flatten=args.flatten
    )

    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    # split dataset into train, val, test [0.7. 0.1, 0.2]
    totalN = len(dataset)
    trainN = int(0.6 * totalN)
    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN
    lengths = [trainN, valN, testN]
    train_dataset = TransactionData(
        dataset.data[:lengths[0]],
        args.data_dir,
        args.seq_len,
        args.flatten,
        args.return_labels
    )
    eval_dataset = TransactionData(
        dataset.data[lengths[0]: lengths[0] + lengths[1]],
        args.data_dir,
        args.seq_len,
        args.flatten,
        args.return_labels
    )
    train_dataset = TransactionData(
        dataset.data[lengths[0] + lengths[1]:],
        args.data_dir,
        args.seq_len,
        args.flatten,
        args.return_labels
    )
    # train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

    model = TabFormerBertLM(
        special_tokens=custom_special_tokens,
        vocab=vocab,
        ncols=dataset.ncols,
        field_hidden_size=args.field_hs
    )

    if args.flatten:
        collactor_cls = DataCollatorForLanguageModeling
    else:
        collactor_cls = TransDataCollatorForLanguageModeling
    
    data_collator = collactor_cls(tokenizer=model.tokenizer, mlm=True, mlm_probability=args.mlm_prob)

    model_path = os.path.join(args.model_dir, 'checkpoints')
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=args.epochs,
        save_steps=args.save_step,
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy="steps",
        prediction_loss_only=True,
        overwrite_output_dir=True,
        optim='adamw_torch',
        disable_tqdm=False,
        load_best_model_at_end=True
    )
    trainer = CustomTrainer(
        model=model.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

if __name__ == '__main__':
    args = get_args()
    main(args)
