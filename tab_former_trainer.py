import os
import torch
import random
import argparse
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from Scripts.data import Data, TransactionData
from Scripts.utils import random_split_dataset
from Scripts.tab_former_dl import CustomTrainer
from Scripts.custom_trainer import load_model, train
from Scripts.tabformer_models import TabFormerBertLM
from Scripts.datacollator import TransDataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling, TrainingArguments, ProgressCallback, PrinterCallback

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./Data')
    parser.add_argument("--model_dir", type=str, default='./Models')
    parser.add_argument("--save_dir", type=str, default='/content/drive/MyDrive/IBM_Dataset/checkpoints')
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--nbins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--field_hs", type=int, default=768)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=3)
    parser.add_argument("--start_step", type=int, default=3)
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument("--return_labels", action='store_true')
    parser.add_argument("--skip_user", action='store_true')
    parser.add_argument("--flatten", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--load_all", action='store_true')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    return parser.parse_args()


def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    ids = {576,  102, 1429,  486, 1394, 1996, 1982, 1069,  341,  252, 1844,
        557, 1514, 1585,  266, 1419, 1515, 1722, 1060,   98, 1911, 1278,
       1739,   41,  759,  946, 1096,  185,  970, 1983,  764,  854,  257,
       1129,  605, 1685,  923, 1271, 1572,   81,  499,  680, 1792,  395,
        814, 1029,  996, 1680, 1399,  619,  624, 1539,  625, 1357,  262,
       1324, 1766,  549, 1126,  138,  817, 1304,  987, 1423,  417, 1731,
        683, 1681, 1378,  913, 1330, 1338,  220, 1211, 1532,  446,  504,
        100, 1101, 1880,  320,  838, 1327, 1002,  780,  945, 1196, 1122,
       1885, 1528,  413, 1248,  615, 1607,  190,  914,  897,  813, 1709,
       1019, 1604, 1425, 1487, 1064}

    dataset = Data(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        nbins=args.nbins,
        return_labels=args.return_labels,
        skip_user=args.skip_user,
        flatten=args.flatten,
        ids=ids
    )

    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    # split dataset into train, val, test [0.7. 0.1, 0.2]
    totalN = len(dataset.data)
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
        args.return_labels,
        args.load_all
    )
    eval_dataset = TransactionData(
        dataset.data[lengths[0]: lengths[0] + lengths[1]],
        args.data_dir,
        args.seq_len,
        args.flatten,
        args.return_labels,
        args.load_all
    )
    # test_dataset = TransactionData(
    #     dataset.data[lengths[0] + lengths[1]:],
    #     args.data_dir,
    #     args.seq_len,
    #     args.flatten,
    #     args.return_labels,
    #     args.load_all
    # )

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

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=None, num_workers=2, collate_fn=data_collator)
    val_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, sampler=None, num_workers=2, collate_fn=data_collator)

    optimizer = AdamW(model.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.model, optimizer, start_epoch, start_step, prev_val_loss = load_model(
        os.path.join(args.save_dir, f'tabformer_epoch{args.start_epoch}_step{args.start_step}.pth'), 
        model.model, 
        optimizer
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model.model.to(device), optimizer, start_step, args.save_step, start_epoch, args.epochs, train_loader, val_loader, prev_val_loss, args.save_dir, device)


# def main(args):
#     seed = args.seed
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
    
#     ids = {576,  102, 1429,  486, 1394, 1996, 1982, 1069,  341,  252, 1844,
#         557, 1514, 1585,  266, 1419, 1515, 1722, 1060,   98, 1911, 1278,
#        1739,   41,  759,  946, 1096,  185,  970, 1983,  764,  854,  257,
#        1129,  605, 1685,  923, 1271, 1572,   81,  499,  680, 1792,  395,
#         814, 1029,  996, 1680, 1399,  619,  624, 1539,  625, 1357,  262,
#        1324, 1766,  549, 1126,  138,  817, 1304,  987, 1423,  417, 1731,
#         683, 1681, 1378,  913, 1330, 1338,  220, 1211, 1532,  446,  504,
#         100, 1101, 1880,  320,  838, 1327, 1002,  780,  945, 1196, 1122,
#        1885, 1528,  413, 1248,  615, 1607,  190,  914,  897,  813, 1709,
#        1019, 1604, 1425, 1487, 1064}

#     dataset = Data(
#         data_dir=args.data_dir,
#         model_dir=args.model_dir,
#         seq_len=args.seq_len,
#         stride=args.stride,
#         nbins=args.nbins,
#         return_labels=args.return_labels,
#         skip_user=args.skip_user,
#         flatten=args.flatten,
#         ids=ids
#     )

#     vocab = dataset.vocab
#     custom_special_tokens = vocab.get_special_tokens()

#     # split dataset into train, val, test [0.7. 0.1, 0.2]
#     totalN = len(dataset.data)
#     trainN = int(0.6 * totalN)
#     valtestN = totalN - trainN
#     valN = int(valtestN * 0.5)
#     testN = valtestN - valN
#     lengths = [trainN, valN, testN]

#     train_dataset = TransactionData(
#         dataset.data[:lengths[0]],
#         args.data_dir,
#         args.seq_len,
#         args.flatten,
#         args.return_labels,
#         args.load_all
#     )
#     eval_dataset = TransactionData(
#         dataset.data[lengths[0]: lengths[0] + lengths[1]],
#         args.data_dir,
#         args.seq_len,
#         args.flatten,
#         args.return_labels,
#         args.load_all
#     )
#     test_dataset = TransactionData(
#         dataset.data[lengths[0] + lengths[1]:],
#         args.data_dir,
#         args.seq_len,
#         args.flatten,
#         args.return_labels,
#         args.load_all
#     )
#     # train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

#     model = TabFormerBertLM(
#         special_tokens=custom_special_tokens,
#         vocab=vocab,
#         ncols=dataset.ncols,
#         field_hidden_size=args.field_hs
#     )

#     if args.flatten:
#         collactor_cls = DataCollatorForLanguageModeling
#     else:
#         collactor_cls = TransDataCollatorForLanguageModeling
    
#     data_collator = collactor_cls(tokenizer=model.tokenizer, mlm=True, mlm_probability=args.mlm_prob)

#     model_path = os.path.join(args.model_dir, 'checkpoints')
#     training_args = TrainingArguments(
#         output_dir=model_path,
#         num_train_epochs=args.epochs,
#         save_steps=args.save_step,
#         do_train=args.do_train,
#         do_eval=args.do_eval,
#         evaluation_strategy="steps",
#         prediction_loss_only=True,
#         overwrite_output_dir=True,
#         optim='adamw_torch',
#         disable_tqdm=False,
#         load_best_model_at_end=True
#     )
#     trainer = CustomTrainer(
#         model=model.model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#     )
#     trainer.remove_callback(PrinterCallback)
#     trainer.train()

if __name__ == '__main__':
    args = get_args()
    main(args)
