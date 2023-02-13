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
    
    ids = {1869, 1897, 1594, 1727, 1627, 1527, 1474, 1418, 1307, 1807, 1021,
        1997,  761,  553,  752,  288,  533,  151,  129,  165,  529,   16,
           3,  802,  521,   42,  440, 1898, 1397,  786, 1893,   91, 1034,
        1428,  453, 1146,  209, 1778,  398, 1534, 1622,   22,  291,  952,
        1192, 1564,  539, 1328,   89,  359, 1305,  330,  420,  644,  836,
         256,  688,  234,  574, 1188,  953, 1204, 1508, 1420, 1524,  139,
         699, 1083, 1349, 1343, 1091,   62,  748,  794, 1102, 1103,  744,
        1393,  473,   97, 1291, 1896, 1584,  182,  128,  540, 1725,  943,
         899,  490, 1601,  494, 1589,  428,  938,  853, 1224,  515,  883,
         823,  692, 1823, 1417, 1887, 1777, 1938, 1361, 1306, 1319,   53,
        1755, 1759, 1977,  966,  657,  711, 1115, 1895,  876, 1882,  718,
        1990, 1017,  150,  720,  751, 1478,  463,  271, 1180, 1693,  934,
        1266, 1656, 1649,  418,  864, 1169,  672,  979,  215, 1533,  595,
         901,  559,  714,  569, 1657,  520, 1962, 1934, 1038, 1107, 1006,
        1841, 1809, 1802, 1567, 1773,  268,  299, 1736,  318, 1790,  782,
         462,  907,  896, 1597,  452,  509, 1258,  729,   75,  881, 1913,
        1877,  365,  188,   61,   77,  809,  503, 1158, 1735,  662, 1764,
         831, 1223, 1178, 1470,  776,   49,   55, 1945, 1929,  621,  647,
        1820,  726, 1918, 1519, 1908,  544,  180, 1522, 1827, 1444, 1734,
           0, 1166, 1008, 1106, 1049,  827, 1052, 1173,  790,  885,  920,
        1236,  929,  693, 1547, 1545,  172,  684,  677,  822,  661,  884,
         376, 1452,  976,  697, 1760,  967,  959, 1199, 1710,  344, 1800,
        1403, 1660, 1563, 1963, 1370,  747,   29,  791,  641,  578,  362,
         758,  768,  797,  760, 1689, 1690, 1024, 1730, 1482, 1761, 1228,
        1526, 1488, 1011,  186, 1579, 1242,  137, 1575, 1881, 1323,  286,
          19, 1254,  849,  156,  255, 1683,  705,  367, 1392,  829,  206,
        1816, 1813,  223,  481, 1598, 1529,  980,  421,  824, 1315, 1791,
         931,  690, 1697, 1085,  598, 1272, 1490, 1400,  101, 1033,   73,
        1794, 1379,  240, 1884,  187,  787, 1185, 1092,   15, 1241, 1825,
         464, 1154, 1150, 1100, 1629,   17,  736, 1201,  433, 1398,  406,
         857, 1675,   36,  254,  629, 1955,  285,  114,  880, 1634, 1362,
           1, 1903, 1206,  909,  319,  924, 1215, 1698, 1726,  488,  450,
        1354, 1558, 1298, 1283, 1477,  579,  620, 1855, 1249, 1958, 1590,
         502, 1837, 1593, 1520, 1641, 1353, 1235, 1063, 1104, 1581, 1297,
         338, 1483,  243,  566,  731, 1636,  606,  304,  949,   74, 1290,
        1174, 1662, 1780, 1723,  994, 1576, 1347, 1320, 1570,  111,  807,
         991, 1111, 1247, 1523,  591, 1497, 1537, 1530,  275,   92, 1234,
         973, 1099,  239, 1041,  588, 1167,   50,   47,  193,  687,  439,
        1062, 1448, 1292,  489,  841, 1799, 1753,  832, 1136, 1032,  311,
         487,  170,  242, 1926, 1441,   44,  804, 1402,  307,  331, 1059,
         424, 1691, 1109,  385, 1468, 1212, 1422,  986, 1010,  448, 1894,
        1776,  112, 1071, 1889,  806,  848,  292, 1669, 1933, 1446, 1782,
        1036, 1081, 1499,  343,  545,  874,  393, 1588,  333,  510, 1075,
        1993, 1295,  614,  491,  531, 1003,  556,  103,  583,  842, 1717,
        1742,  309, 1191,  149,  658,  212, 1163,  390,  773,    2, 1334,
        1571, 1876,  580,  207,  562, 1286,  861,  109, 1852, 1020, 1366,
         124,  634, 1473, 1946, 1342, 1415, 1311,  928, 1917,  482,  983,
        1932,  461, 1098, 1779,  466, 1117,  777,  497, 1810,  799,  294,
        1789,   66,  161,  940,  908,  866, 1744, 1476, 1546,  576,  102,
        1429,  486, 1394, 1996, 1982, 1069,  341,  252, 1844,  557, 1514,
        1585,  266, 1419, 1515, 1722, 1060,   98, 1911, 1278, 1739,   41,
         759,  946, 1096,  185,  970, 1983,  764,  854,  257, 1129,  605,
        1685,  923, 1271, 1572,   81,  499,  680, 1792,  395,  814, 1029,
         996, 1680, 1399,  619,  624, 1539,  625, 1357,  262, 1324, 1766,
         549, 1126,  138,  817, 1304,  987, 1423,  417, 1731,  683, 1681,
        1378,  913, 1330, 1338,  220, 1211, 1532,  446,  504,  100, 1101,
        1880,  320,  838, 1327, 1002,  780,  945, 1196, 1122, 1885, 1528,
         413, 1248,  615, 1607,  190,  914,  897,  813, 1709, 1019, 1604,
        1425, 1487, 1064}

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


if __name__ == '__main__':
    args = get_args()
    main(args)
