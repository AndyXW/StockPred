import argparse
import json
import os
import time
import datetime
import wandb
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

from models import AlphaNetV3
from dataset import StockDataset
from utils import AverageMeter
from data import TrainValData, TimeSeriesData
from loss import ClassBCELoss, CELossWithPenalty, JSD

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


def get_args_parser():
    parser = argparse.ArgumentParser('AlphaNet Prediction', add_help=False)
    parser.add_argument('--batch_size', default=4096, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # Optimizer parameters
    parser.add_argument('--optim', type=str, default='adam',
                        help='optim method')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--hidden_dim', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--loss', type=str, default='with_penalty')
    parser.add_argument('--p_weight', default=5.0, type=float)
    
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/work/bd/summer2022/insample/datacache', type=str,
                        help='dataset path')
    

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--save_path', default='results/adam_cls_lr0.01', type=str,
                        help='save result path')
    parser.add_argument('--save_freq', default=50, type=int, help='save frequency')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids')
    parser.add_argument('--jsd', action='store_true')

    return parser


def preprocess_y(df_y):
    df_y_mean = df_y.mean(axis=1)
    df_y_std = df_y.std(axis=1)
    max_5_sigma = df_y_mean + 5 * df_y_std
    min_5_sigma = df_y_mean - 5 * df_y_std
    
    for row in range(df_y.shape[0]):
        df_y.iloc[row][df_y.iloc[row]>max_5_sigma.values[row]] = np.nan
        df_y.iloc[row][df_y.iloc[row]<min_5_sigma.values[row]] = np.nan

    df_y_q_33 = df_y.quantile(q=0.33, axis=1)
    df_y_q_66 = df_y.quantile(q=0.67, axis=1)

    for row in range(df_y.shape[0]):
    # row = 700
        rank_0 = (df_y.iloc[row] <= df_y_q_33.values[row])
        rank_1 = ((df_y.iloc[row]>df_y_q_33.values[row]) & (df_y.iloc[row]<=df_y_q_66.values[row]))
        rank_2 = (df_y.iloc[row]>df_y_q_66.values[row])

        df_y.iloc[row][rank_0] = 0
        df_y.iloc[row][rank_1] = 1
        df_y.iloc[row][rank_2] = 2


def to_one_hot(y, num_cls=3):
    one_hot_label = np.empty((len(y), num_cls))
    one_hot_label[:] = np.NaN
    y = y.astype(np.int64)
    for row in range(len(one_hot_label)):
        if y[row] >= 0 and y[row] < num_cls:
            label = np.zeros(num_cls)
            label[y[row]] = 1
            one_hot_label[row] = label
    return one_hot_label
    

def main(args):
    device = torch.device('cuda')

    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # build dataset
    dailydata = os.path.join(args.data_path, 'dailydata')
    labeldata = os.path.join(args.data_path, 'labeldata')

    df_close = pd.read_parquet(os.path.join(dailydata, 'close.parquet'))
    df_high = pd.read_parquet(os.path.join(dailydata, 'high.parquet'))
    df_low = pd.read_parquet(os.path.join(dailydata, 'low.parquet'))
    df_open = pd.read_parquet(os.path.join(dailydata, 'open.parquet'))
    df_tvrvalue = pd.read_parquet(os.path.join(dailydata, 'tvrvalue.parquet'))
    df_tvrvolume = pd.read_parquet(os.path.join(dailydata, 'tvrvolume.parquet'))

    df_y = pd.read_parquet(os.path.join(labeldata, 'Y_0.parquet'))
    preprocess_y(df_y=df_y)

    features = [df_open, df_high, df_low, df_tvrvalue, df_tvrvolume, df_close]
    stock_data_list = []

    stocks = df_open.columns.values
    for stock in stocks:
        one_stock_features = []
        for feature in features:
            one_stock_features.append(feature[stock].values[:-2].reshape(-1, 1))
        stock_np_features = np.concatenate(one_stock_features, axis=1)
        dates = feature.index.values[:-2]
        labels = to_one_hot(df_y[stock].values[1:])
        # print(stock_np_features.shape, dates.shape, labels.shape)
        stock_data_list.append(TimeSeriesData(dates=dates, data=stock_np_features, labels=labels))
    
    train_val_data = TrainValData(time_series_list=stock_data_list, train_length=800, validate_length=150, history_length=10, train_val_gap=10, sample_step=1)
    train, val, dates_info = train_val_data.get(20180102, order='by_date')
    # print(dates_info)
    
    train_dataset = StockDataset(stock_data=train[0], stock_label=train[1])
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataset = StockDataset(stock_data=val[0], stock_label=val[1])
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # set model
    model = AlphaNetV3(feat_dim=6, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=0.0, num_classes=3)
    model = model.to(device)

    #　set optimizer and criterion
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.MSELoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    elif args.loss == 'clsBCE':
        criterion = ClassBCELoss(num_classes=3, weights=[0.15, 0.05, 0.8])
    elif args.loss == 'with_penalty':
        criterion = CELossWithPenalty(weight=args.p_weight)
    elif args.loss == 'logitsCE':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 0.5, 1.0]).cuda())
    
    loss_jsd = JSD()
    away_p = torch.tensor([[0.4, 0.2, 0.4]])

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            ckpt = torch.load(args.resume)
            model.load_state_dict(ckpt['state_dict'])
            args.start_epoch = ckpt['epoch']
            optimizer.load_state_dict(ckpt['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, ckpt['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    print(f"Start training for {args.epochs} epochs\n")
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_ret = train_one_epoch(train_loader, model, criterion, loss_jsd, optimizer, epoch, device, args)
        val_ret = validate_one_epoch(val_loader, model, criterion, epoch, device, args)
        print(train_ret, val_ret)
        train_ret.update(val_ret)
        wandb.log(train_ret)
        scheduler.step()

        if val_ret['val_acc'] > best_acc:
            best_acc = val_ret['val_acc']
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, os.path.join(args.save_path, f"best_model.ckpt"))
    
    torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(args.save_path, f"checkpoint_{epoch+1}.ckpt"))
    


def train_one_epoch(train_loader, model, criterion, jsd, optimizer, epoch, device, args):
    model.train()

    losses = AverageMeter()
    acces = AverageMeter()

    for i, (feat, label) in enumerate(train_loader):
        feat = feat.to(device).to(torch.float32)
        label = label.to(device).to(torch.float32)

        output = model(feat)
        loss = criterion(output, label)

        if args.jsd:
            p = torch.tensor([[0.4, 0.2, 0.4]]).to(device).float()
            p = p.expand_as(output)
            loss_jsd = jsd(output, p)
            loss -= loss_jsd

        with torch.no_grad():
            prediction = output.argmax(dim=1)
            labelindex = label.argmax(dim=1)
            acc = accuracy_score(labelindex.cpu().numpy(), prediction.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch:[{:d}][{:d}/{:d}], loss {:.4f}, acc {:.4f}".format(epoch, i+1, len(train_loader), loss.item(), acc))
        losses.update(loss.item(), feat.size(0))
        acces.update(acc, feat.size(0))
    
    ret = {'train_loss': losses.avg,
           'train_acc': acces.avg}

    return ret


def validate_one_epoch(val_loader, model, criterion, epoch, device, args):
    model.eval()

    losses = AverageMeter()
    acces = AverageMeter()
    with torch.no_grad():

        for i, (feat, label) in enumerate(val_loader):
            feat = feat.to(device).to(torch.float32)
            label = label.to(device).to(torch.float32)

            output = model(feat)
            loss = criterion(output, label)

            prediction = output.argmax(dim=1)
            labelindex = label.argmax(dim=1)
            acc = accuracy_score(labelindex.cpu().numpy(), prediction.cpu().numpy())

            print("Epoch:[{:d}][{:d}/{:d}], loss {:.4f}, acc {:.4f}".format(epoch, i+1, len(val_loader), loss.item(), acc))
            losses.update(loss.item(), feat.size(0))
            acces.update(acc, feat.size(0))
    
    ret = {'val_loss': losses.avg,
           'val_acc': acces.avg}

    return ret
    



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    if args.loss == 'with_penalty':
        run_name = 'baseline_181_pce_tm_lr0.001'
    else:
        run_name = 'baseline_split_181_{}_{}_0.5'.format(args.loss, args.lr)
    args.save_path = os.path.join('results', run_name)
    os.makedirs(args.save_path, exist_ok=True)
    wandb.init(project='BaiLu', name=run_name, settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args)
    main(args)

