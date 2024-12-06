# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

# def get_val_data(name, data_dir, height, width, batch_size, workers):
#     root = osp.join(data_dir, name)
#
#     dataset = datasets.create(name, root)
#
#     normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#
#     test_transformer = T.Compose([
#              T.Resize((height, width), interpolation=3),
#              T.ToTensor(),
#              normalizer
#          ])
#
#     test_loader = DataLoader(
#         Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
#                      root=dataset.images_dir, transform=test_transformer),
#         batch_size=batch_size, num_workers=workers,
#         shuffle=False, pin_memory=True)
#     return dataset, test_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader
def get_val_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    # if testset is None:
    testset = set(dataset.query)

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader
# def get_val_loader(args, dataset, height, width, batch_size, workers, num_instances, iters, testset=None, no_cam=False):
#     normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#
#     test_transformer = T.Compose([
#         T.Resize((height, width), interpolation=3),
#         T.ToTensor(),
#         normalizer
#     ])
#
#     test_set = sorted(dataset.query) if testset is None else sorted(testset)
#     rmgs_flag = num_instances > 0
#     if rmgs_flag:
#         if no_cam:
#             sampler = RandomMultipleGallerySamplerNoCam(test_set, num_instances)
#         else:
#             sampler = RandomMultipleGallerySampler(test_set, num_instances)
#     else:
#         sampler = None
#     test_loader = IterLoader(
#         DataLoader(Preprocessor(test_set, root=dataset.images_dir, transform=test_transformer),
#                    batch_size=batch_size, num_workers=workers, sampler=sampler,
#                    shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
#
#     return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            if epoch != 0:
            #     # cluster_features = memory.cluster_features.clone()
            #     # features = features.cuda()
            #     # outlier_f = memory.outliers.clone()
                features = torch.cat([features, cluster_features], dim=0)
                features = torch.cat([features, outlier_f], dim=0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)  # rerank_dist (12936,12936)

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                eps_loose = eps + args.eps_gap
                print('Clustering criterion: eps: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_loose))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
                cluster_loose = DBSCAN(eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)


            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)

            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
#-----------------------------------
            indexs = []
            for index in range(len(pseudo_labels_loose)):
                if pseudo_labels_loose[index] == pseudo_labels[index]:
                    continue
                else:
                    if pseudo_labels[index] == -1:
                        indexs.append(index)
                    else:
                        continue

            max_id = np.max(pseudo_labels)
            for ii in indexs:
                pseudo_labels[ii] = max_id + 1
                max_id = max_id + 1
#--------------------------------

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):   # labels = 12936  , features = (12936,2048)
            centers = collections.defaultdict(list)
            outliers = []
            L = 0
            for i, label in enumerate(labels):
                if label != -1:
                    centers[labels[i]].append(features[i])
                else :
                    outliers.append(features[i])
                    L += 1

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())    # centers = list 301
            ]
            centers = torch.stack(centers, dim=0)   # (301*2048)
            outlier_f = torch.stack(outliers).squeeze()
            return centers,outlier_f , L   # inputs = list 3209  label_list = list 3208

        cluster_features, outlier_f, num_outliers = generate_cluster_features(pseudo_labels, features)

#----------------------
        outlier_f_select = features[indexs]
        hybrid_features = torch.cat([cluster_features, outlier_f_select], dim=0)
#-----------------------------------
        del cluster_loader, features

        # Create hybrid memory
        if args.memorybank == 'CM':
            memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, mode=args.memorybank, hard_weight=args.hard_weight).cuda()
            memory.features = F.normalize(cluster_features, dim=1).cuda()
        elif args.memorybank=='CMhybrid':
            memory = ClusterMemory(model.module.num_features, num_cluster+ len(indexs), temp=args.temp,
                               momentum=args.momentum, mode=args.memorybank, hard_weight=args.hard_weight).cuda()
            memory.features = F.normalize(hybrid_features, dim=1).cuda()  # cluster_features = tensor(280,2048)
        elif args.memorybank=='CMhybrid_v2':
            memory = ClusterMemory(model.module.num_features, num_cluster + len(indexs), temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, hard_weight=args.hard_weight).cuda()
            memory.features = F.normalize(hybrid_features.repeat(2, 1), dim=1).cuda()
        elif args.memorybank=='CMhybrid_v3':
            memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, hard_weight=args.hard_weight).cuda()
            memory.features = F.normalize(cluster_features.repeat(2, 1), dim=1).cuda()
        elif args.memorybank=='CMhybrid_v4':
            memory = ClusterMemory(model.module.num_features, num_cluster + len(indexs), temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, hard_weight=args.hard_weight).cuda()
            memory.features = F.normalize(hybrid_features.repeat(3, 1), dim=1).cuda()


        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters, {} outliers.'.format(epoch, num_cluster, num_outliers))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, no_cam=args.no_cam)
        # val_dataset, val_loader = get_val_data(args.dataset, args.data_dir, args.height,
        #                                 args.width, args.batch_size, args.workers)
        val_loader = get_val_loader(dataset, args.height, args.width,
                                        args.batch_size, args.workers, testset=None)
        train_loader.new_epoch()

        trainer.train(epoch, train_loader, val_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader),val_iters=len(val_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('-mb', '--memorybank', type=str, default='CM', choices=['CM', 'CMhard', 'CMhybrid', 'CMhybrid_v2', 'CMhybrid_v3','CMhybrid_v4'])
    parser.add_argument('--hard-weight', type=float, default=0.5, help="hard weights")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")

    main()
