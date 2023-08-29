#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda import amp
from torch.cuda.amp import autocast

from models import load_model
from utils.logger import Logger
from utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from utils.datasets import ListDataset
from utils.augmentations import AUGMENTATION_TRANSFORMS
# from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from utils.parse_config import parse_data_config
from utils.loss import ComputeLoss
from test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS,
        is_train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="./config/Config.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="./config/sun.data",
                        help="Path to data config file (.data)")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument('--output', type=str, default='./netout', help='output')
    parser.add_argument('--description', type=str, default='train_phase_1',
                        help='description of model')  

    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")

    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1,
                        help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.4,
                        help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.15, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3,
                        help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument('--cuda_idx', type=str, default='0', help='cuda')

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_idx

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(f"{args.output}/{args.description}/{args.logdir}")  # Tensorboard logger

    # Create output directories if missing
    os.makedirs(f"{args.output}/{args.description}/output", exist_ok=True)
    os.makedirs(f"{args.output}/{args.description}/checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]

    # modified
    if "test" in data_config.keys():
        test_path = data_config["test"]

    class_names = load_classes(data_config["names"])
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_idx

    # ############
    # Create model
    # ############

    model = load_model(args.model, args.pretrained_weights, args)
    computeLoss = ComputeLoss(model)

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    # modified
    if "test" in data_config.keys():
        test_dataloader = _create_validation_data_loader(
            test_path,
            mini_batch_size,
            model.hyperparams['height'],
            args.n_cpu)

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    scaler = amp.GradScaler()

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    best_ap = -1.0
    max_iterations = args.epochs * len(dataloader)
    for epoch in range(1, args.epochs + 1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.cuda()
            targets = targets.cuda()

            with autocast():
                outputs = model(imgs)

                # loss, loss_components = compute_loss(outputs, targets, model)
                loss, loss_components = computeLoss(outputs, targets)

            # loss.backward()
            scaler.scale(loss).backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                # lr = lr * (1.0 - batches_done / max_iterations) ** 0.9  # 设置学习率调整模式
                lr = lr * 0.99 ** (300 * (batches_done / max_iterations))

                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                # iou_thres=args.iou_thres,
                iou_thres=0.3,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if "test" in data_config.keys():
                _, _, test_AP, _, _, _ = _evaluate(
                    model,
                    test_dataloader,
                    class_names,
                    img_size=model.hyperparams['height'],
                    iou_thres=0.3,
                    conf_thres=args.conf_thres,
                    nms_thres=args.nms_thres,
                    verbose=args.verbose
                )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class, mIOU = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    # ("validation/test_mAP", test_AP.mean()),
                    ("validation/f1", f1.mean()),
                    ("validation/mIOU", mIOU)]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            if AP.mean() > best_ap:
                best_ap = AP.mean()
                print()
                checkpoint_path = f"{args.output}/{args.description}/checkpoints/yolov3_ckpt_{epoch}_ap_{best_ap:.5f}.pth"
                best_path = f"{args.output}/{args.description}/checkpoints/{args.description}_best_model.pth"
                print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(model.state_dict(), best_path)


if __name__ == "__main__":
    run()
