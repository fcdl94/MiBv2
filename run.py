import utils
import argparser
import os
from utils.logger import Logger
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import get_dataset
from metrics import StreamSegMetrics
from segmentation_module import make_model

from train import Trainer
import tasks


def load_ckpt(opts, checkpoint_path, model, optimizer, scheduler, trainer, logger):

    if opts.continue_ckpt:
        opts.ckpt = checkpoint_path
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        logger.info("[!] Start from epoch 0")
        cur_epoch = 0
        best_score = 0.
    return cur_epoch, best_score


def get_optimizer(opts, model, loader):
    params = []
    if not opts.freeze:
        params.append({"params": filter(lambda p: p.requires_grad, model.body.parameters()),
                       'weight_decay': opts.weight_decay})

    params.append({"params": filter(lambda p: p.requires_grad, model.head.parameters()),
                   'weight_decay': opts.weight_decay})

    params.append({"params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                   'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * len(loader), power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict()
    }
    torch.save(state, path)


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Initialize logging
    task_name = f"{opts.dataset}-{opts.task}"
    if opts.overlap and opts.dataset == 'voc':
        task_name += "-ov"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    ckpt_path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step}.pth"
    if not os.path.exists(f"checkpoints/step/{task_name}"):
        os.makedirs(f"checkpoints/step/{task_name}")

    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    train_dst, val_dst, test_dst, labels, n_classes = get_dataset(opts)
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")

    # xxx Set up model
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    logger.debug(model)
    # logger.info(f"Backbone: {opts.backbone}")
    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))

    optimizer, scheduler = get_optimizer(opts, model, train_loader)

    if model_old is not None:
        model_old = DistributedDataParallel(model_old.to(device), device_ids=[opts.local_rank],
                                            output_device=opts.local_rank, find_unused_parameters=False)

    model = DistributedDataParallel(model.to(device), device_ids=[opts.local_rank],
                                    output_device=opts.local_rank, find_unused_parameters=False)

    # xxx Load old model from old weights if step > 0!

    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}.pth"

        # generate model from path
        if os.path.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False because of incr. classifiers
            if opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                model.module.init_new_classifier(device)
            # Load state dict from the model state dict, that contains the old model parameters
            model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters
            logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif opts.debug:
            logger.info(f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?")
        else:
            raise FileNotFoundError(path)

        # freeze old model and set eval mode
        for par in model_old.parameters():
            par.requires_grad = False
        model_old.eval()

    # xxx Set up Trainer
    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(model, model_old, device=device, opts=opts,
                      classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))

    # Load training checkpoint if any
    cur_epoch, best_score = load_ckpt(opts, ckpt_path, model, optimizer, scheduler, trainer, logger)

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    TRAIN = not opts.test
    val_metrics = StreamSegMetrics(n_classes)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here
    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        model.train()

        epoch_loss = trainer.train(cur_epoch=cur_epoch, optim=optimizer,
                                   train_loader=train_loader, scheduler=scheduler, logger=logger, rank=rank)

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0] + epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss", epoch_loss[0] + epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            model.eval()
            val_score, ret_samples = trainer.validate(loader=val_loader, metrics=val_metrics,
                                                      ret_samples_ids=sample_ids, logger=logger)

            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}")

            logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            if rank == 0:  # save best model at the last iteration
                score = val_score['Mean IoU']
                # best model to build incremental steps
                save_ckpt(ckpt_path,
                          model, trainer, optimizer, scheduler, cur_epoch, score)
                logger.info("[!] Checkpoint saved.")

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

            for k, (img, target, lbl) in enumerate(ret_samples):
                img = (denorm(img) * 255).astype(np.uint8)
                target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(ckpt_path,
                  model, trainer, optimizer, scheduler, cur_epoch, best_score)
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    # load best model
    if TRAIN:
        model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
        # Put the model on GPU
        model = DistributedDataParallel(model.cuda(device))
        ckpt = ckpt_path
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"*** Model restored from {ckpt}")
        del checkpoint
        trainer = Trainer(model, None, device=device, opts=opts)

    model.eval()

    val_score, _ = trainer.validate(loader=test_loader, metrics=val_metrics, logger=logger)
    logger.info(f"*** End of Test")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)

    logger.log_results(task=task_name, name=opts.name, results=val_score['Class IoU'].values())
    logger.log_aggregates(task=task_name, name=opts.name, results=val_score['Agg'])
    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs("checkpoints/step", exist_ok=True)

    main(opts)
