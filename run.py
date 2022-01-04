import os
import random

import numpy as np
import torch
from torch import distributed
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

import argparser
import utils
from dataset import get_dataset
from metrics import StreamSegMetrics
from train import Trainer
from utils.logger import WandBLogger


def save_ckpt(path, trainer, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "scheduler_state": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "best_score": best_score,
    }
    torch.save(state, path)


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)
    opts.device_id = device_id

    # Initialize logging
    task_name = f"{opts.dataset}-{opts.task}"
    if opts.overlap and opts.dataset == 'voc':
        task_name += "-ov"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    logger = WandBLogger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step,
                         name=f"{task_name}_{opts.name}")

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

    opts.batch_size = opts.batch_size // world_size  # MAKE sure it is evenly divisible
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1, shuffle=False,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")

    logger.info(f"Total batch size is {opts.batch_size * world_size}")
    opts.max_iters = opts.epochs * len(train_loader)

    # xxx Set up Trainer
    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(logger, device=device, opts=opts)

    # xxx Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}.pth"
        trainer.load_step_ckpt(path)

    # Load training checkpoint if any
    if opts.continue_ckpt:
        opts.ckpt = ckpt_path
    if opts.ckpt is not None:
        cur_epoch, best_score = trainer.load_ckpt(ckpt_path)
    else:
        logger.info("[!] Start from epoch 0")
        cur_epoch = 0
        best_score = 0.

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_config(opts)

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

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here
    if TRAIN:
        trainer.before(train_loader=train_loader, logger=logger)

    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        epoch_loss = trainer.train(cur_epoch=cur_epoch, train_loader=train_loader)

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0] + epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("Train/Tot", epoch_loss[0] + epoch_loss[1], cur_epoch)
        logger.add_scalar("Train/Reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("Train/Cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            val_score, ret_samples = trainer.validate(loader=val_loader, metrics=val_metrics,
                                                      ret_samples_ids=sample_ids)
            logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            if rank == 0:  # save best model at the last iteration
                score = val_score['Mean IoU']
                # best model to build incremental steps
                save_ckpt(ckpt_path, trainer, cur_epoch, score)
                logger.info("[!] Checkpoint saved.")

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("Val/Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val/MeanAcc", val_score['Agg'][1], cur_epoch)
            logger.add_scalar("Val/MeanPrec", val_score['Agg'][2], cur_epoch)
            logger.add_scalar("Val/MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val/Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val/Acc_IoU", val_score['Class Acc'], cur_epoch)
            logger.add_figure("Val/Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

            for k, (img, target, lbl) in enumerate(ret_samples):
                img = (denorm(img) * 255).astype(np.uint8)
                target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                logger.add_image(f'Sample/{k}', concat_img, cur_epoch)
            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}")

        logger.commit()
        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(ckpt_path, trainer, cur_epoch, best_score)
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size, shuffle=False,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    val_score, _ = trainer.validate(loader=test_loader, metrics=val_metrics)
    logger.info(f"*** End of Test")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test/Class_IoU", val_score['Class IoU'])
    logger.add_table("Test/Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test/Confusion_Matrix", val_score['Confusion Matrix'])

    logger.add_scalar("Test/Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("Test/MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("Test/MeanAcc", val_score['Mean Acc'], opts.step)
    logger.commit()

    logger.log_results(task=task_name, name=opts.name, results=val_score['Class IoU'].values())
    logger.log_aggregates(task=task_name, name=opts.name, results=val_score['Agg'])
    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs("checkpoints/step", exist_ok=True)

    main(opts)
