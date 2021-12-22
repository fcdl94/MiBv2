import os.path as osp
from functools import reduce

import torch
import torch.nn as nn
import tqdm
from torch import distributed
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel

import tasks
import utils
from segmentation_module import make_model
from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss
from utils.plop import find_median, entropy, features_distillation


class Trainer:
    def __init__(self, logger, device, opts):
        self.logger = logger
        self.device = device
        self.opts = opts
        self.scaler = amp.GradScaler()
        self.step = opts.step

        classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
        new_classes = classes[-1]
        tot_classes = reduce(lambda a, b: a + b, classes)
        self.old_classes = tot_classes - new_classes
        self.nb_classes = opts.num_classes
        self.nb_current_classes = tot_classes
        self.nb_new_classes = new_classes

        self.model = make_model(opts, classes=classes)

        if opts.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            self.model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))
            self.model_old.to(self.device)
            # freeze old model and set eval mode
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        self.optimizer, self.scheduler = self.get_optimizer(opts)

        self.distribute(opts)

        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
        else:
            self.old_classes = 0

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and self.model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and self.model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and self.model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and self.model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        self.ret_intermediate = self.lde or (opts.pod is not None)

        self.pseudo_labeling = None
        self.classif_adaptive_factor = False
        self.thresholds, self.max_entropy = None, None
        if opts.pseudo:
            self.pseudo_labeling = "entropy"
            self.threshold = 0.001
            self.classif_adaptive_factor = True
            self.classif_adaptive_min_factor = 0.0

        self.pod = None
        if opts.pod:
            self.pod = "local"
            self.pod_factor = 0.01
            self.pod_logits = True
            self.pod_apply = 'all'
            self.pod_deeplab_mask = False
            self.pod_deeplab_mask_factor = None
            self.pod_options = {"switch": {"after": {"extra_channels": "sum", "factor": 0.0005, "type": "local"}}}
            self.use_pod_schedule = True
            self.pod_interpolate_last = False
            self.pod_large_logits = False
            self.pod_prepro = 'pow'
            self.deeplab_mask_downscale = False
            self.spp_scales = [1, 2, 4]

        self.compute_model_old = (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or
                                  self.pod is not None or self.pseudo_labeling is not None)
        self.compute_model_old = self.compute_model_old and self.model_old is not None

    def get_optimizer(self, opts):
        params = []
        if not opts.freeze:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                           'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                       'weight_decay': opts.weight_decay})

        optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

        if opts.lr_policy == 'poly':
            scheduler = utils.PolyLR(optimizer, max_iters=opts.max_iters, power=opts.lr_power)
        elif opts.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step,
                                                        gamma=opts.lr_decay_factor)
        else:
            raise NotImplementedError

        return optimizer, scheduler

    def distribute(self, opts):
        self.model = DistributedDataParallel(self.model.to(self.device), device_ids=[opts.device_id],
                                             output_device=opts.device_id, find_unused_parameters=False)

    def before(self, train_loader, logger):
        if self.pseudo_labeling is None:
            return
        with amp.autocast():
            if self.pseudo_labeling.split("_")[0] == "median" and self.step > 0:
                logger.info("Find median score")
                self.thresholds, _ = find_median(train_loader, self.device, logger, self.model_old,
                                                 self.nb_current_classes, self.threshold)
            elif self.pseudo_labeling.split("_")[0] == "entropy" and self.step > 0:
                logger.info("Find median score")
                self.thresholds, self.max_entropy = find_median(train_loader, self.device, logger, self.model_old,
                                                                self.nb_current_classes, self.threshold, mode="entropy")

    def train(self, cur_epoch, train_loader, print_int=10):
        """Train and return epoch loss"""
        optim = self.optimizer
        scheduler = self.scheduler
        device = self.device
        model = self.model
        criterion = self.criterion
        logger = self.logger

        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        pod_loss = torch.tensor(0.)

        train_loader.sampler.set_epoch(cur_epoch)

        if distributed.get_rank() == 0:
            tq = tqdm.tqdm(total=len(train_loader))
            tq.set_description("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
        else:
            tq = None

        model.train()
        for cur_step, (images, labels) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            with amp.autocast():
                if self.compute_model_old:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, ret_intermediate=self.ret_intermediate)

                classif_adaptive_factor = 1.0
                if self.step > 0 and self.pseudo_labeling is not None:
                    mask_background = labels < self.old_classes

                    if self.pseudo_labeling == "naive":
                        labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]
                    elif self.pseudo_labeling is not None and self.pseudo_labeling.startswith("threshold_"):
                        threshold = float(self.pseudo_labeling.split("_")[1])
                        probs = torch.softmax(outputs_old, dim=1)
                        pseudo_labels = probs.argmax(dim=1)
                        pseudo_labels[probs.max(dim=1)[0] < threshold] = 255
                        labels[mask_background] = pseudo_labels[mask_background]
                    elif self.pseudo_labeling == "median":
                        probs = torch.softmax(outputs_old, dim=1)
                        max_probs, pseudo_labels = probs.max(dim=1)
                        pseudo_labels[max_probs < self.thresholds[pseudo_labels]] = 255
                        labels[mask_background] = pseudo_labels[mask_background]
                    elif self.pseudo_labeling == "entropy":
                        probs = torch.softmax(outputs_old, dim=1)
                        max_probs, pseudo_labels = probs.max(dim=1)

                        mask_valid_pseudo = (entropy(probs) /
                                             self.max_entropy) < self.thresholds[pseudo_labels]

                        # All old labels that are NOT confident enough to be used as pseudo labels:
                        labels[~mask_valid_pseudo & mask_background] = 255
                        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                    mask_background]
                        if self.classif_adaptive_factor:
                            # Number of old/bg pixels that are certain
                            num = (mask_valid_pseudo & mask_background).float().sum(dim=(1, 2))
                            # Number of old/bg pixels
                            den = mask_background.float().sum(dim=(1, 2))
                            # If all old/bg pixels are certain the factor is 1 (loss not changed)
                            # Else the factor is < 1, i.e. the loss is reduced to avoid
                            # giving too much importance to new pixels
                            classif_adaptive_factor = num / (den + 1e-6)
                            classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                            if self.classif_adaptive_min_factor:
                                classif_adaptive_factor = classif_adaptive_factor.clamp(
                                    min=self.classif_adaptive_min_factor)

                optim.zero_grad()
                outputs, features = model(images, ret_intermediate=self.ret_intermediate)

                # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    # ICaRL loss -- unique CE+KD
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                if self.classif_adaptive_factor:
                    loss = classif_adaptive_factor * loss

                loss = loss.mean()  # scalar

                # xxx ICARL DISTILLATION
                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                  torch.sigmoid(outputs_old))

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde * self.lde_loss(features['body'], features_old['body'])

                if self.lkd_flag:
                    # resize new output to remove new logits and keep only the old ones
                    lkd = self.lkd * self.lkd_loss(outputs, outputs_old)

                if self.pod is not None and self.step > 0:
                    attentions_old = features_old["attentions"]
                    attentions_new = features["attentions"]

                    if self.pod_logits:
                        attentions_old.append(features_old["sem_logits_small"])
                        attentions_new.append(features["sem_logits_small"])
                    elif self.pod_large_logits:
                        attentions_old.append(outputs_old)
                        attentions_new.append(outputs)

                    pod_loss = features_distillation(
                        attentions_old,
                        attentions_new,
                        collapse_channels=self.pod,
                        labels=labels,
                        index_new_class=self.old_classes,
                        pod_apply=self.pod_apply,
                        pod_deeplab_mask=self.pod_deeplab_mask,
                        pod_deeplab_mask_factor=self.pod_deeplab_mask_factor,
                        interpolate_last=self.pod_interpolate_last,
                        pod_factor=self.pod_factor,
                        prepro=self.pod_prepro,
                        deeplabmask_upscale=not self.deeplab_mask_downscale,
                        spp_scales=self.spp_scales,
                        pod_options=self.pod_options,
                        outputs_old=outputs_old,
                        use_pod_schedule=self.use_pod_schedule,
                        nb_current_classes=self.nb_current_classes,
                        nb_new_classes=self.nb_new_classes
                    )

                # xxx first backprop of previous loss (compute the gradients for regularization methods)
                loss_tot = loss + lkd + lde + l_icarl + pod_loss

            self.scaler.scale(loss_tot).backward()

            self.scaler.step(optim)
            if scheduler is not None:
                scheduler.step()
            self.scaler.update()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item() + pod_loss.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if tq is not None:
                tq.update(1)
                tq.set_postfix(loss='%.6f' % loss)

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.debug(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                             f" Loss={interval_loss}")
                logger.debug(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss/Tot', interval_loss, x, intermediate=True)
                    logger.commit(intermediate=True)
                interval_loss = 0.0

        if tq is not None:
            tq.close()

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return (epoch_loss, reg_loss)

    def validate(self, loader, metrics, ret_samples_ids=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device

        model.eval()

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                with amp.autocast():
                    outputs, features = model(images, ret_intermediate=False)
                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0],
                                        prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

        return score, ret_samples

    def load_step_ckpt(self, path):
        # generate model from path
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            if self.opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                self.model.module.init_new_classifier(self.device)
            # Load state dict from the model state dict, that contains the old model parameters
            new_state = {}
            for k, v in step_checkpoint['model_state'].items():
                new_state[k[7:]] = v
            self.model_old.load_state_dict(new_state, strict=True)  # Load also here old parameters
            self.logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif self.opts.debug:
            self.logger.info(f"[!] WARNING: Unable to find of step {self.opts.step - 1}! "
                             f"Do you really want to do from scratch?")
        else:
            raise FileNotFoundError(path)

    def load_ckpt(self, path):
        opts = self.opts
        assert osp.isfile(path), f"Error, ckpt not found in {path}"

        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        self.logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        del checkpoint

        return cur_epoch, best_score
