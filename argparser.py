import argparse

import tasks


def modify_command_options(opts):
    opts.pooling = round(opts.crop_size / opts.output_stride)

    if opts.dataset == 'voc':
        opts.num_classes = 21
    if opts.dataset == 'ade':
        opts.num_classes = 150

    if not opts.visualize:
        opts.sample_num = 0

    if opts.method is not None:
        if opts.method == 'FT':
            pass
        if opts.method == 'LWF':
            opts.loss_kd = 100
        # if opts.method == 'LWF-MC':
        #     opts.icarl = True
        #     opts.icarl_importance = 10
        if opts.method == 'ILT':
            opts.loss_kd = 100
            opts.loss_de = 100
        if opts.method.upper() == 'MIB':
            opts.loss_kd = 10 if "15-1" not in opts.task else 100
            opts.unce = True
            opts.unkd = True
            opts.init_balanced = True
        if opts.method == 'PLOP':
            if opts.overlap:
                opts.threshold = 0.001
                opts.pod_factor = 0.0005
            else:
                opts.threshold = 0.5
                opts.pod_factor = 0.0001
            opts.pod = True
            opts.pseudo = True
            opts.init_balanced = True
        if opts.method == "IC":
            opts.icarl = True
            opts.icarl_bkg = -1

    opts.no_overlap = not opts.overlap

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help='number of workers (default: 1)')

    # Datset Options
    parser.add_argument("--data_root", type=str, default='data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'ade'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Method Options
    # BE CAREFUL USING THIS, THEY WILL OVERRIDE ALL THE OTHER PARAMETERS.
    parser.add_argument("--method", type=str, default=None,
                        choices=['FT', 'LWF', 'IC', 'ILT', 'EWC', 'RW', 'PI', 'MiB', 'PLOP'],
                        help="The method you want to use. BE CAREFUL USING THIS, IT MAY OVERRIDE OTHER PARAMETERS.")

    # Train Options
    parser.add_argument("--epochs", type=int, default=30,
                        help="epoch number (default: 30)")

    parser.add_argument("--batch_size", type=int, default=24,
                        help='batch size (default: 24)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 512)")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step'], help="lr schedule policy (default: poly)")
    parser.add_argument("--lr_decay_step", type=int, default=5000,
                        help="decay step for stepLR (default: 5000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    parser.add_argument("--bce", default=False, action='store_true',
                        help="Whether to use BCE or not (default: no)")

    # Validation Options
    parser.add_argument("--val_on_trainset", action='store_true', default=False,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--crop_val", action='store_false', default=True,
                        help='do crop for validation (default: True)')

    # Logging Options
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=0,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument("--debug",  action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--visualize",  action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")

    # Model Options
    parser.add_argument("--backbone", type=str, default='resnet101',
                        choices=['resnet50', 'resnet101'], help='backbone for the body (def: resnet50)')
    parser.add_argument("--deeplab", type=str, default="v3",
                        choices=['v3', 'v2', 'none'], help='network head')
    parser.add_argument("--output_stride", type=int, default=16,
                        choices=[8, 16], help='stride for the backbone (def: 16)')
    parser.add_argument("--no_pretrained", action='store_true', default=False,
                        help='Whether to use pretrained or not (def: True)')
    parser.add_argument("--norm_act", type=str, default="iabn_sync",
                        help='Which BN to use (def: iabn_sync')
    parser.add_argument("--pooling", type=int, default=32,
                        help='pooling in ASPP for the validation phase (def: 32)')
    parser.add_argument("--cosine", action='store_true', default=False,
                        help='Whether to use cosine classifier (def: False)')

    # Test and Checkpoint options
    parser.add_argument("--test",  action='store_true', default=False,
                        help="Whether to train or test only (def: train and test)")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--continue_ckpt", default=False, action='store_true',
                        help="Restart from the ckpt. Named taken automatically from method name.")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="epoch interval for saving model (default: 1)")

    # Parameters for Knowledge Distillation of ILTSS (https://arxiv.org/abs/1907.13372)
    parser.add_argument("--freeze", action='store_true', default=False,
                        help="Use this to freeze the feature extractor in incremental steps")
    parser.add_argument("--loss_de", type=float, default=0.,  # Distillation on Encoder
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable distillation on Encoder (L2)")
    parser.add_argument("--loss_kd", type=float, default=0.,  # Distillation on Output
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable Knowlesge Distillation (Soft-CrossEntropy)")

    # Arguments for ICaRL (from https://arxiv.org/abs/1611.07725)
    parser.add_argument("--icarl", default=False, action='store_true',
                        help="If enable ICaRL or not (def is not)")
    parser.add_argument("--icarl_bkg", action='store_true', default=False,
                        help="If use background from GT (def: No)")

    # MiB
    parser.add_argument("--init_balanced", default=False, action='store_true',
                        help="Enable Background-based initialization for new classes")
    parser.add_argument("--unkd", default=False, action='store_true',
                        help="Enable Unbiased Knowledge Distillation instead of Knowledge Distillation")
    parser.add_argument("--alpha", default=1., type=float,
                        help="The parameter to hard-ify the soft-labels. Def is 1.")
    parser.add_argument("--unce", default=False, action='store_true',
                        help="Enable Unbiased Cross Entropy instead of CrossEntropy")

    # # PLOP
    parser.add_argument("--pod", default=None, type=str,
                        help="Enable POD distillation")
    parser.add_argument("--pseudo", default=None, type=str,
                        help="Enable POD distillation")

    # Incremental parameters
    parser.add_argument("--task", type=str, default="19-1", choices=tasks.get_task_list(),
                        help="Task to be executed (default: 19-1)")
    parser.add_argument("--step", type=int, default=0,
                        help="The incremental step in execution (default: 0)")
    parser.add_argument("--no_mask", action='store_true', default=False,
                        help="Use this to not mask the old classes in new training set")
    parser.add_argument("--overlap", action='store_true', default=False,
                        help="Use this to not use the new classes in the old training set")
    parser.add_argument("--step_ckpt", default=None, type=str,
                        help="path to trained model at previous step. Leave it None if you want to use def path")

    return parser
