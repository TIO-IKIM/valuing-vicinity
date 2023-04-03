"""
Supervised Config parser
"""

from typing import List
import configargparse

from src.exp_management.config import Config

class AttentionSegmentationConfig(Config):
    """
    Defines the parameters for a supervised config
    """
    def __init__(self, config_paths: List[str]) -> None:
        super().__init__(config_paths = config_paths)

    def parse_config(self,
                     verbose: bool = False,
                     testmode: bool = False
                     ) -> configargparse.Namespace:

        parser = configargparse.ArgumentParser(description='PyTorch Training',
                                               default_config_files=self.config_paths)
        parser.add_argument('--train-data', default=None,
                            help='path to train dataset')
        parser.add_argument('--test-data', default=None,
                            help='path to test dataset. Optional.')
        parser.add_argument('--vali-split', default=None, type=float, 
                            help='ratio to split valiset from trainset')
        parser.add_argument('--arch', default='unet',
                            choices=['unet', 'unetplusplus', 'deeplabv3',
                                     'manet', 'linknet', 'fpn', 'pspnet',
                                     'deeplabv3plus', 'pan'],
                            help='segmentation model architecture')
        parser.add_argument('--encoder', default='resnet50',
                            choices= ['resnet18', 'resnet50', 'resnet101'])
        parser.add_argument('--workers', default=32, type=int,
                            help='number of data loading workers (default: 32)')
        parser.add_argument('--epochs', default=100, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--batch-size', default=512, type=int,
                            help='mini-batch size (default: 512), this is the total '
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--val-batch-size', default=512, type=int,
                            help='mini-batch size for validation run')
        parser.add_argument('--test-batch-size', default=None, type=int,
                            help='mini-batch size for test run. If None, uses validation batch size.')
        parser.add_argument('--learning-rate', default=0.001, type=float,
                            help='initial (base) learning rate', dest='lr')
        parser.add_argument('--lr-gamma', default=0.95, type=float, help='gamma of exp learnign rate decay')
        parser.add_argument('--adjust-lr', action='store_true', default=False,
                            help='select to adjust learning rate.')
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='momentum of SGD solver')
        parser.add_argument('--print-freq', default=20, type=int,
                            help='print frequency (default: 20)')
        parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--include-val-inference', action='store_true', default=False,
                            help='select to apply inference on val data.')
        parser.add_argument('--gpu', default=0, type=int,
                            help='GPU to use.')
        parser.add_argument('--gpus', default=None, nargs="*", type=int,
                            help='GPUs to use for cross-validation.')
        parser.add_argument('--patch-sampling', default=None,
                            choices=['over', 'under', 'overunder'],
                            help='Choose wether train data should be over- or undersampled.'
                            'overunder find as balance factor inbetween.')
        parser.add_argument('--logdir', default='logdir', type=str, help='path to store logs.')
        parser.add_argument('--label-map-file', default=None, type=str,
                            help='path to label mapping json')
        parser.add_argument('--patch-label-type', default='mask', type=str, choices= ['patch', 'image', 'mask'],
                            help='Select which label you want to plot for each patch representation')
        parser.add_argument('--pretrained', action='store_true',
                            help='select to start training with Imagenet-pretrained weights')
        parser.add_argument('--image-label-in-path', action='store_true',
                            help='Select if wsi are stored in image label folder')
        parser.add_argument('--exclude-classes', default=[], nargs="*", type=str,
                            help='Select class indices of patch class to be excluded')
        parser.add_argument('--include-classes', default=['0', '1', '2', '3', '7', '8', '9', '10', '13', '15', '16', '17'], nargs="*", type=str,
                            help='Select class indices of patch class to be included')
        parser.add_argument('--nfold', default=None, type=int,
                            help='Select the number of cv folds.')
        parser.add_argument('--folds', default=None, nargs="*", type=int,
                            help='Select folds to run specifically. If None, runs all folds. Default None')
        parser.add_argument('--draw-patches-per-class', default=None, type=int,
                            help='Select a number how many patches shell be drawn from each WSI per class.'
                                'If number exceeds existing patches per class in one wsi,'
                                'all are drawn from this class')
        parser.add_argument('--draw-patches-per-wsi', default=None, type=int,
                            help='Select a number how many patches shell be drawn from each WSI.')
        parser.add_argument('--hard-example-mining-ratio', type=float, default=None,
                            help='Select ratio to apply hard example mining on every batch.'
                            'Top ratio-percentage hardest examples are selected')
        parser.add_argument('--hard-example-mining-switch', action='store_true',
                            help='select to switch hem activation on every batch.')
        parser.add_argument('--hard-example-mining-per-class', action='store_true',
                            help='select to apply hem class-wise.')
        parser.add_argument('--hue', type=float, default=0.05,
                            help='Select hue for pytorch jitter-transform in trainset.')
        parser.add_argument('--brightness', type=float, default=0.05,
                            help='Select brightness for pytorch jitter-transform in trainset.')
        parser.add_argument('--saturation', type=float, default=0.05,
                            help='Select saturation for pytorch jitter-transform in trainset.')
        parser.add_argument('--contrast', type=float, default=0.05,
                            help='Select contrast for pytorch jitter-transform in trainset.')
        parser.add_argument('--augment', action='store_true',
                            help='select to apply augmentations.')
        parser.add_argument('--criterion', default='cross_entropy', type=str,
                            choices= ['cross_entropy', 'focal_tversky', 'dice', 'focal',
                                      'focal+dice', 'focal+focal_tversky', 'ce+dice'],
                            help='Select a loss function. If None, uses CrossEntropy')
        parser.add_argument('--use-ce-weights', action='store_true',
                            help='Set to use ce weights.')
        parser.add_argument('--combine-weight', type=float, default=0.5,
                            help='Loss combination weight. Weight for base loss.')
        parser.add_argument('--combine-criterion-after-epoch', default=None, type=int,
                            help='Select epoch after which to start combining a combined criterion.')
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Loss alpha param')
        parser.add_argument('--beta', type=float, default=0.5,
                            help='Loss beta param')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Loss gamma param')
        parser.add_argument('--weight-decay', type=float, default=0,
                            help='Select weight decay for l2 regularization')
        parser.add_argument('--normalize', action='store_true',
                            help='set to apply normalization in pytorch transformation')
        parser.add_argument('--freeze', action='store_true',
                            help='select to freeze all weights but the fc ones')
        parser.add_argument('--reload-model-folder', default=None, type=str,
                            help='Provide model log folder to reload a trained model. Training step is ignored then.')
        parser.add_argument('--evaluate-every', default=1, type=int,
                            help='Select epoch steps after which the test and val data is evaluated.'
                            'Select 0 to have a final evaluation on the best model only')
        parser.add_argument('--warm-up-epochs', default=None, type=int,
                            help='Select number of epochs without early stopping.')
        parser.add_argument('--early-stopping-epochs', default=None, type=int,
                            help='Select number of epochs without val loss improving to stop early.')
        parser.add_argument('--merge-classes', default=[], type=int, nargs='+', action='append',
                            help='Merge classes together. First class is kept. Per merge, provde a list of classes')
        parser.add_argument('--overlay-polygons', action='store_true',
                            help='Set to plot ground truth polygons. '
                            'If set, prehist-config must be given in data root folder. Optional.')
        parser.add_argument('--n-eval-wsis', default=None, type=int,
                            help='Select number of wsis for in-training evaluation.')
        parser.add_argument('--attention-on', action='store_true',
                            help='Set to activate attention memory mechanism. Else, "normal" segmentation is performed.')
        parser.add_argument('--embedding-dim', default=1024, type=int,
                            help='Select number of embedding dim. If None, memory is deactivated.')
        parser.add_argument('--k-neighbours', default=None, type=int,
                            help='Select number of neighbouring patches to attend to.')
        parser.add_argument('--context-conv', default=1, type=int,
                            help='Select kernel size of context convolution. Default to 1x1.')
        parser.add_argument('--num-attention-heads', default=8, type=int,
                            help='Select number of attention heads for MSA.')
        parser.add_argument('--attention-hidden-dim', default=1024, type=int,
                            help='Select number of MSA hidden dimension (after linear proj.).')
        parser.add_argument('--mlp-hidden-dim', default=2048, type=int,
                            help='Select number of transformer block mlp hidden dimension .')
        parser.add_argument('--transformer-depth', default=4, type=int,
                            help='Select transformer depth (layers).')
        parser.add_argument('--emb-dropout', type=float, default=0,
                            help='Select transformer/msa embedding dropout')
        parser.add_argument('--att-dropout', type=float, default=0,
                            help='Select MHA/transformer dropout (mlp & att)')
        parser.add_argument('--use-ln', action='store_true',
                            help='Set to use layer normalization at MSA beginning.')
        parser.add_argument('--sin-pos-encoding', action='store_true',
                            help='Set to use sinusoidal position encoding in MSA.')
        parser.add_argument('--learn-pos-encoding', action='store_true',
                            help='Set to activate learnable 2d position encoding.')
        parser.add_argument('--use-self-attention', action='store_true',
                            help='Set to consider central patch with respect to neighbour patches in MSA.')
        parser.add_argument('--use-transformer', action='store_true',
                            help='Set to use transformer encoder instead of MSA.')
        parser.add_argument('--online', action='store_true',
                            help='Set to create neighbour embs in real time (instead of memory).')
        parser.add_argument('--log-details', action='store_true',
                            help='Set to log computation-intense metrics.')
        parser.add_argument('--performance-metric', default='loss', type=str,
                             choices= ['loss', 'dice'],
                            help='Select a performance metric for early stopping and model selection.')
        parser.add_argument('--multiscale-on', action='store_true',
                            help='Set to run with multiscale model from Schmitz et al.')
        parser.add_argument('--sample-size', default=None, type=int,
                            help='Select a WSI sample size (e.g. for debugging).')
        parser.add_argument('--fill-in-eval', action='store_true', default=False,
                            help='Set to always fill memory in evaluation mode.')
        parser.add_argument('--wsi-batch', action='store_true', default=False,
                            help='Set to enforece wsi-oriented batching.')
        parser.add_argument('--helper-loss', action='store_true', default=False,
                            help='Set to enable class distribution helper loss.')
        parser.add_argument('--memory-to-cpu', action='store_true', default=False,
                            help='Set to move memory to cpu RAM.')
        
        parser.add_argument('--conf_file', default=None, type=str,
                            help='Dummy var for conf file')

        args, unknown = parser.parse_known_args()

        if not testmode and len(unknown) > 0:
            raise Exception(f"Unknonw args {unknown}")
        if verbose:
            print(args)
        return args