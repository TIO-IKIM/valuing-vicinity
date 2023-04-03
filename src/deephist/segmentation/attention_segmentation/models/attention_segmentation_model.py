from contextlib import contextmanager
import logging

from segmentation_models_pytorch import create_model
from segmentation_models_pytorch.base.modules import Conv2dReLU
import torch
from torch import nn
from tqdm import tqdm

from src.deephist.segmentation.attention_segmentation.models.memory import Memory
from src.deephist.segmentation.attention_segmentation.models.multihead_attention_model import \
    MultiheadAttention
from src.deephist.segmentation.attention_segmentation.models.transformer_model import ViT


class AttentionSegmentationModel(torch.nn.Module):
    
    def __init__(self, 
                 arch: str,
                 encoder_name: str,
                 encoder_weights: str,
                 number_of_classes: int,
                 attention_input_dim: int,
                 k: int,
                 attention_hidden_dim: int = 1024,
                 mlp_hidden_dim: int = 2048,
                 num_attention_heads: int = 8,
                 transformer_depth: int = 8,
                 emb_dropout: float = 0.,
                 att_dropout: float = 0.,
                 use_ln: bool = False,
                 use_central_attention: bool = False,
                 sin_pos_encoding: bool = False,
                 learn_pos_encoding: bool = False,
                 context_conv: int = 1,
                 attention_on: bool = True,
                 use_transformer: bool = False,
                 use_helperloss: bool = False,
                 fill_in_eval: bool = True,
                 online: bool = False
                 ) -> None:
        """_summary_

        Args:
            arch (str): Select Pytorch Segmentation Models architeture (e.g. unet, deeplabv3)
            encoder_name (str): Select Pytorch Segmentation Models encoder (e.g. resnet50)
            encoder_weights (str): Select Pytorch Segmentation Models weights (e.g. pretrained)
            number_of_classes (int): Provide number of segmentation classes
            attention_input_dim (int): Token dimension of memory embedding - equivalent to MHA/Transformer input dimension. 
            k (int): Neighbourhood size (radius).
            attention_hidden_dim (int, optional): Internal dimension of MHA/Transformer after linear projection. Defaults to 1024.
            mlp_hidden_dim (int, optional): Internal dimension of Transformer within MLP-module. Defaults to 2048.
            num_attention_heads (int, optional): Number attention heads of MHA/Transformer. Defaults to 8.
            transformer_depth (int, optional): Depth of Transformer (num layers). Defaults to 8.
            emb_dropout (float, optional): Dropout for embeddings (MHA/Transformer). Defaults to 0..
            att_dropout (float, optional): Dropout for attention module (MHA/Transformer). Defaults to 0..
            use_ln (bool, optional): Use layer normalization (MHA only) for k, q, v. Defaults to False.
            use_central_attention (bool, optional): Self-attention to central embedding turned on. 
                If false, masking is applied to central patch. . Defaults to False.
            sin_pos_encoding (bool, optional): Use sinusiodal position encoding (MHA/Transformer). Defaults to False.
            learn_pos_encoding (bool, optional): If true, position encoding is learned. Applies only to MHA . Defaults to False.
            context_conv (int): Size of convolution kernel to fuse context embedding with bottleneck features maps.
            attention_on (bool, optional): Use MHA/Transformer. If false, base segmentation model is applied. Defaults to True.
            use_transformer (bool, optional): Switch to use Transformer instead of MHA. Defaults to False.
            use_helperloss (bool, optional): Activates a tissue distribution helper loss per patch. 
                Applied after MHA/Transformer as lin. layer on attended embedding. Defaults to False.
            fill_in_eval (bool, optional): Set to always fill memory in evalution mode. Defaults to True.
            online (bool, optional): Set to get neighbour embeddings on the fly (instead of using the memory). Defaults to False.
            
        """
        super().__init__()
        
        self.online = online
        self.attention_on = attention_on
        self.use_central_attention = use_central_attention
        self.attention_input_dim = attention_input_dim
        self.use_transformer = use_transformer
        self.use_helperloss = use_helperloss
        self.fill_in_eval = fill_in_eval
        
        if self.attention_on:
            self.block_memory = False # option to skip memory attention in forward-pass
            
            self.kernel_size = k*2+1
            
            if not self.use_central_attention and not self.use_transformer:
                # mask central patch
                mask_central = torch.full((self.kernel_size,self.kernel_size), fill_value=1)
                mask_central[k,k] = 0
                self.register_buffer('mask_central', mask_central, persistent=False)
                    
        # Pytorch Segmentation Models: Baseline
        self.base_model = create_model(arch=arch,
                                       encoder_name=encoder_name,
                                       encoder_weights=encoder_weights,
                                       classes=number_of_classes)
        
        if self.attention_on:
            # f_emb
            self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
            # number of feature maps of encoder output: e.g. 2048 for U-net 5 layers 
            self.lin_proj = nn.Linear(in_features=self.base_model.encoder._out_channels[-1], 
                                      out_features=attention_input_dim)
            # f_fuse
            self.context_conv = Conv2dReLU(in_channels=self.base_model.encoder._out_channels[-1]+attention_input_dim, 
                                           out_channels=self.base_model.encoder._out_channels[-1], 
                                           kernel_size=(context_conv, context_conv),
                                           padding=(context_conv-1)//2,
                                           use_batchnorm=True)
            if self.use_transformer:
                self.transformer = ViT(kernel_size=self.kernel_size,
                                       dim=attention_input_dim,
                                       depth=transformer_depth,
                                       heads=num_attention_heads,
                                       mlp_dim=mlp_hidden_dim,
                                       hidden_dim=attention_hidden_dim,
                                       emb_dropout=emb_dropout,
                                       att_dropout=att_dropout,
                                       sin_pos_encoding=sin_pos_encoding,
                                       )
            else: # use MHA
                self.msa = MultiheadAttention(input_dim=attention_input_dim, 
                                              hidden_dim=attention_hidden_dim,
                                              num_heads=num_attention_heads,
                                              kernel_size= self.kernel_size,
                                              use_ln=use_ln,
                                              sin_pos_encoding=sin_pos_encoding,
                                              learn_pos_encoding=learn_pos_encoding,
                                              emb_dropout=emb_dropout,
                                              att_dropout=att_dropout)
            if self.use_helperloss:
                self.classifier = nn.Sequential(
                                    nn.Dropout(p=0.25),
                                    nn.Linear(in_features=attention_input_dim, out_features=number_of_classes),
                                  )
            # flag used for wsibatch (runs always with model.training) -
            # to remember which memory to use (instead of deriving from model.training)
            self._use_eval_mem = None
           
    @property 
    def use_eval_mem(self):
        return self._use_eval_mem
    
    @use_eval_mem.setter
    def use_eval_mem(self, eval_mem: bool):
        self._use_eval_mem = eval_mem
        if hasattr(self, 'val_memory'):
            self.val_memory.use_eval_mem = eval_mem
        
        if hasattr(self, 'train_memory'):
            self.train_memory.use_eval_mem = eval_mem
            
    def initialize_memory(self,
                          gpu: int,
                          reset=True,
                          **memory_params):
        """Initializes (train/eval) Memory - use model.eval() to initialize validation/evaluation Memory.

        Args:
            is_eval (bool, optional): If true, creates eval Memory - else
            train Memory. Later on, controlled by model.eval(). Defaults to False.
        """
        if self.use_eval_mem is not None:
            is_training = not self.use_eval_mem 
        else:
            is_training = self.training
            
        if is_training:
            if not hasattr(self, 'train_memory') or reset:
                logging.getLogger('exp').info("Initializing train memory")
                train_memory = Memory(**memory_params, is_eval=False, gpu=gpu, eval_mem=self.use_eval_mem)
                super(AttentionSegmentationModel, self).add_module('train_memory', train_memory)
        else:
            if not hasattr(self, 'val_memory') or reset:
                logging.getLogger('exp').info("Initializing eval memory")
                val_memory = Memory(**memory_params, is_eval=True, gpu=gpu, eval_mem=self.use_eval_mem)
                super(AttentionSegmentationModel, self).add_module('val_memory', val_memory)

    def fill_memory(self, 
                    data_loader: torch.utils.data.dataloader.DataLoader,
                    gpu: int,
                    use_phase: bool = None,
                    debug: bool = False):
        """Fill the memory by providing a dataloader that iterates the patches. 
        Iterate must provide all n_p patches.
        
        Note: Fill memory must be done in eval-mode for best performance
        
        Args:
            data_loader (torch.utils.data.dataloader.DataLoader): DataLoader
        """
        if self.block_memory:
            raise Exception("Memory is blocked. If you really want to fill memory, set 'block_memory' to False")
        
        # select memory:
        if use_phase is None:
            # derive from model.training status
            memory = self.memory
        else:
            if use_phase == 'train':
                memory = self.train_memory
            elif use_phase == 'vali':
                memory = self.val_memory
            
        #reset memory first to ensure consistency
        memory._reset()
        
        logging.getLogger('exp').info("Filling memory..")
        
        # no matter what, enforce all patch mode to create complete memory
        with data_loader.dataset.all_patch_mode():
            with torch.no_grad():
                for batch in tqdm(data_loader):
                    
                    images = batch['img']
                    patches_idx = batch['patch_idx']
                    
                    images = images.cuda(gpu, non_blocking=True)
                    
                    with self.eval_mode(): # check if filling memory should be done in eval
                        embeddings = self(images,
                                          return_embeddings=True)
                    if debug:
                        raise Exception("Debug mode turned off")
                        # replace with forward-running patch idx
                        # e = (patches_idx[1]-self.memory.k + (patches_idx[2]-self.memory.k) * self.memory.n_x).unsqueeze(dim=-1).expand(embeddings.shape)
                        # embeddings = e.type(torch.float32).to(embeddings.device)
                        
                    memory.update_embeddings(patches_idx=patches_idx,
                                             embeddings=embeddings)
            # flag memory ready to use
            memory.set_ready(n_patches=data_loader.dataset.__len__())
    
    @contextmanager
    def eval_mode(self):
        """
        Provides a context of model in eval setting. Afterwards, the previous state is reset.
        """
        if self.fill_in_eval:
            current_mode = self.training
            self.training = False 
            yield           
            self.training = current_mode
        else:
            yield

    @property
    def memory(self):
        
        if self.use_eval_mem is not None:
            use_train_mem = not self.use_eval_mem 
        else:
            use_train_mem = self.training

        if use_train_mem:
            if not hasattr(self, 'train_memory'):
                raise Exception("""Train Memory is not initialized yet. Please use the initialize_memory 
                function of the AttentionSegmentationModel and specify the required dimensions""")
            return self.train_memory
        else:
            if not hasattr(self, 'val_memory'):
                raise Exception("""Validation Memory is not initialized yet. Please use the initialize_memory 
                function of the AttentionSegmentationModel and specify the required dimensions""")
            return self.val_memory
        
    def forward(self, 
                images: torch.Tensor, 
                neighbours_idx: torch.Tensor = None,
                neighbour_imgs: torch.Tensor = None,
                return_embeddings: bool = False):
        """ 
        Attention segmentation model:
        If return_embeddings is True, only images must be provided and the model returns the compressed
        patch representation after the encoder + pooling + lineanr projection.
        
        If return_emebeddings is False, neighbour_idx must be provided to point to the coordinates in the memory.
        
        If model is set to online, instead of neighbour_idx you have to provide neighbour_imgs, as the neighbourhood
        patch embeddings will be derived, simultaneously.
   
        Args:
            images (torch.Tensor, optional): B x C x h x w normalized image tensor.
            neighbours_idx (torch.Tensor, optional): Indixes of neihgbourhood memory. Must be provided as long as return_embeddings is False. Defaults to None.
            neighbour_imgs (torch.Tensor, optional): Must be provided if model is set to online. Defaults to None.
            return_embeddings (bool, optional): Set to True to receive the patch embeddings, only. Defaults to False.

        Returns:
            [torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: 
            Either embeddings tensor - or tuple of prediction masks, attention maps and binary neighbour masks tensors
        """
        # set default values
        attention = neighbour_masks = cls_logits = None
        
        # segmentation encoder
        features = self.base_model.encoder(images)
        
        if self.attention_on and not self.block_memory:
            # add context to deepest feature maps feat_l by attending neighbourhood embeddings
            encoder_map = features[-1]
            
            # f_emb:
            # pool feature maps + linear proj to patch emb
            pooled = self.pooling(encoder_map)
            embeddings = self.lin_proj(pooled.flatten(1))
            
            # sanity check
            assert not torch.any(torch.max(embeddings, dim=1)[0] == 0), "Embeddings must have values."
                                
            if return_embeddings:
                return embeddings
            else:
                tmp_batch_size, ch, x, y = images.shape
                # online: get neighbour embeddings (with grads) concurrently
                if self.online:
                    with torch.no_grad():
                        neighbour_features = self.base_model.encoder(neighbour_imgs.view(-1,ch,x,y))
                    encoder_map = neighbour_features[-1]
                    # f_emb:
                    # pool feature maps + linear proj to patch emb
                    pooled = self.pooling(encoder_map)
                    neighbour_embeddings = self.lin_proj(pooled.flatten(1))
                    neighbour_embeddings = neighbour_embeddings.view(tmp_batch_size,self.kernel_size*self.kernel_size,-1)
                else: 
                    # query memory for context information
                    neighbour_embeddings, neighbour_masks = self.memory.get_k_neighbour_embeddings(neighbours_idx=neighbours_idx)
                    # ensure query result is on gpu:
                    if not neighbour_embeddings.is_cuda:
                        neighbour_embeddings = neighbour_embeddings.to(embeddings.device)
                        neighbour_masks = neighbour_masks.to(embeddings.device)
                    
                
                embeddings = torch.unsqueeze(embeddings, 1)

                # sanity check: all embeddings should have values  
                if not self.online:                  
                    assert torch.sum(neighbour_masks).item() == torch.sum(torch.max(neighbour_embeddings, dim=-1)[0] != 0).item(), \
                        'all embeddings should have values'
                        
                if not self.use_central_attention and not self.use_transformer:  
                    # add empty central patches - happens when self-attention is turned off
                    neighbour_masks = neighbour_masks * self.mask_central

                # from "2d" to "1d"
                k_neighbour_masks = neighbour_masks.view(tmp_batch_size, 1, 1, -1)
                neighbour_embeddings = neighbour_embeddings.view(tmp_batch_size, -1, self.attention_input_dim)
            
                if self.use_transformer:
                    # replace central patch embedding with current embedding
                    c_pos = (self.kernel_size*self.kernel_size-1)//2
                    neighbour_embeddings[:,c_pos:(c_pos+1),:] = embeddings
                    attended_embeddings, attention = self.transformer(x=neighbour_embeddings,
                                                                      mask=k_neighbour_masks,
                                                                      return_attention=True) 
                else: #MHA
                    attended_embeddings, attention =  self.msa(q=embeddings,
                                                               kv=neighbour_embeddings,
                                                               mask=k_neighbour_masks,
                                                               return_attention=True)
                    
                # helper loss for classification - implement after attention to bring network to use memory attention
                # for classification task 
                if self.use_helperloss:
                    cls_logits = self.classifier(attended_embeddings.squeeze(dim=1))
                    
                # f_fuse:
                # concatinate attended embeddings to encoded features      
                attended_embeddings = torch.squeeze(attended_embeddings, 1)
                # expand over e.g 8x8-convoluted feature map for Unet - or 32x32 for deeplabv3
                attended_embeddings = attended_embeddings[:,:,None, None].expand(-1, -1, encoder_map.shape[-2], encoder_map.shape[-1])
                features_with_neighbour_context = torch.cat((features[-1], attended_embeddings),1)
                # 1x1 conv to merge features to 2048 again
                features_with_neighbour_context = self.context_conv(features_with_neighbour_context)
                
                # exchange feat_l with feat_l'
                features[-1] = features_with_neighbour_context
        
        # segmentation decoder    
        decoder_output = self.base_model.decoder(*features)
        # segmentation head
        logits = self.base_model.segmentation_head(decoder_output)
        
        return {'logits': logits,
                'cls_logits': cls_logits, 
                'attention': attention, 
                'neighbour_masks': neighbour_masks
            }
       
    