
import torch

from src.exp_management import tracking
from src.exp_management.evaluation.dice import dice_coef, dice_denominator, dice_nominator


def initialize_logging(metric_logger, phase, args, num_heads=None):
    
    metric_logger.add_meter(f'{phase}_loss',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg'))
    metric_logger.add_meter(f'{phase}_pixel_accuracy',
                            tracking.SmoothedValue(window_size=1,
                                                    type='global_avg'))

    metric_logger.add_meter(f'{phase}_dice_coef',
                            tracking.SmoothedValue(window_size=1,
                                                    type='global_avg'))
    
    metric_logger.add_meter(f'{phase}_step_dice',
                            tracking.SmoothedValue(window_size=1,
                                                    type='global_avg',
                                                    to_tensorboard=False))
    # add attention logger per head
    if args.attention_on:
        for i in range(num_heads):
            metric_logger.add_meter(f'{phase}_ex_con_central_attention/head_{i}',
                                tracking.SmoothedValue(window_size=1,
                                                        type='global_avg'))
            metric_logger.add_meter(f'{phase}_coeff_var_neighbour_attention/head_{i}',
                                tracking.SmoothedValue(window_size=1,
                                                        type='global_avg'))
    
    if args.helper_loss:
        metric_logger.add_meter(f'{phase}_cls_loss',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg'))
        metric_logger.add_meter(f'{phase}_mask_loss',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg'))


def log_step(phase,
             metric_logger,
             loss,
             logits_gpu,
             labels_gpu, 
             images,
             args,
             attention_gpu = None,
             neighbour_masks = None,
             mask_loss = None,
             helper_loss = None
             ):
    
    if args.log_details:  
        labels_gpu.detach()
        logits_gpu.detach()
        
        # statistics over attention scores:
        if attention_gpu is not None and not args.use_transformer:   
            attention = attention_gpu.detach()             
            # ensure central patch is not considered:
            k = (neighbour_masks.shape[-1]-1)//2
            neighbour_masks[:, k, k] = 0
            neighbour_masks = neighbour_masks.view(-1,1,1,(k*2+1)*(k*2+1)).cuda(args.gpu, non_blocking=True) # over all heads, neighbour to 1d
            
            number_of_attentions = torch.sum(neighbour_masks, dim=-1)
            if args.use_self_attention:
                # we have a centre patch which gets one attention score. 
                # now, we want to determine if this attention score differs from the "expected" attention part (1/#attention_objects)
                ratio_neighbour_patches = (number_of_attentions/(number_of_attentions+1))
            else:
                # we dont have a central patch attention score
                ratio_neighbour_patches = 1
            excess_contribution_central_attention = torch.mean(-((torch.sum(attention * neighbour_masks,-1)-(ratio_neighbour_patches)))/ratio_neighbour_patches,dim=0)
            
            mean_of_attentions_per_head = (torch.sum(attention * neighbour_masks,-1)/number_of_attentions)
            att_deviation_from_mean = attention - mean_of_attentions_per_head.unsqueeze(3) 
            var_attention_per_head_and_neighbourhood = torch.sum((att_deviation_from_mean * att_deviation_from_mean) * neighbour_masks, dim=-1) / \
                torch.sum(neighbour_masks, dim=-1)
            # coefficient of variance: sd/mean -> mean of attention differs per #neighbour patches 
            coeff_var_attention = torch.mean(torch.sqrt(var_attention_per_head_and_neighbourhood) / mean_of_attentions_per_head, dim=0)
            
            excess_contribution_central_attention = excess_contribution_central_attention.cpu().numpy()
            coeff_var_neighbour_attention = coeff_var_attention.cpu().numpy()
        else:
            excess_contribution_central_attention = None
            coeff_var_neighbour_attention = None
            
        batch_accuracy = torch.sum(logits_gpu.argmax(axis=1) == labels_gpu)/torch.numel(labels_gpu)

    else:    
        # step_dice = 0
        # step_dice_nominator = 0
        # step_dice_denominator = 0
        batch_accuracy = 0
        excess_contribution_central_attention = None
        coeff_var_neighbour_attention = None
    
    step_dice_nominator = dice_nominator(y_true=labels_gpu,
                                         y_pred=torch.argmax(logits_gpu, dim=1),
                                         n_classes=args.number_of_classes)
    step_dice_denominator = dice_denominator(y_true=labels_gpu,
                                             y_pred=torch.argmax(logits_gpu, dim=1),
                                             n_classes=args.number_of_classes)
    
    step_dice, _ = dice_coef(dice_nominator=step_dice_nominator,
                             dice_denominator=step_dice_denominator,
                             n_classes=args.number_of_classes)
    
    if phase == 'train':
        metric_logger.update(train_pixel_accuracy=(batch_accuracy, len(images)),
                             train_loss=(loss.item(), len(images)),
                             train_step_dice=step_dice)
        if excess_contribution_central_attention is not None:
            # update attention logger per head
            for i in range(args.num_attention_heads):
                k = f'train_ex_con_central_attention_slash_head_{i}'
                metric_logger.meters[k].update(excess_contribution_central_attention[i], len(images))
                k = f'train_coeff_var_neighbour_attention_slash_head_{i}'
                metric_logger.meters[k].update(coeff_var_neighbour_attention[i], len(images))
        if mask_loss is not None:
            metric_logger.update(train_mask_loss=(mask_loss, len(images)))
        if helper_loss is not None:
            metric_logger.update(train_helper_loss=(helper_loss, len(images)))
    else:
        metric_logger.update(vali_pixel_accuracy=(batch_accuracy, len(images)),
                             vali_loss=(loss.item(), len(images)),
                             vali_step_dice=step_dice)
        if excess_contribution_central_attention is not None:
            # update attention logger per head
            for i in range(args.num_attention_heads):
                k = f'vali_ex_con_central_attention_slash_head_{i}'
                metric_logger.meters[k].update(excess_contribution_central_attention[i], len(images))
                k = f'vali_coeff_var_neighbour_attention_slash_head_{i}'
                metric_logger.meters[k].update(coeff_var_neighbour_attention[i], len(images))
        if mask_loss is not None:
            metric_logger.update(vali_mask_loss=(mask_loss, len(images)))
        if helper_loss is not None:
            metric_logger.update(vali_helper_loss=(helper_loss, len(images)))
                
    return step_dice_nominator, step_dice_denominator
    
def log_epoch(phase,
              metric_logger, 
              viz, 
              epoch_dice_nominator,
              epoch_dice_denominator,
              model,
              sample_images,
              sample_labels,
              sample_preds,
              label_handler,
              epoch,
              args): 
    epoch_dice, _ = dice_coef(dice_nominator=epoch_dice_nominator,
                              dice_denominator=epoch_dice_denominator,
                              n_classes=args.number_of_classes)

    if phase == 'train':
        metric_logger.update(train_dice_coef=epoch_dice)
    else:
        metric_logger.update(vali_dice_coef=epoch_dice)
    
    metric_logger.send_meters_to_tensorboard(step=epoch)

    if args.log_details:
        if phase == 'train':
            viz.plot_position_embeddings(tag=f'pos_embeddings',
                                            model=model,
                                            epoch=epoch)
            
        viz.plot_samples(tag=f'samples/{phase}_patch_samples',
                        images=sample_images,
                        col_size=8,
                        row_size=4,                       
                        epoch=epoch)
        
        viz.plot_masks(tag=f'samples/{phase}_mask_samples',
                    masks=sample_labels,
                    label_handler=label_handler,
                    col_size=8,
                    row_size=4,                       
                    epoch=epoch)
        
        viz.plot_masks(tag=f'samples/{phase}_pred_samples',
                    masks=sample_preds,
                    label_handler=label_handler,
                    col_size=8,
                    row_size=4,   
                    epoch=epoch)
