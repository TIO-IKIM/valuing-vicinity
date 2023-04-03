
import logging

from pathlib import Path
import torch

from src.deephist.segmentation.semantic_segmentation.semantic_inference import do_inference
from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import reload_model
from src.exp_management.tracking import Visualizer


def evaluate_details(patch_coordinates,
                     include_radius,
                     exp,
                     model,
                     wsis):
    
        viz = Visualizer(save_to_folder=True)

        for wsi_name, patch_coordinates in patch_coordinates.items():
            # select WSI from wsis
            try:
                selected_wsi = [wsi for wsi in wsis if wsi.name == wsi_name][0]
            except Exception as e:
                logging.getLogger('exp').error(f"Warning: Cannot find WSI {wsi_name}. Contueing")
                continue
            
            log_path = Path(exp.log_path) / 'inference' / wsi_name
            log_path.mkdir(exist_ok=True, parents=True)
            
            # build memory on that WSI
            with selected_wsi.inference_mode(): # sets wsi to idx 0 for memory
                for x, y in patch_coordinates:
                    try:
                        patch = selected_wsi.get_patch_from_position(x,y)
                        if patch is None:
                            logging.getLogger('exp').info(f"Patch {x}, {y} does not exist.")
                            continue
                        
                        context_patches, _  = patch.get_neighbours(k=include_radius)
                        
                        patches = [p for p in list(context_patches.flatten()) if p is not None]
                        
                        patches_loader = exp.data_provider.get_wsi_loader(patches=patches)

                        outputs, labels = do_inference(data_loader=patches_loader,
                                                       model=model,
                                                       gpu=exp.args.gpu) 
                        outputs, labels = torch.cat(outputs), torch.cat(labels)

                        # append results to patch object
                        for i, patch in enumerate(patches):
                            patch.prediction = exp.mask_to_img(mask=outputs[i],
                                                               label_handler=exp.data_provider.label_handler,
                                                               org_size=True) 
                            patch.mask = exp.mask_to_img(mask=labels[i],
                                                         label_handler=exp.data_provider.label_handler,
                                                         org_size=True) 
                            
                        viz.plot_wsi_section(section=context_patches,
                                        mode='org',
                                        log_path=log_path)
                        # gt
                        viz.plot_wsi_section(section=context_patches,
                                            mode='gt',
                                            log_path=log_path)
                        # pred
                        viz.plot_wsi_section(section=context_patches,
                                            mode='pred',
                                            log_path=log_path)
                        
                    except Exception as e:
                        logging.getLogger('exp').error(f"Could not visualize patch {x}, {y} of WSI {wsi_name}")
                        raise e
  
if __name__ == "__main__":

    # patch_coordinates = {'RCC-TA-033.001~C': [(14,19), (15,20), (20,20)],
    #                      'RCC-TA-011.001~J': [(20, 15), (20, 17)],
    #                      'RCC-TA-004.001~C': [(21, 35)]
    #                      }
    
    patch_coordinates = {'RCC-TA-163.001~B': [(7,12), (8,12), (8,11), (7,11)],
                         }
    exp_baseline = SegmentationExperiment(config_path='/src/deephist/evaluate/configs/baseline_segmentation_config_inference.yml')

    model_baseline = exp_baseline.model
    reload_baseline_from = Path(exp_baseline.log_path) / exp_baseline.args.reload_model_folder
    reload_model(model=model_baseline,
                 model_path=reload_baseline_from,
                 gpu=exp_baseline.args.gpu)
    
    evaluate_details(patch_coordinates=patch_coordinates,
                     include_k = 8,
                     exp=exp_baseline, 
                     model=model_baseline, 
                     wsis=exp_baseline.data_provider.test_wsi_dataset.wsis)

