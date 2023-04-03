 
from src.deephist.evaluate.attention_analysis import run_attention_analysis
from src.deephist.evaluate.patch_inference import run_patch_inference
from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment

if __name__ == "__main__":
    
    patch_coordinates = {'tumor088': [(110,70),
                                      (100,70),
                                      (120,70),
                                      (100,60),
                                      (110,60),
                                      (120,60),
                                      (100,80),
                                      (110,80),
                                      (120,80)],
                         }
    
    exp = SegmentationExperiment(config_path='configs_inference/cy16_multiscale_segmentation_config_inference.yml')
    
    run_attention_analysis(exp)
    run_patch_inference(exp, patch_coordinates, radius=30)