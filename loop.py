# call from terminal:
# python loop.py


from functools import partial
import logging
import logging.config
import logging.handlers
import queue
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml


def run_job_queue(config_folder: str,
                  gpu_file: str,
                  **kwargs):
    q = queue.Queue()
    
    d = {
        'version': 1,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s'
            },
             'printer': {
                'class': 'logging.Formatter',
                'format': '%(processName)-10s %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'printer',
            },
            'loop': {
                'class': 'logging.FileHandler',
                'filename': 'loop.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            'success': {
                'class': 'logging.FileHandler',
                'filename': 'success.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            'error': {
                'class': 'logging.FileHandler',
                'filename': 'error.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
        },
        'loggers': {
            'success_logger': {
                'handlers': ['success', 'console']
            },
             'error_logger': {
                'handlers': ['error', 'console']
            },
             'loop_logger': {
                'handlers': ['loop', 'console']
            }
        }
    }
    # overwrite log files
    Path("logs").mkdir(exist_ok=True)
    n = datetime.now().strftime("%Y-%m-%d_%H-%H-%S")
    d['handlers']['success']['filename'] = f'logs/log_{n}_success.log'
    d['handlers']['error']['filename'] = f'logs/log_{n}_error.log'
    d['handlers']['loop']['filename'] = f'logs/log_{n}_loop.log'
    
    logging.config.dictConfig(d)
    logger = logging.getLogger('loop_logger')
    logger.setLevel(logging.DEBUG)

    success_logger = logging.getLogger('success_logger')
    success_logger.setLevel(logging.DEBUG)
    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.DEBUG)
    loop_logger = logging.getLogger('loop_logger')
    loop_logger.setLevel(logging.DEBUG)
    
    config_queue = queue.Queue()
    gpu_queue  = queue.Queue()
    used_gpu_queue  = queue.Queue()

    # intialize config queue
    base_configs_files =initialize_config_queue(config_folder, config_queue, logger)
    
    def signal_handler(config_queue, signal, frame):
        print('You pressed Ctrl+C!')
        # put 6x None in config-queue to 
        for _ in range(6):
            config_queue.put(None)
            
    # start listening on stop signal
    signal.signal(signal.SIGINT, partial(signal_handler, config_queue))
    
    gt = threading.Thread(target=gpu_resource_thread, args=(gpu_queue, used_gpu_queue, gpu_file, logger))
    gt.start()
    ct = threading.Thread(target=config_sync_thread, args=(base_configs_files, config_queue, config_folder, logger))
    ct.start()
    
    task_threads = []
    # start subprocesses for each config file
    for _ in range(6): # at most 6 subprocesses
        t = threading.Thread(target=run_job, 
                             args=(config_queue, 
                                   gpu_queue, 
                                   used_gpu_queue, 
                                   kwargs)
                            ) 
        t.start()
        task_threads.append(t)                        
    
    # wait until all task threads are done
    for t in task_threads:
        t.join()
        
    print("Tasks done.")
    
    # finish log thread
    q.put(None)
    used_gpu_queue.put(None)
    gt.join()
    print("GPU resource handling done.")
    ct.join()
    print("Config handling done.")
  
def run_job(config_queue,
            gpu_queue,
            used_gpu_queue,
            kwargs):
         
    success_logger = logging.getLogger('success_logger')
    error_logger = logging.getLogger('error_logger')
    loop_logger = logging.getLogger('loop_logger')

    # run until all configs are processed
    while True:
        # first, pull a config from the queue, if one is still availble
        try:
            config_file = config_queue.get(False) 
            loop_logger.info(f"Taking {config_file} config from queue")
            loop_logger.info(f"{config_queue.qsize()} tasks on task queue left.")
        except queue.Empty:
            # config queue is emtpy, wait 10 secs
            time.sleep(10)
            continue
            
        # thread is killed by None token
        if config_file is None:
            print("Interrupting job thread")
            break
        
        # then, get/wait for free gpu resource from queue
        gpu = gpu_queue.get()
        loop_logger.info(f"Taking GPU {gpu} resource")
        loop_logger.info(f"Starting {config_file} on GPU {gpu}")

        #kwargs to argv
        argv = ""
        for key, val in kwargs.items():
            argv += f" --{key.replace('_','-')} "
            if isinstance(val, list):
                val = ",".join([str(v) for v in val])
            argv += str(val)
            
        # here call subproccess
        p = subprocess.run([f'bash -c "source activate vv; python main.py \
                                --conf_file {config_file} --gpu {str(gpu)}' + argv +'"'], 
                                stderr=subprocess.PIPE, 
                                shell=True)
        
        if p.returncode != 0: 
            error_logger.error(p.stderr)
            error_logger.error(f"Error: {config_file}")
            loop_logger.error(f"Error: {config_file}")
        else:
            success_logger.info(f"Successful: {config_file}")
            loop_logger.info(f"Successful: {config_file}")
        
        # free gpu resource again 
        loop_logger.info(f"Freeing GPU {gpu} resource")
        used_gpu_queue.put(gpu)
        
def initialize_config_queue(config_folder, config_queue, logger):    
    # initial load
    base = Path(config_folder)
    base_configs_files = [str(p) for p in list(base.rglob("*")) if p.is_file()]
    
    logger.info(f"Starting job queue for configs in folder {config_folder}")
    logger.info(f"Task list: {base_configs_files}")
    logger.info(f"#Task: {len(base_configs_files)}")
      
    # initialize config queue
    for config in base_configs_files:
        config_queue.put(config)
    
    return base_configs_files
      
      
def config_sync_thread(base_configs_files,
                       config_queue,
                       config_folder,
                       logger):
    # check for new configs and put into config_queue

    # run until all configs are done
    while config_queue.qsize() > 0:
        time.sleep(5)
        
        # check for new configs
        base = Path(config_folder)
        curr_configs_files = [str(p) for p in list(base.rglob("*")) if p.is_file()]
        new_config_files = [file for file in curr_configs_files if file not in base_configs_files]
        if len(new_config_files) > 0:
            logger.info(f"Detecting {len(new_config_files)} new configs")
            logger.info(f"New: {new_config_files}")
            
            for config in new_config_files:
                config_queue.put(config)
                
            # update base_config_files
            base_configs_files = curr_configs_files
    
def gpu_resource_thread(gpu_queue,
                        used_gpu_queue,
                        gpu_file,
                        logger):
    # manage a queue that holds usable gpus:
    base_gpus = get_gpus_from_file(gpu_file, logger, initial=True)
    logger.info(f"Using gpus {base_gpus} for job queue.")
    for gpu in base_gpus:
        gpu_queue.put(gpu)
        
    while True:
        curr_gpus = None
        # try to get current gpus - on error, return is None - wait until fix
        while curr_gpus is None:
            time.sleep(5)
            curr_gpus = get_gpus_from_file(gpu_file, logger)

        # check for new gpus
        add_gpus = [gpu for gpu in curr_gpus if gpu not in base_gpus]
        if len(add_gpus) > 0:
            for add_gpu in add_gpus:
                gpu_queue.put(add_gpu)
                logger.info(f"Providing new gpu {add_gpu} for job queue.")
     
        while used_gpu_queue.qsize() > 0:
            used_gpu = used_gpu_queue.get()
            # finish thread
            if used_gpu is None: 
                return None
            if used_gpu in curr_gpus:
                gpu_queue.put(used_gpu)
            else:
                logger.info(f"Removing gpu {used_gpu} after job was finished.")
            
        base_gpus = curr_gpus

def get_gpus_from_file(path, logger, initial=False):
    with open(path, "r") as stream:
        try:
            gpus = (yaml.safe_load(stream))['gpus']
            return gpus
        except yaml.YAMLError as exc:
            # info to
            if initial:
                raise exc
            else:
                logger.error("Cannot parse gpus from file. Please fix the file.")
                return None
                
if __name__ == '__main__':
    run_job_queue(gpu_file="gpus.yml",
                  config_folder="configs_paper/configs_cy16/attention/unet_resnet18/variants",
                 )