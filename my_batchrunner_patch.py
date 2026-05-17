# my_batchrunner_patch.py

from collections.abc import Iterable, Mapping
from typing import Any
import pickle
from mesa.model import Model
import os
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, set_start_method
from functools import partial
import mesa.batchrunner
import inspect

set_start_method("spawn", force=True)


def _make_model_kwargs(parameters: Mapping[str, Iterable[Any]]) -> list[dict[str, Any]]:
    '''my change is to return so that the model dont do all combinations, but just all parameters with their respective index (all param with index0, then all with index1 etc'''

    for param, values in parameters.items():
        if not isinstance(values, (list, tuple)):
            raise ValueError("Parameter '%s' must be an iterable." % param)
        if len(values) == 0:
            raise ValueError("Parameter '%s' is empty." % param)

    lengths = [len(v) for v in parameters.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"All parameter iterables must have same length, got {lengths}.")

    return [{k: v[i] for k, v in parameters.items()} for i in range(lengths[0])]


def batch_run(
    model_cls: type[Model],
    parameters: Mapping[str, Any | Iterable[Any]],
    number_processes: int | None = 1,
    iterations: int = 1,
    data_collection_period: int = -1,
    max_steps: int = 1000,
    display_progress: bool = True,
    path_save: str = 'NOPATHPROVIDED',
) -> list[dict[str, Any]]:
    
    '''my only change is to save the data instead of returning it'''
    
    #original code
    #runs_list = []
    #run_id = 0
    #for iteration in range(iterations):
    #    for kwargs in _make_model_kwargs(parameters):
    #        runs_list.append((run_id, iteration, kwargs))
    #        run_id += 1
    #I also change this part to ensure that for each param combination we have two iterations, since we can terminate batch in the middle
    
    runs_list = []
    run_id = 0
    #Version without baseline and pathway changes
    for kwargs in _make_model_kwargs(parameters):
        #each paramcomb can get multiple iterations
        for iteration in range(iterations): 
            runs_list.append((run_id, iteration, kwargs))
            run_id += 1          
       
    #Version with baseline for pathway paper
    '''for kwargs in _make_model_kwargs(parameters):
        if kwargs["PW_beta"] is None:
            # Standard behavior: multiple iterations
            for iteration in range(iterations):
                runs_list.append((run_id, iteration, kwargs))
                run_id += 1
        else:
            
            # Iteration 0: baseline
            run_kwargs_0 = kwargs.copy()
            runs_list.append((run_id, 0, run_kwargs_0))

            # Iteration 1: modified
            run_kwargs_1 = kwargs.copy()
            run_kwargs_1["beta"] = run_kwargs_1["beta"] * kwargs["PW_beta"]
            run_kwargs_1["watertemperature"] = run_kwargs_1["watertemperature"] + kwargs["PW_watertemperature"]
            runs_list.append((run_id, 1, run_kwargs_1))            
         
            run_id += 1  # Only increment once per paramcomb'''
        
    process_func = partial(
        mesa.batchrunner._model_run_func,
        model_cls,
        max_steps=max_steps,
        data_collection_period=data_collection_period,
    )

    with tqdm(total=len(runs_list), disable=not display_progress) as pbar:
        if number_processes == 1:
            for run in runs_list:
                data = process_func(run)
                
                ######### save data
                df_res = pd.DataFrame(data)
                #save the dictionaries from the last step, as it contains all info
                m = df_res['Step'].max()
                iteration = df_res['iteration'].unique()[0]
                model_info = df_res[df_res['Step']==m]['model_info'].values[0]
                dico_step_nodeid_virusinenv = model_info['dico_step_nodeid_virusinenv']
                with open(os.path.join(path_save, str(run[0])+'_'+ str(iteration)+"_dico_step_nodeid_virusinenv.pkl"), 'wb') as f:
                    pickle.dump(dico_step_nodeid_virusinenv, f)
             
                if 'dico_step_nodeid_birdidINFO' in model_info:
                    dico_step_nodeid_birdidINFO = model_info['dico_step_nodeid_birdidINFO']
                    with open(os.path.join(path_save, str(run[0])+'_'+ str(iteration)+"_dico_step_nodeid_birdidINFO.pkl"), 'wb') as f:
                        pickle.dump(dico_step_nodeid_birdidINFO, f)

                #small processing and save data
                df_res.rename(columns={'Step':'step','RunId':'runid'}, inplace=True)
                df_res['followedagent_info'] = df_res['followedagent_info'].astype(str)
                df_res.drop(columns=['model_info']).to_parquet(os.path.join(path_save, str(run[0]) +'_'+ str(iteration)+\
                                                                            "_df_res.parquet"),index=False)
                    
                pbar.update()
                                

        else:
            with Pool(number_processes) as pool:
                for data in pool.imap_unordered(process_func, runs_list):

                    ######### save data
                    run_id = data[0]["RunId"]
                    iteration = data[0]['iteration'] #TODO: CORRECT?
                    df_res = pd.DataFrame(data)
                    #save the dictionaries from the last step, as it contains all info
                    m = df_res['Step'].max()
                    model_info = df_res[df_res['Step']==m]['model_info'].values[0]
                    dico_step_nodeid_virusinenv = model_info['dico_step_nodeid_virusinenv']
                    with open(os.path.join(path_save, str(run_id)+'_'+ str(iteration)+"_dico_step_nodeid_virusinenv.pkl"), 'wb') as f:
                        pickle.dump(dico_step_nodeid_virusinenv, f)

                    if 'dico_step_nodeid_birdidINFO' in model_info:
                        dico_step_nodeid_birdidINFO = model_info['dico_step_nodeid_birdidINFO']
                        with open(os.path.join(path_save, str(run_id)+'_'+ str(iteration)+"_dico_step_nodeid_birdidINFO.pkl"), 'wb') as f:
                            pickle.dump(dico_step_nodeid_birdidINFO, f)

                    #save data
                    df_res.rename(columns={'Step':'step','RunId':'runid'}, inplace=True)
                    df_res['followedagent_info'] = df_res['followedagent_info'].astype(str)
                    df_res.drop(columns=['model_info']).to_parquet(os.path.join(path_save, str(run_id) +'_'+ str(iteration)+\
                                                                                "_df_res.parquet"),index=False)
         
                    pbar.update()


# Monkey patch
def patch_batchrunner():
    mesa.batchrunner._make_model_kwargs = _make_model_kwargs
    mesa.batchrunner.batch_run = batch_run
