import uproot
print(f"uproot verion: {uproot.__version__}")
import distributed
print(f"distributed verion: {distributed.__version__}")
import numpy as np
import awkward as ak
import dask_awkward as dak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from distributed import Client, performance_report
import json 
import glob
import os
import tqdm
import time
from itertools import islice
import copy
import dask
from coffea.dataset_tools import (
    max_chunks
)
from dask_gateway import Gateway


def divide_chunks(data: dict, SIZE: int):
    """
    This takes a big sample of a dataset consisting of multiple root files and divides them to smaller sets of root files.
    Similar to coffea.dataset_tools maxfile function, but not exactly the same 
    """
    it = iter(data)
    for i in range(0, len(data), SIZE):
      yield {k:data[k] for k in islice(it, SIZE)}


if __name__ == '__main__':
    do_regular_restart = False
    # client = Client(n_workers=1,  threads_per_worker=1, processes=True, memory_limit='0.7 GiB')
    # gateway = Gateway()
    gateway = Gateway(
        "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
        proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    )
    cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
    client = gateway.connect(cluster_info.name).get_client()
    # sample_path = "./input_file.json"
    sample_path = "./input_file_Big.json"
    with open(sample_path) as file:
        samples = json.loads(file.read())
    samples  = max_chunks(samples, 20)
    # dataset = list(samples.keys())[0]
    # sample = list(samples.values())[0]
    n_files_processed = 0
    for dataset, sample in tqdm.tqdm(samples.items()):
        max_file_len = 1
        smaller_files = list(divide_chunks(sample["files"], max_file_len))
        for idx in tqdm.tqdm(range(len(smaller_files)), leave=False):
            smaller_sample = copy.deepcopy(sample)
            smaller_sample["files"] = smaller_files[idx]
            # fnames = list(smaller_sample["files"].keys())
            # input = {fname: "Events" for fname in fnames}
            events = NanoEventsFactory.from_root(
                smaller_sample["files"],
                schemaclass=NanoAODSchema,
                metadata= smaller_sample["metadata"],
                uproot_options={"handler" : uproot.XRootDSource}
            ).events()
            nmuons = ak.num(events.Muon, axis=1)
            muon_selection = (
                events.Muon.pt > 20 &
                nmuons == 2
            )
            muons = events.Muon[muon_selection]
            dask.compute(muons.pt)
            if do_regular_restart:
                client.restart(wait_for_workers=False)
            print(f"n_files_processed so far: {n_files_processed}")
            n_files_processed += 1
    
    
    print(f"Success! gone through all {n_files_processed} files!")