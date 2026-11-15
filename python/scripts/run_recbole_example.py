"""Example only. Needs recbole (and a dataset) installed. Not part of the numpy self-test."""

from recbole.quick_start import run_recbole

# Week-1 gate. Then swap dataset to ml-1m and config_file_list.
run_recbole(
    model="LightGCN",
    dataset="ml-100k",
    config_file_list=["config/base.yaml", "config/lightgcn.yaml"],
)

# Extension: pass the class, not a string, until it is registered.
# from model.lightgcn_adaptive import LightGCNAdaptive
# run_recbole(model=LightGCNAdaptive, dataset="ml-1m", config_file_list=["config/base.yaml", "config/lightgcn.yaml"])
