# On the Completeness of Invariant Geometric Deep Learning Models

Pytorch implementation of the submission *On the Completeness of Invariant Geometric Deep Learning Models*

## Dependencies

python=3.8

pytorch=2.1.0+cu121

pytorch-lightning=2.1.3

torch_geometric=2.4.0

Dependencies downloaded by `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html`

Additional dependencies: setproctitle, yacs, tensorboardx



## Train and Test GeoNGNN on molecular dynamics datasets 


### GeoNGNN on rMD17 and MD22 
To train and test GeoNGNN, run 

```bash
python scripts/ngnn_script_MD.py [--model <model_name>] [--ds <dataset_name>] [--dname <data_name>] [--devices <device_id>] [--data_dir <data_dir>] [--version <version>] [--resume] [--skip_train] [--skip_test] [--ckpt <checkpoint_path>] [--merge <merge hparam list>] [--use_wandb] [--proj_name <project_name>]
```

Arguments description:

+ --model: ["GeoNGNN"], corresponds to the model name.
+ --ds: ["md17", "md22"], corresponds to the dataset name.
+ --dname: In MD17, choices are the name of molecules, which can be found in [md17](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MD17.html?highlight=md17#torch_geometric.datasets.MD17) (Note that the results are evaluated on revised MD17, therefore the dnames should be "revised **" accordingly); In MD22, choices include ["Ac", "DHA", "AT", "ATCG", "Bu", "Do", "St"], represents the abbreviation of molecules in [md22](http://www.sgdml.org/#datasets).
+ --devices: The number of the GPU devices. For example, if you want to use GPU 0, please specify it as 0. If no devices are specified, the code will run on the GPU with minimum memory usage. If it's set to -1, the code will run on CPU. Currently, the code does not support multi-GPU training.
+ --data_dir: The directory of the dataset. By default, it is "~/datasets/MD17".
+ --version: The version of the log. By default, it is "NO_VERSION".
+ --resume: Whether to resume the training from the latest checkpoint. By default, it is False.
+ --skip_train: Whether to skip the training process. By default, it is False.
+ --skip_test: Whether to skip the test process. By default, it is False.
+ --ckpt: The path of the checkpoint. By default, it is None.
+ --merge: The hparams to merge. For example, if you want to modify the hparam model_config.block_num to 4, then use --merge model_config.block_num 4.
+ --use_wandb: Whether to use wandb to log the training process. If set as False, the code will use tensorboard to log the training process. By default, it is False.
+ --proj_name: The name of the project in wandb. By default, it is None.

Hyperparameters are specified in the dictionary `hparams`.

For example, to run GeoNGNN on revised MD17 (revised ethanol) using gpu 0:
```bash
python scripts/ngnn_script_MD.py --model GeoNGNN --ds md17 --dname 'revised ethanol' --devices 0 --data_dir <your_data_dir> --version test 
```

<font color=red>**For Training and Testing on md22 dataset, please first download the raw numpy dataset file mannually from the [source](http://www.sgdml.org/#datasets), and put it under the path `<data_dir>/<dname>/raw/*.npz>`, where `<data_dir>` and `<dname>` are specified script parameters metioned previously**</font>.

### GeoNGNN on 3BPA

We provide the dataset for 3BPA in the directory `./data/triBPA`. One can also refer to its original paper (cited in the main paper) for the source file. To train and test GeoNGNN on 3BPA, run 

```bash
python scripts/ngnn_script_3bpa.py [--model <model_name>] [--dname <data_name>] [--devices <device_id>] [--data_dir <data_dir>] [--version <version>] [--resume] [--skip_train] [--skip_test] [--ckpt <checkpoint_path>] [--merge <merge hparam list>] [--use_wandb] [--proj_name <project_name>]
```

Arguments are similar to the description in the previous section. One should first move the provided dataset to the path `<data_dir>`.


### Vanilla DisGNN

To run Vanilla DisGNN, one can simply set the hyperparameter of GeoNGNN `model_config.ablation_innerGNN` to "True", and set `model_config.outer_layer_num` to the desired number of Vanilla DisGNN layers.

## Train and Test GeoNGNN on synthetic datasets

We construct a synthetic dataset based on the counterexamples proposed by [Li. et al.](https://arxiv.org/pdf/2302.05743.pdf), as mentioned in the main paper. The corresponding dataset file is `./datasets/CE.py`. To test GeoNGNN's seperation power on the synthetic dataset, run 

```bash
python scripts/ngnn_script_CE.py [--dname <data_name>] [--combine] [--devices <device_id>]  [--version <version>] [--use_wandb] [--proj_name <project_name>]
```

Arguments description:

+ --dname: The name of the counterexample. By default, it is "r12-0". It can be chosen from ["r12-0", "r12-1", "r12-2", "r12-3", "r12-4", "r12-5", "r20-0", "cr8-0", "cr8-1", "cc-0"], representing the 10 isolated counterexamples. The 7 combinatory cases will be generated if the argument `--combine` is set to True.
+ --combine: Whether to to use combinatory counterexamples. By default, it is False.

Other arguments are similar to the main model.






## Train and Test GeoNGNN_C on chiral datasets

GeoNGNN-C is independently implemented in the directory `./GeoNGNN_variants/GeoNGNN_C`. One should first cd to the directory `./GeoNGNN_variants/GeoNGNN_C`.

Additional dependency should also be installed: pandas.

To train and test GeoNGNN_C on the two datasets, one should first install the two datasets from [source](https://figshare.com/s/e23be65a884ce7fc8543) and put the dataset under the path as follows:

**<data_dir>/raw/ds/**: `train_small_enantiomers_stable_full_screen_docking_MOL_margin3_234622_48384_24192.pkl`, `validation_small_enantiomers_stable_full_screen_docking_MOL_margin3_49878_10368_5184.pkl`,
`test_small_enantiomers_stable_full_screen_docking_MOL_margin3_50571_10368_5184.pkl`


**<data_dir>/raw/rs/**: `train_RS_classification_enantiomers_MOL_326865_55084_27542.pkl`, `validation_RS_classification_enantiomers_MOL_70099_11748_5874.pkl`, `test_RS_classification_enantiomers_MOL_69719_11680_5840.pkl`

Then, run the following command to process the dataset:

```bash
python datasets/process_DS.py --data_dir <data_dir>; python datasets/process_RS.py --data_dir <data_dir>
```

After sucessfully processing the dataset, run the following command to train and test GeoNGNN_C:

```bash
python scripts/ngnn_script_DS_RS.py [--model <model_name>] [--ds <dataset_name>] [--dname <data_name>] [--devices <device_id>] [--data_dir <data_dir>] [--version <version>] [--resume] [--skip_train] [--skip_test] [--ckpt <checkpoint_path>] [--merge <merge hparam list>] [--use_wandb] [--proj_name <project_name>]
```

The overall arguments are similar to the main model. Different arguments are listed below:

+ --model: ["GeoNGNN_C"], corresponds to the model name.
+ --ds: ["DockingScore", "rs"], corresponds to the two dataset names. DockingScore is the dataset for enatiomer ranking and binding affinity prediction tasks, which share the same underlying dataset and differs in the evaluation metric. rs is the dataset for R/S classification task.
+ --dname: One should not specify the dname for the two datasets, since these two datasets do not have sub-datasets.

Note that in both two datasets, the script will first batch and transform the datas instead of doing them on the fly, for faster training and testing. Therefore the pre-processing may take a while. 


## Train and Test GeoNGNN on QM9
```bash
python scripts/ngnn_script_QM9.py [--model <model_name>] [--ds <dataset_name>] [--dname <data_name>] [--devices <device_id>] [--data_dir <data_dir>] [--version <version>] [--resume] [--skip_train] [--skip_test] [--ckpt <checkpoint_path>] [--merge <merge hparam list>] [--use_wandb] [--proj_name <project_name>]
```

+ `--dname`: The index of the QM9 property, ranging from 0 to 18. The correspondence between the indices and the property index to property name correspondence can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html?highlight=qm9#torch_geometric.datasets.QM9).

**About hyperparameters:** We adopted the default hyper-parameters (in `./hparams/GeoNGNN_qm9`) for most property training, with the following exceptions: we adjusted the learning rate to 3e-4 for properties 12-15, and to 5e-4 for properties 5 and 11.


**For DisGNN:** To switch the model to DisGNN, simply set `ablation_innerGNN` to True and `outer_layer_num` to 5 in `./hparams/GeoNGNN_qm9`. The hyperparameter adjustments remain the same as previously stated.

**About gap property:** Consistent with DimeNet, we calculate the homo-lumo gap property by using the models trained on homo and lumo. To calculate the error on the gap, you can use the script `cal_gap.py` by specifying the trained model paths.

