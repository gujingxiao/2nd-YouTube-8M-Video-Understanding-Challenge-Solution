# Model Documentation
Hi guys! Welcome to our first GIT. From now on, well, to the end of the competition, we will share our code and models here. Although our ranking is not that good and our models are naive, we still keep trying and are finding out better-performed models. No more to say, come on guys! To get our Metal! (Bronze? No! Hold the Silver and looking forward to the GOLD!)

Our code is based on the [YouTube-8M Tensorflow Starter](https://github.com/google/youtube-8m) Code.

## History
    09th-June-2018, Joined the competition.
    29th-June-2018, Created, commit version v0.1 which gets GAP=0.8513 upon video-level data.
    04th-July-2018, Updated, Add MultiTaskCombineChainModel which gets GAP = 0.86035 upon video-level data.
    20th-July-2018, Updated, Add new frame level models which gets greater than GAP= 0.865 upon frame-level data.
    05th-August-2018, Updated, Add MultiEnsembleFrameModelLF and update GatedDbofWithNetFVModelLF to GAP = 0.87089.

## Instructions
#### Video Level Models

|Models|Loss Function|Base LR|Batch Size|LR_Decay|Other Parameters|GAP|
|:---|:---|:---|:---|:---|:---|:---|
|DeepCombineChainModel|MultiTaskCrossEntropyLoss|0.01|1024|0.85|deep_chain_layers=3, deep_chain_relu_cells=1024|0.85407|
|MultiTaskCombineChainModel|MultiTaskChainCrossEntropyLoss|0.01|1024|0.85|chain_layers_1=3, chain_elu_cells=896, chain_layers_2=2, chain_leaky_relu_cells=896|0.86035|

#### Frame Level Models

|Models|Classifier|Loss Function|Base LR|Batch Size|LR_Decay|Other Parameters|GAP|
|:---|:---|:---|:---|:---|:---|:---|:---|
|MultiCombinedFeatureFrameModelLF|DeepCombineChainModel|MultiTaskCrossEntropyLoss|0.002|128|0.85|netvlad_cluster_size=56, netvlad_hidden_size=768, fv_cluster_size=56, fv_hidden_size=768, fv_coupling_factor=0.01, dbof_cluster_size=2048, dbof_hidden_size=512|0.86930|
|GatedDbofWithNetFVModelLF|DeepCombineChainModel|MultiTaskCrossEntropyLoss|0.002|128|0.85|fv_cluster_size=52, fv_hidden_size=1024, fv_coupling_factor=0.01, dbof_cluster_size=2560, dbof_hidden_size=1024|0.87089|
|MultiEnsembleFrameModelLF|MultiEnsembleChainModel|MultiTaskChainCrossEntropyLoss|0.002|128|0.8|netvlad_cluster_size=40, netvlad_hidden_size=736, fv_cluster_size=40, fv_hidden_size=736, fv_coupling_factor=0.01, dbof_cluster_size=2048, dbof_hidden_size=736|Training|

## Training Tricks
* It's better to use the model right after an entire epoch training. For example, while the batch size is 1024, the entire epoch is equal to 3800 steps. So if the batch size is getting smaller, the corresponding steps also need to be changed.

* No need to evaluate all validation data. What I do is to keep only 1/4th validation data for evaluation and put others into training data. In this way, we can get slightly increasing of GAP.

* I'm working on making an analysis on all data. And then do data augmentation.
