# Model Documentation
Hi guys! Welcome to our first GIT. From now on, well, to the end of the competition, we will share our code and models here. Although our ranking is not that good and our models are naive, we still keep trying and are finding out better-performed models. No more to say, come on guys! To get our Metal! (Bronze? No! Hold the Silver and looking forward to the GOLD!)

Our code is based on the [YouTube-8M Tensorflow Starter](https://github.com/google/youtube-8m) Code.

## History
    09th-June-2018, Joined the competition.
    29th-June-2018, Created, commit version v0.1 which gets GAP=0.8513 upon video-level data.
    04th-July-2018, Updated, Add MultiTaskCombineChainModel which gets GAP=0.86035 upon video-level data.

## Instructions
* Video Level Models

|Models|Loss Function|Base LR|Batch Size|LR_Decay|Other Parameters|GAP|
|:---|:---|:---|:---|:---|:---|:---|
|DeepCombineChainModel|MultiTaskCrossEntropyLoss|0.01|1024|0.85|deep_chain_layers=3, deep_chain_relu_cells=1024|0.85407|
|MultiTaskCombineChainModel|MultiTaskChainCrossEntropyLoss|0.01|1024|0.85|chain_layers_1=3, chain_elu_cells=896, chain_layers_2=2, chain_leaky_relu_cells=896|0.86035|

* Frame Level Models

|Models|Loss Function|Base LR|Batch Size|LR_Decay|Other Parameters|GAP|
|:---|:---|:---|:---|:---|:---|:---|
|AttentionBasedLstmMoeModel|MultiTaskCrossEntropyLoss|0.005|256|0.85|None|None|
