## Abstract

>Human-object interaction (HOI) detection extends beyond the localization of humans and objects in images to identify the interactions between them. Existing approaches predominantly frame the HOI task as a detection problem, emphasizing the identification of human and object-bounding boxes alongside their interactions. However, we argue that this bounding-box-based paradigm is inherently limited in achieving fine-grained HOI detection due to its coarse localization. In this work, we focus on a more fine-grained case of the HOI: Segmentation-grounded Human-object Interaction (SHOI) detection. This new task shifts the focus from coarse bounding boxes to pixel-level masks, enabling more precise human-object interaction recognition. First, we introduce three benchmark datasets—VS-COCO, HICO-SG, and InterSeg to build the benchmark of SHOI and establish a comprehensive benchmark including a number of two-stage and one-stage models, systematically evaluating their effectiveness in SHOI detection. Furthermore, we introduce a novel baseline model, Ag-BiGT, agent-based query with bipartite graph-transformer.  Specifically, we first leverage a bipartite graph transformer to refine the human and object nodes by long-range and local context reasoning modules. Subsequently, we exploit multiple agents to generate spare-aware interaction queries integrated with the human-object pairs. Extensive experiments on both SHOI and HOI tasks demonstrate our proposed Ag-BiGT is superior to other state-of-the-art HOI models on a suite of SHOI and HOI datasets.  
<img src="figure/model.jpg" alt="Your Image" width="600" height="300">


## Comparison
SHOI detection shares the same goal as HOI detection, also aiming to predict a set of interaction triples, but the only difference lies in the form of localization for humans and objects. More concretely, SHOI   emphasizes generated detailed and nuanced masks for each instance instead of the conventional coarse bounding boxes.

<img src="figure/compare2.jpg" alt="Your Image" width="375" height="200">

Human-object Interaction (HOI) detection vs. Segmentation-grounded Human-Object Interaction (SHOI) detection.  (Left) illustrates an examplefrom the HICO-DET dataset. (Right) showcases the corresponding masks from our curated HICO-SG dataset. The conventional HOI task localizes humans and objects via bounding boxes involving irrelevant pixels. Our proposed SHOI task defines a more fine-grained and cleaner HOI representation with more accurate localizations of objects.

## Datastes

We present construct three SHOI datasets. 
For more comprehensive construction details and dataset statistic, please refer to the paper.

<img src="figure/dataset.jpg" alt="Your Image" width="600" height="300">


#### Automated Construction of SHOI Dataset using Pipeline
In this study, we introduce a novel approach for the automated creation of Spatial Human-Object-Interaction (SHOI) datasets.  Our method leverages segmentation models in conjunction with multimodal large-scale models within a cohesive pipeline.  This pipeline facilitates the seamless generation of SHOI datasets, thereby streamlining the data acquisition process.
#### Enhancement of Existing and Widely Used HOI Datasets
Furthermore, we contribute to the advancement of the field by enhancing existing and widely adopted Human-Object-Interaction (HOI) datasets.  Through careful curation and augmentation techniques, we improve the quality and diversity of the dataset, addressing potential limitations present in conventional datasets.  

<img src="figure/step2.jpg" alt="Your Image" width="600" height="250">

For more examples of image datasets, please refer to the Datasets folder.

