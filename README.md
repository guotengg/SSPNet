# SSPG: Scale and Spatial Priors Guided Generalizable and Interpretable Pedestrian Attribute Recognition
A novel SSPG is proposed with the guidence of both scale and spatial prior for Pedestrian Attribute Recognition (PAR). The SSPG is mainly composed of the AFSS and PLE modules. The AFSS module learns to provide reasonable scale prior information for different attribute types, allowing the model to focus on different levels of feature maps with varying semantic granularity. The PLE module reveals potential attribute spatial prior information, which avoids unnecessary attention on irrelevant areas and lowers the risk of model over-fitting. 

![图片](https://user-images.githubusercontent.com/91515102/232787432-d9c88c54-751b-491f-b733-564c7b77bce1.png)

Overview of the proposed pipeline (The arrows show the forward path of our pipeline, which passes through the backbone, feature pyramid to extract multi-scale feature maps. Then, the prior knowledge is incorporated with AFSS and PLE module. Finally, the prediction are output based on the hierarchical recognition structure, as well as the attribute localization with modified CAM.)
