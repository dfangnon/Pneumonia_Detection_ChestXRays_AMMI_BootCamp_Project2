# Pneumonia_Detection_ChestXRays_AMMI_BootCamp_Project2
In this experimented project, we reimplemented some existing algorithms and also tried others capable of detecting pneumonia from ChestX-rays at a level above that of practicing radiologists. The existing algorithm we reimplemented is CheXNet, which is a 121-layer convolutional neural network trained on ChestX-ray14, the largest publicly accessible ChestX-ray dataset, containing over 100,000 images of frontal X-rays of 14 diseases. As a new experiment, we implemented the CheXResNet152V2 model for comparison. In order to assess the performance of the models for comparison, a set of tests annotated by radiologists was used. According to the experiments in the base article, the existing ChexNet model outperforms the average radiologist using the F1 measure. For the detection of the 14 ChestX-ray14 diseases, we note that the new model experimented, CheXResNet152V2 performs better than the existent model, CheXNet.