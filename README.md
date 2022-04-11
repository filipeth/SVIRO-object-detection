# SVIRO-object-detection
This project aims to develop an object detection using transfer leaning from EfficientDet.

**Goal**: Build a solution for child seat localization in the passanger vehicle.

**Dataset**: The dataset used for this project is a made of synthetic data and it can be found [here](https://sviro.kl.dfki.de/).

**Problem**: Detect child seat in the passanger vehicle.

## Model 

For this problem I chose the model Efficientdet ([arxiv](https://arxiv.org/abs/1911.09070)). This model is said to achieve state-of-the-art 55.1mAP on COCO test-dev using fewer parammeters than other SOTA algorithms. Many versions of this models is available for finetunning as it is a good model for transfer learning. The EfficientDet model uses as backbone the [EfficientNet architecture](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html) that is made to obtain better accuracy and efficiency.

For this project I will be using  **EfficientDet-D3**, a 12.0M params model that achieved 47.5 mAP on COCO dataset. 

## Results

After trainning the model for **5000** steps we got the following metrics:

- DetectionBoxes_Precision/mAP: **0.225832**
- DetectionBoxes_Recall/AR@100: **0.408889**
- Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.226
- Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.604
- Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.056

After trainning the model for **30000** steps we got the following metrics:

- DetectionBoxes_Precision/mAP: **0.345371**
- DetectionBoxes_Recall/AR@100: **0.537969**
- Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
- Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.646
- Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.330


## Next Steps

In a short time we were able to train a detection model to detect object/persons in the backseat of 2 different car models, BMW i3 and Tesla Model3, with an accuracy close to the ones found by the SVIRO team. More time is needed to investigate the best approach to train this model, such as:

- Train 1 model for each car and compare the results with a single model for every car model.
- Train model on the full dataset for each car model.
- Train model with a higher number of steps.
- Apply other data augmentation techniques such as, rotations, color distortion and image occlusion.
- Train model with a bigger BATCH_SIZE.

## Credits

- [SVIRO dataset](https://sviro.kl.dfki.de/)
- [Tensorflow object detection](https://www.tensorflow.org/hub/tutorials/tf2_object_detection)
- [EfficientNet](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
- [EfficientDet](https://arxiv.org/abs/1911.09070)
