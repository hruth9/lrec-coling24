# The Key Points: Using Feature Importance to Identify Shortcomings in Sign Language Recognition Models

The following repository details an example process of the experiments detailed in the above paper. 

### Paper Abstract
Pose estimation keypoints are widely used in Sign Language Recognition (SLR) as a means of generalising to unseen signers. Despite the advantages of keypoints, SLR models struggle to achieve high recognition accuracy for many signed languages due to the large degree of variability between occurrences of the same signs, the lack of large datasets and the imbalanced nature of the data therein. In this paper, we seek to provide a deeper analysis into th ways that these keypoints are used by models in order to determine which are the most informative to SLR, identify potentially redundant ones and investigate whether keypoints that are central to differentiating signs in practice are being effectively used as expected by models. 

### Pose-Estimation Keypoints
The keypoints used here were extracted using MediaPipe's Holistic package, specifically the Pose and Hand solutions. This extraction results in a total of 75 keypoints, each composed of x, y and z coordinates. 

[mediapipe pose map](images/mp_body.png)
Figure 1: MediaPipe Pose keypoint mapping. [Ref](https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md)

[mediapipe hand map](images/mp_hand.png)
Figure 2: MediaPipe Hand keypoint mapping. [Ref](https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md)

A mapping of keypoint to associated body part is provided [here](misc/kp_map.json). For clarity of reference within the adjoining notebook: 
 - Keypoint indices 0-32 refer to the body pose
 - Keypoint indices 33-53 refer to the left hand
 - Keypoint indices 54-74 refer to the right hand

### Model
The model architecture used in these experiments is as detailed in [1](https://openaccess.thecvf.com/content/ICCV2023W/ACVR/papers/Holmes_From_Scarcity_to_Understanding_Transfer_Learning_for_the_Extremely_Low_ICCVW_2023_paper.pdf) and summarised below: 

1. Four stacked 1D convolutional layers with increasing kernel size (3, 5, 7, 9) to learn local temporal patterns for each coordinate individually. Padding added to maintain sequence length. 
2. Embeddings generated independently from each frame in the sequence to learn non-linear relationships between individual keypoint coordinates. This consists of four blocks containing:
    - Linear layer
    - Layer normalisation
    - GeLU activation function (except last block)
    - Dropout (except last block)
3. Similar to first step except that each block now contains two convolutional layers with GeLU activation between them. This step detects local temporal patters within the embedding sequence with a limited receptive field. 
4. Global temporal information is learned using self-attention in which the receptive field covers the entire sequence. 
5. The resulting vector is used as input to the final classification layer. 

This architecture is illustrated below: 

[model architecture](images/PoseFormer.png)
Figure 3: Model architecture. [1](https://openaccess.thecvf.com/content/ICCV2023W/ACVR/papers/Holmes_From_Scarcity_to_Understanding_Transfer_Learning_for_the_Extremely_Low_ICCVW_2023_paper.pdf)


The same hyperparameters were also used in these experiments and are summarized below: 

| Hyperparameter | Value |
| --- | --- |
| Batch size | 64 |
| No. attention layers | 4 | 
| No. attention heads | 8 |
| Feature size | 134 |
| Embedding size | 192 |
| Initial learning rate | 0.0003 |

Once trained, the model state from the best performing epoch in terms of validation accuracy was used to determine macro F1 scores for the baseline and all permuted runs.

### Permutation Feature Importance
Permutation Feature Importance[2](https://link.springer.com/content/pdf/10.1023/a:1010933404324.pdf) is a model-agnostic procedure for determining the features that most contribute to the performance of a trained model. The procedure typically involves randomly shuffling the values of the feature of interest in order to remove any association between that independent feature and the target variable. If this feature contributes significantly to correct predictions, this shuffling operation should result in a marked decrease in performance. Conversely, if the decrease in performance is negligible, this suggests that this feature does not contribute significantly to classification decisions. Here, we use this measure of importance with a slight modification which avoids the potentially costly shuffling operation. We, in place of shuffling a given feature, replace it with values drawn uniformly at random within the range of all features in the dataset. An example process can be found in the adjoining notebook titled [example_process.ipynb](example_process.ipynb).


### References
[1]  Ruth Holmes, Ellen Rushe, Mathieu De Coster, Maxim Bonnaerens, Shin’ichi Satoh, Akihiro Sugimoto, and Anthony Ventresque. 2023. From scarcity to understanding: Transfer learning for the extremely low resource irish sign language. *In Proceedings of the Eleventh International Workshop on Assistive Computer Vision and Robotics, in conjunction with IEEE/CVF International Conference on Computer Vision*, pages 2008–2017.

[2]  Leo Breiman. 2001. Random forests. *Machine learning*, 45:5–32.