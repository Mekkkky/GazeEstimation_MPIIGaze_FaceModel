# Gaze Estimation Demo 

Implementation of existing code, [here](https://github.com/hysts/pytorch_mpiigaze) is the source code.

In this implementation, only MPIIGaze has been downloaded and trained. If you are also interested in MPIIFaceGaze, you can open the source code and follow the instructions. 



## Note

YACS has been used to do configuration management.
Default parameters are specified in
[`gaze_estimation/config/defaults.py`](gaze_estimation/config/defaults.py)

You can overwrite those default parameters using a YAML file like
[`configs/mpiigaze/lenet_train.yaml`](configs/mpiigaze/lenet_train.yaml).


### Training and Evaluation

```bash
python train.py --config configs/mpiigaze/lenet_train.yaml
python evaluate.py --config configs/mpiigaze/lenet_eval.yaml
```


### Demo

This demo program runs gaze estimation on the video from a webcam.

Specify the model path and the path of the camera calibration results
in the configuration file as in
[`configs/demo_mpiigaze_resnet.yaml`](configs/demo_mpiigaze_resnet.yaml).

```bash
python demo.py --config configs/demo_mpiigaze_resnet.yaml
```
