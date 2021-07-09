# Capstone Project

This is the final Capstone project for the Udacity Self-Driving Car Nanodegree course.
I developed software to guide a real self-driving car around a test track. Using the Robot Operating System (ROS) and created nodes for traffic light detection, classifier, trajectory planning and control.

---
## Rubric Points:
1. The code is built successfully and connects to the simulator.
2. Waypoints are published to plan Carla’s route around the track.
3. Controller commands are published to operate Carla’s throttle, brake, and steering.
4. Successfully navigate the full track more than once.

---
## Project Components

### Traffic Light Detector and Classifier
Car receives image from the camera, system can detect and classify a traffic light color.  
First part, is to detect a traffic light and Second part, is to classify a color of the detected image. If the traffic light is not detected, the component returns "None" for traffic light.

#### Traffic Light Detector
This project is aimed at detecting traffic light on the incoming image from Carla.
I've used [UNet Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and applied Loss function based on [dice coefficient](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2).
I've used a pre-trained model that looks like below:

```bash
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 96, 128, 1)    0
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 96, 128, 32)   320         input_1[0][0]
____________________________________________________________________________________________________
conv_1_2 (Conv2D)                (None, 96, 128, 32)   9248        conv2d_1[0][0]
____________________________________________________________________________________________________
maxpool_1 (MaxPooling2D)         (None, 48, 64, 32)    0           conv_1_2[0][0]
____________________________________________________________________________________________________
conv_2_1 (Conv2D)                (None, 48, 64, 64)    18496       maxpool_1[0][0]
____________________________________________________________________________________________________
conv_2_2 (Conv2D)                (None, 48, 64, 64)    36928       conv_2_1[0][0]
____________________________________________________________________________________________________
maxpool_2 (MaxPooling2D)         (None, 24, 32, 64)    0           conv_2_2[0][0]
____________________________________________________________________________________________________
conv_3_1 (Conv2D)                (None, 24, 32, 128)   73856       maxpool_2[0][0]
____________________________________________________________________________________________________
conv_3_2 (Conv2D)                (None, 24, 32, 128)   147584      conv_3_1[0][0]
____________________________________________________________________________________________________
maxpool_3 (MaxPooling2D)         (None, 12, 16, 128)   0           conv_3_2[0][0]
____________________________________________________________________________________________________
conv_4_1 (Conv2D)                (None, 12, 16, 256)   295168      maxpool_3[0][0]
____________________________________________________________________________________________________
conv_4_2 (Conv2D)                (None, 12, 16, 256)   590080      conv_4_1[0][0]
____________________________________________________________________________________________________
maxpool_4 (MaxPooling2D)         (None, 6, 8, 256)     0           conv_4_2[0][0]
____________________________________________________________________________________________________
conv_5_1 (Conv2D)                (None, 6, 8, 512)     1180160     maxpool_4[0][0]
____________________________________________________________________________________________________
conv_5_2 (Conv2D)                (None, 6, 8, 512)     2359808     conv_5_1[0][0]
____________________________________________________________________________________________________
convtran_6 (Conv2DTranspose)     (None, 12, 16, 256)   524544      conv_5_2[0][0]
____________________________________________________________________________________________________
up_6 (Concatenate)               (None, 12, 16, 512)   0           convtran_6[0][0]
                                                                   conv_4_2[0][0]
____________________________________________________________________________________________________
conv_6_1 (Conv2D)                (None, 12, 16, 256)   1179904     up_6[0][0]
____________________________________________________________________________________________________
conv_6_2 (Conv2D)                (None, 12, 16, 256)   590080      conv_6_1[0][0]
____________________________________________________________________________________________________
convtran_7 (Conv2DTranspose)     (None, 24, 32, 128)   131200      conv_6_2[0][0]
____________________________________________________________________________________________________
up_7 (Concatenate)               (None, 24, 32, 256)   0           convtran_7[0][0]
                                                                   conv_3_2[0][0]
____________________________________________________________________________________________________
conv_7_1 (Conv2D)                (None, 24, 32, 128)   295040      up_7[0][0]
____________________________________________________________________________________________________
conv_7_2 (Conv2D)                (None, 24, 32, 128)   147584      conv_7_1[0][0]
____________________________________________________________________________________________________
convtran_8 (Conv2DTranspose)     (None, 48, 64, 64)    32832       conv_7_2[0][0]
____________________________________________________________________________________________________
up_8 (Concatenate)               (None, 48, 64, 128)   0           convtran_8[0][0]
                                                                   conv_2_2[0][0]
____________________________________________________________________________________________________
conv_8_1 (Conv2D)                (None, 48, 64, 64)    73792       up_8[0][0]
____________________________________________________________________________________________________
conv_8_2 (Conv2D)                (None, 48, 64, 64)    36928       conv_8_1[0][0]
____________________________________________________________________________________________________
convtran_9 (Conv2DTranspose)     (None, 96, 128, 32)   8224        conv_8_2[0][0]
____________________________________________________________________________________________________
up_9 (Concatenate)               (None, 96, 128, 64)   0           convtran_9[0][0]
                                                                   conv_1_2[0][0]
____________________________________________________________________________________________________
conv_9_1 (Conv2D)                (None, 96, 128, 32)   18464       up_9[0][0]
____________________________________________________________________________________________________
conv_9_2 (Conv2D)                (None, 96, 128, 32)   9248        conv_9_1[0][0]
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 96, 128, 1)    33          conv_9_2[0][0]
====================================================================================================
```

Model has been trained using
```python
IMAGE_ROWS = 96
IMAGE_COLS = 128
COLORS = 3
SMOOTH = 1.
ACTIVATION = 'relu'
PADDING = 'same'
KERNEL_SIZE = (3, 3)
STRIDES = (2, 2)
```

### Traffic Light Classifier
Node for classifier is aimed at classfying traffic light. 
Classfier model is implemented like below:
```bash
_________________________________________________________________
Layer (type)                   Output Shape             Param #   
=================================================================
conv2d_3 (Conv2D)             (None, 64, 32, 32)        896       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2) (None, 32, 16, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)             (None, 32, 16, 32)        9248      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2) (None, 16, 8, 32)         0         
_________________________________________________________________
flatten_2 (Flatten)           (None, 4096)              0         
_________________________________________________________________
dense_3 (Dense)               (None, 8)                 32776     
_________________________________________________________________
dense_4 (Dense)               (None, 4)                 36        
=================================================================
```
---
## Waypoint Updater  
Waypoint updater performs the following at each current pose update

### Find closest waypoint
1) searching for the waypoint which has the closest 2D Euclidean distance to the current pose among the waypoint list  
   It's conducted with a KDtree algorithm. O(log n) in time complexity.
2) transforming the closest waypoint to vehicle coordinate system in order to determine wheter it is ahead of the vehicle and advance one waypoint if found to be  behind

### Calculate trajectory
1) The target speed at the next waypoint is calculated as the expected speed (v)  
   traversing the distance (s) : from the next waypoint to the traffic light stop line  
   the largest deceleration (a)
2) Using linear motion equations, it can be shown that v = sqrt(2 * a * s)
3) If there is no traffic light stopline, then target speed is set to the maximum

### Construct final waypoints
1) Pulished final waypoints by extracting the number of LOOKAHEAD waypoints at the calculated next point
2) The speed at published waypoints are set to the lower of target speed and maximum speed of the particular waypoint

---
## DBW

### Throttle and brake  
- Throttle and brake is controlled via PID controller  
1) inputs  
   - reference : linear x velocity of the twist command from waypoint follower
   - measurement : linear x velocity of the current velocity

2) outputs
   - positive : converted to throttle by clipping between 0 ~ 1
   - negative : converted to brake by normalizing and modulate tha max braking torque

### Steering angle 
- It is calculated using below equation
- steering angle = arctan(wheel base / turning radius) x steer ratio
- turning radius = current linear velocity / target angular velocity
- target angular velocity = current angular velocity x (current linear velocity / target linear velocity)
- steering value is filterd with a LowPass Filter to avoid fast steering changess


