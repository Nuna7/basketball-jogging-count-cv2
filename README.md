**Project Overview:**
- Detect basketball using HSV value, masking techniques, and contours.
- Track ball movement by storing initial bounce coordinates.
- Identify ball bounces when reaching a specified distance from the initial coordinates.
- Using frame number to track fastest and slowest bounce

**Specific to the video:**
- This project is specific to the video or the ball movement.
- The logic and distance is tuned to the temporal characteristics of the ball movement.

**Main Logic:**
1. Detect the basketball using HSV value, masking, and contours.
2. Track ball coordinates and store initial bounce coordinates.
3. Identify a bounce when the ball reaches a distance `REACH_DISTANCE` from the initial coordinates.
4. Count bounces when the ball moves a distance greater than `BOUNCE_DISTANCE + 10`.
5. Fastest and slowest bounce are identify using the frame number

**Important Parameters:**
- `REACH_DISTANCE`: Distance from initial coordinates considered as reaching the ground.
- `BOUNCE_DISTANCE`: Distance moved after which a bounce is counted.

**NOTE**
This project only use cv2 and the analysis perform can significantly improve by adding deep learning techniques.

Watch the couting of jogging (bouncing) of ball using opencv [here](./BOUNCE_COUNT_WITH_CV2.mp4)
