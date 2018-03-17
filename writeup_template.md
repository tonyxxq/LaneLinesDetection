# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 

-  I converted the image to grayscale.
-  Using a threshold makes a binary image. 
- Gaussian smoothing  to reduce image noise. 
- Edge detection using Canny operator.
- Using Hoffman transform to find a straight lines.
- Merge the output image to the original image,and write it to a file.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

- Go through each line, and judge whether the line is in the left lane or on the right lane, and put in two lists according to the size of the slope. Use cv2.fitLine to find the most suitable two straight lines. The bottom and 2/3 of the image are taken as the start and end of the Y axis. So each line gets two points， drawn respectively.

<img src="./test_images_output/whiteCarLaneSwitch.jpg" width="480" alt="Combined Image" />


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the road is a curve, the lane line detection is inaccurate.

Another shortcoming could be If it is at night or light condition is not very good, lane detection is not accurate.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
