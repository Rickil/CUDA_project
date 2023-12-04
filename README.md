# IRGPUA Project

## Project Overview

Welcome to the IRGPUA project, an initiative by AMDIA to address a critical issue in the world of the internet. The challenge at hand involves the corruption of images by a notorious hacker, Riri, who has introduced garbage data into Internet pipelines. The corruption includes the insertion of lines of "-27" and modifications to pixel values, resulting in images with awful colors. AMDIA engineers have devised a solution, but it's currently too slow when executed on CPUs. Your mission is to implement a real-time solution on GPUs.

## Problem Statement

1. **Garbage Data Removal:**
   - Introduce a real-time GPU-based solution to remove garbage data introduced by the hacker in the form of lines of "-27" in the images.

2. **Pixel Value Modification:**
   - Correct pixel values according to the specified pattern: 0:+1, 1:-5, 2:+3, 3:-8, and so on.

3. **Histogram Equalization:**
   - Apply histogram equalization to improve the colors of the corrected images.

4. **Performance Optimization:**
   - Optimize algorithms and patterns for efficient GPU processing.

5. **Statistics and Sorting:**
   - Compute the total sum of pixel values for each image.
   - Optionally, sort the images based on their total sum.

6. **Alternative Libraries:**
   - Implement a second, faster version using libraries such as CUB, Thrust, minimizing the use of hand-made kernels.
