# ActionLocalization_CVPR16

This folder contains demo code for “What If We Do Not Have Multiple Videos of the Same Action? -- Video Action Localization Using Web Images, CVPR16”

The main file is demo_Code.m.

In order to use the code, you need to compute Action Proposals and CNN features etc. We have include pre-computed proposals for ‘Spatio-Temporal Object Detection Proposals, ECCV’14’. Please see comments in the code for more details.

The video frames should be in folder given by “Videos_frame_Path”.

This software requires the following resources, which are already integrated. We have changed the codes of ([2] and [3]); changed version are included.
[1] A. Vedaldi and K. Lenc. Matconvnet – convolutional neural
networks for matlab. 2015
[2] M. Cho, S. Kwak, C. Schmid, and J. Ponce. Unsupervised
object discovery and localization in the wild: Part-based
matching with bottom-up region proposals. In CVPR, 2015.
[3] M. Schmidt. Graphical model structure learning with l1-
regularization. In Ph.D. Thesis, 2010.


In case of any comments, please drop me an email at waqas5163@gmail.com.


If you found this code useful in your research, please cite the following paper:

 @InProceedings{Sultani_2016_CVPR,
author = {Sultani, Waqas and Shah, Mubarak},
title = {What If We Do Not Have Multiple Videos of the Same Action? -- Video Action Localization Using Web Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}


Thank you!
