clc
clear all
close all
%% AL Set Path

Image_Dataset_Path='../Data/UCF_Sports/Google_Images';
Noisy_Image_Dataset_Path='../Data/UCF_Sports/Google_ImagesNoisy';
Coloc_Data_Path='../Data/UCF_Sports/Google_Images_Coloc_Data';
Coloc_Result_Path='../Data/UCF_Sports/Google_Images_Coloc';
Coloc_BBX_Path='../Data/UCF_Sports/Google_Images_BBX';
Coloc_BBX_CNN_Path='../Data/UCF_Sports/Google_Images_CNN_BBX';
ImageCNNPath='../Data/UCF_Sports/Google_Images_CNN';

Videos_frame_Path='../Data/UCF_Sports/Ground_Truth_Video';
Video_Proposal_Path='../Data/UCF_Sports/Dan_proposals';
Video_Motion_Path='../Data/UCF_Sports/Motion_Mask';
Video_Motion_Score_Path='../Data/UCF_Sports/Motion_Score';
Motion_Salient_Proposals='../Data/UCF_Sports/BBX_MotionSalient';

Video_Proposals_CNN_Path='../Data/UCF_Sports/Videos_Proposal_CNN';
Video_Proposals_Recons_Path='../Data/UCF_Sports/Reconstruction_Rank';
GT_Path='../Data/UCF_Sports/Ground_Truth_Video';
Visulization_Path='../Data/UCF_Sports/Visualization';


MatConvNet_Path='./MatConvNet';
L1_General_Path='./L1General_modified';
addpath(genpath('./UODL_v1_modified'));
addpath(genpath('/vlfeat-0.9.20'));
addpath(genpath(L1_General_Path));


%% 1. Find Noisy Images:
%  i. Compute CNN features from whole images
%  ii. Use Random Walk to find low relevance images.   

% i. Compute CNN features from whole images
disp('1. Computing CNN features of Web Images ....')
Compute_CNN_Images(Image_Dataset_Path,ImageCNNPath,MatConvNet_Path)

% ii. Use Random Walk to find low relevance images.
disp('1.1 Removing (some of) Noisy images using Random walk ....')
RandomWalk_NoisyImages(Image_Dataset_Path,ImageCNNPath,Noisy_Image_Dataset_Path);

%% 2.  Perform Collocalization to obtain Bounding boxes in images:
% We used the following method to achieve this:  
%Proceedings{cho2015,
%  author = {Minsu Cho and Suha Kwak and Cordelia Schmid and Jean Ponce},
%  title = {Unsupervised Object Discovery and Localization in the Wild: Part-based Matching with Bottom-up Region Proposals},
%  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
%  year = {2015}
% vlf_path = './vlfeat-0.9.20/toolbox/vl_setup';
% rp_root = './rp-master';             % randomized prim: bounding box proposal
% run(vlf_path);
% addpath('./UODL_v1_modified/hough-match');
% addpath('./UODL_v1_modified/commonFunctions/');
% addpath(genpath('./UODL_v1_modified/HOG/'));
% addpath(fullfile(rp_root,'cmex'));
% addpath(fullfile(rp_root,'matlab'));
% addpath(genpath('./UODL_v1_modified/tools'))
% set_path


disp('2. Automatic  Bounding Box extraction in Web images ....')

% Extract region Proposals from image
 extract_boxes('VOC2007_6x2',Image_Dataset_Path,Coloc_Data_Path);
% Extract GIST feature from image
 extract_gist('VOC2007_6x2',Image_Dataset_Path,Coloc_Data_Path);
% Peform Col-localization using PHM
 run_localization_fastW('VOC2007_6x2', 5, 10,Image_Dataset_Path,Coloc_Data_Path,Coloc_Result_Path);	% running faster
% Save Colocalized Bounding Boxes.

disp('2.1 Saving automatically obtained Bounding Box ....')

save_Image_proposal_BBX(Coloc_Result_Path,Coloc_Data_Path,Coloc_BBX_Path)
%% 3. Compute CNN features in BBX obtained from 2.

disp('3. Compute CNN features within Bounding Boxes in Web Images ....')
Compute_CNN_Image_BBX(Image_Dataset_Path,Coloc_BBX_Path,Coloc_BBX_CNN_Path,MatConvNet_Path)


%%
% Untill now, we have obtained CNN features of Bounding Boxes in Web Images. 

%  In what follows, we will extract CNN features of video proposals in key frames.


%% 4 Obtain Video Proposals

disp('4. Please compute Action Proposals, We have not provided their codes here as they have their own dependencies')

% To obtain video proposals, you can use any proposals method such as:

% Action Localization with Tublets from Motion, CVPR,2014
% Spatio-Temporal Object Detection Proposals, ECCV, 2014
% APT: Action Localization Proposals from Dense Trajectories, BMVC 2015.


%% 5 Compute Motion Scores.
disp('5. Compute Optical flow and its derivatives')
disp('We have provided IDs of 50 Motion Salient Proposal in folder "BBX_MotionSalient" for "Spatio-Temporal Object Detection Proposals, ECCV, 2014" (in folder"Dan_proposals")');
% Calculate Optical flow and its derivative ( Forbnious Norm) and after NMS
% keep Top 50 proposals as described in equation 7 of the paper



% We assume videos of all actions are in the same folder.
% Video_ID is index of video in that folder. You can change the file path
% convention if you want.



%% 6. Compute CNN features from every key frames of Video proposals.
disp('6. Compute CNN features within Bounding Boxes of Video proposals ....') 

for Video_ID=1:2

Compute_CNN_BBX_Video(Video_ID,Videos_frame_Path,Video_Proposal_Path,Video_Proposals_CNN_Path,Motion_Salient_Proposals,MatConvNet_Path)

end

%% 7. Reconstruct Video Proposals using Images Proposals.
for Video_ID=1:2
 VideoProposalsReconstruction(Video_ID,Video_Proposals_CNN_Path,Coloc_BBX_CNN_Path,Video_Proposals_Recons_Path,L1_General_Path)
% 
 end
%% Draw Top Ranked Video Proposals.


for Video_ID=1:2  
 
    close all
     DrawTopRankedProposals(Video_ID,Video_Proposals_Recons_Path,Video_Proposal_Path,Motion_Salient_Proposals,GT_Path,Visulization_Path)
end

