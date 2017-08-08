% script for extracting segmentation proposals and their features
%
% written by Minsu Cho, modified by Suha Kwak

function extract_boxes(name_experiment,Image_Data_Path,Coloc_Dataset_Path)

set_path;
evalc(['setup_', name_experiment]);

% suha: parameters to filter out small proposals
%size_thres_ratio = 20;

% initialize structs
feat = struct;
boxes = struct;

% setup params for randomized prim
%'config/rp_4segs.mat' to sample from 4 segmentations (slower but higher recall)
%'config/rp.mat' to sample from 1 segmentations (faster but lower recall)
params_rp = LoadConfigFile(fullfile(rp_root,'config/rp_4segs.mat'));

%AllAction_Path='/media/waqas/Data/Action_Recognition/Rebuttal_CVPR2016/UCF_Sports/Google_Images';
%AllResultPath='/media/waqas/Data/Action_Recognition/Rebuttal_CVPR2016/UCF_Sports/Google_Images';
AllAction=dir(Image_Data_Path);
AllAction=AllAction(3:end);


if ~exist(Coloc_Dataset_Path,'dir')
    
    mkdir(Coloc_Dataset_Path)
end



% ImagePath='/media/waqas/Data/Action_Recognition/UCF_Sport_DATA_July/Google_Images/Walking_Google';
% images=dir([ImagePath,'/*.jpg']);

for iAction=1:length(AllAction)
    
    
  ImagePath=[Image_Data_Path,'/',AllAction(iAction).name];
  Result_Image_Path=[Coloc_Dataset_Path,'/',AllAction(iAction).name];
 
    if ~exist(Result_Image_Path,'dir')
      
        mkdir(Result_Image_Path)
        
    end
% ImagePath='/media/waqas/Data/Action_Recognition/UCF_Sport_DATA_July/Google_Images/Walking_Google';
images=dir([ImagePath,'/*.jpg']);



% loop through images
for i = 1: numel(images)
    %:-1:1
    %:-1:1
   % for i = 200:-1:1
    if exist('modk') && exist('modv') && mod(i, modv) ~= modk
        continue;
    end
    %path_img = images{i};
    path_img=[ImagePath,'/',images(i).name];
    
    % box-proposals and their HOG descriptors
    path_feat = [Result_Image_Path,'/',images(i).name(1:end-4), conf.postfix_feat, '.mat'];
    if exist(path_feat, 'file') && ~overwrite
        fprintf('Extracting boxes & descriptors (%d/%d): %s done\n', i, numel(images), path_img);
        continue;
    end

    % suha: images should have 3 channels 
    %       for running "extract_segfeat_hog" without problem.
    img_org = imread(path_img);
    %%
    %img_org=imresize(img_org,[240 320]);
    
    
    %%
    
    if ndims(img_org) < 3
        img_col = cat(3, img_org, img_org, img_org);
    elseif size(img_org, 3) == 1
        img_col = cat(3, img_org(:, :, 1), img_org(:, :, 1), img_org(:, :, 1));
    else
        img_col = img_org;
    end
    img = standarizeImage(img_col);
    fprintf('Extracting boxes & descriptors (%d/%d): %s ', i, numel(images), path_img);

    % compute segment proposals for the given image
    seg.coords = RP(img, params_rp); %[xmin, ymin, xmax, ymax]
    seg.coords = [seg.coords; 1, 1, size(img,2), size(img,1)]; % add a whole box

    % suha: filltering too small proposals out
    %size_thres = min(size(img, 1), size(img, 2)) / size_thres_ratio;
    %seg_WH = [seg.coords(:, 3) - seg.coords(:, 1), seg.coords(:, 4) - seg.coords(:, 2)];
    %list_candidate = find(min(seg_WH, [], 2) > size_thres);
    %seg.coords = seg.coords(list_candidate, :);
    
    % compute features for proposals
    ticId = tic;
    feat = extract_segfeat_hog(img,seg);
    fprintf('took %f secs.\n',toc(ticId));

    % save feats for the given image
    save(path_feat, 'feat');

    clear img_org;
    clear img_col;
    clear seg;
    clear img;
    clear feat;
end
end
