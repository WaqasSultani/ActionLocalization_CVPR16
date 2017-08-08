function Compute_CNN_Images(Image_Dataset_Path,ImageCNNPath,MatConvNet_Path)
% AL: This code compute VGG-CNN features from all the images in the folder.
% AL: You can download the images from http://crcv.ucf.edu/projects/videolocalization_images//

if ~exist(ImageCNNPath,'dir')
    mkdir(ImageCNNPath)
end

curr_path=pwd;
cd(MatConvNet_Path)
run matlab/vl_setupnn
net = load('imagenet-vgg-verydeep-16.mat') ;

cd(curr_path)
AllAction=dir(Image_Dataset_Path);
AllAction=AllAction(3:end);

if length(AllAction)==0
   disp('No image found')
   return
    
end


for iAction=1:length(AllAction)
 
    fprintf('Extracting CNN feats in Action Images (%d/%d) done\n', iAction, numel(AllAction));
    ActionPath=[Image_Dataset_Path,'/',AllAction(iAction).name];
    ActionResultPath=[ImageCNNPath,'/',AllAction(iAction).name];
    
    if ~exist(ActionResultPath,'dir')
        
        mkdir(ActionResultPath)
        
    else
       
        continue;
        
    end
    
Allimages=dir([ActionPath,'/*.jpg']);
% Allimages=Allimages(3:end);

for iIm=1:length(Allimages)
    
   iIm
    Im_Path=[ActionPath,'/',Allimages(iIm).name];
    Re_Path=[ActionResultPath,'/',Allimages(iIm).name(1:end-4),'.mat'];
    
    if exist(Re_Path,'file')
       
        continue;
    
    end
        


im = imread(Im_Path) ;
im_ = single(im) ; % note: 255 range

if size(im,3)==1
% If downloaded images in Not RGB, delete it
    delete(Im_Path);    
    continue;
  

end


im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = im_ - net.normalization.averageImage ;

% run the CNN
 res = vl_simplenn(net, im_) ;
 res=reshape(res(36).x,[1 4096]);
save(Re_Path,'res');


end
end