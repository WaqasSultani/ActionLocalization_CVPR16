function Compute_CNN_Image_BBX(Image_Dataset_Path,Coloc_BBX_Path,Coloc_BBX_CNN_Path,MatConvNet_Path)
 
%AL: This code is to compute CNN features within Bounding Boxes in each image 
curr_path=pwd;
cd(MatConvNet_Path)
run matlab/vl_setupnn
net = load('imagenet-vgg-verydeep-16.mat') ;
cd(curr_path)
 
%Coloc_BBX_Path='/media/waqas/Data/Action_Recognition/Rebuttal_CVPR2016/UCF_Sports/Google_Images_BBX';
%Coloc_BBX_CNN_Path='/media/waqas/Data/Action_Recognition/Rebuttal_CVPR2016/UCF_Sports/Google_Images_CNN_BBX';

if ~exist(Coloc_BBX_CNN_Path,'dir')
    mkdir(Coloc_BBX_CNN_Path)
end


All_Actions=dir(Image_Dataset_Path);
All_Actions=All_Actions(3:end);

for iAction=length(All_Actions):-1:1
    
    ImagePath=[Image_Dataset_Path,'/',All_Actions(iAction).name];
    ImageProposal=[Coloc_BBX_Path,'/',All_Actions(iAction).name];
   
    
    images=dir([ImagePath,'/*.jpg']);
    
    
    if exist([Coloc_BBX_CNN_Path,'/',All_Actions(iAction).name],'dir')
       
       % continue;
        
    else
        mkdir([Coloc_BBX_CNN_Path,'/',All_Actions(iAction).name])
    end
     load(ImageProposal);
for im =1:numel(images)
  
    fprintf('Extracting CNN feats (%d/%d) done\n', im, numel(images));
    
    if exist([Coloc_BBX_CNN_Path,'/',All_Actions(iAction).name,'/',images(im).name(1:end-4),'.mat'],'file')
        
        continue
    end
        
    
    
    img_org=imread([ImagePath,'/',images(im).name]);
    Image_BBX_CNN=zeros(size(BBX_proposal(im).BBX,1),4096);
    for ibbx=1:size(BBX_proposal(im).BBX,1)
        
      bbox=BBX_proposal(im).BBX(ibbx,:);
      BBX_Image=img_org(max(bbox(2),1):min(bbox(4)+bbox(2),size(img_org,1)),max(bbox(1),1):min(bbox(3)+bbox(1),size(img_org,2)),:);
      im_ = single(BBX_Image) ; % note: 255 range
      im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
      im_ = im_ - net.normalization.averageImage ;
      res = vl_simplenn(net, im_) ;
      Image_BBX_CNN(ibbx,:)=reshape(res(36).x,[1 4096]);
       
    end
  
    save([Coloc_BBX_CNN_Path,'/',All_Actions(iAction).name,'/',images(im).name(1:end-4)],'Image_BBX_CNN')
    
    clear Image_BBX_CNN
    
end


end


