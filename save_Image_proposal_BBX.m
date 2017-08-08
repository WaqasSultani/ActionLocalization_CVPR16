function save_Image_proposal_BBX(Coloc_Result_Path,Coloc_Data_Path,Coloc_BBX_Path)
%AL: This code save bounding Boxes obtained within each image.

if ~exist(Coloc_BBX_Path,'dir')
    mkdir(Coloc_BBX_Path)
end
set_path;
addpath(genpath('./tools/'));
All_Actions=dir(Coloc_Result_Path);
All_Actions=All_Actions(3:end);

for iAction=1:length(All_Actions)
    
    Coloc_Res=[Coloc_Result_Path,'/',All_Actions(iAction).name];
    Data_Path=[Coloc_Data_Path,'/',All_Actions(iAction).name];
    Coloc_BBX=[Coloc_BBX_Path,'/',All_Actions(iAction).name];
    
    if exist([Coloc_BBX,'.mat'],'file')
        
        continue;
    end
    
    
    images=dir([Data_Path,'/*_gist.mat']);


    BBX_proposal=struct;
    m=0
    nimage=length(images);

for im = 1:nimage
  
	fprintf('Images: %d / %d\n', im, nimage);
    conf.postfix_feat='_seg';
    conf.postfix_gist='_gist';
	idata   = loadView_seg([Data_Path,'/',images(im).name(1:end-5)], 'conf', conf);
	boxes  = frame2box(idata.frame)';
	
    load(fullfile( Coloc_Res,  sprintf('sai_%03s_i%02d.mat',images(im).name(1:end-9),5)))
    [ ranki, saliv ] = select_kbestbox(boxes', saliency, 5);
    if length(saliv)<5
       m=m+1 
    end
     bbox=zeros(length(saliv),4);
   
   
    for i=1:length(saliv)
     
        A1 = boxes(ranki(i), :);
        % Misu Draw box = [xmin, ymin, xmax, ymax]
        
         bbox(i,:)=[A1(1), A1(2), A1(3)-A1(1),A1(4)-A1(2)];
    end
    
    Minsu_box=bbox;
    
    clear bbox

      BBX_proposal(im).BBX=Minsu_box;
      clear Misnu_box
    
    
end

save(Coloc_BBX,'BBX_proposal');

clear BBX_proposal


end





