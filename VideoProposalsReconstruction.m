
function  VideoProposalsReconstruction(Video_ID,Video_Proposals_CNN_Path,Coloc_BBX_CNN_Path,Video_Proposals_Recons_Path,L1_General_Path)
addpath(genpath(L1_General_Path))
% AL
if ~exist(Video_Proposals_Recons_Path,'dir')
   
    mkdir(Video_Proposals_Recons_Path);
    
end

%% You need to provide image features corrosponding the video action. Please change the path in following code accordingly.
% We have used below "Video_ID==1  Image_iaction='Diving_Side'" and "Video_ID==2  Image_iaction='Kicking'" because two videos in  
% 'Ground_Truth_Video' folder are of diving and kicking action.
% respectively.

     if Video_ID==1
        Image_iaction='Diving_Side';
     elseif Video_ID==2
        Image_iaction='Kicking';    
       
     end

    
   ImageProposals=[Coloc_BBX_CNN_Path,'/',Image_iaction];
  
	if ~exist(Video_Proposals_Recons_Path,'dir')
   
	   	 mkdir(Video_Proposals_Recons_Path);
    
    end


    All_Videos=dir(Video_Proposals_CNN_Path);
    All_Videos=All_Videos(3:end);

     ResultVideoPath=[Video_Proposals_Recons_Path,'/',All_Videos(Video_ID-1).name];
     
     fprintf('Computing Reconstruction Error for  video (%d/%d) done\n',Video_ID, numel(All_Videos));
    
      if exist(ResultVideoPath,'file')
%               try
%                   load(ResultVideoPath)
%                   waqas
             % catch
               return    
              %end
     else
         
         waqas=1;
         save(ResultVideoPath,'waqas');
          
      end
    
    
    
    
    % Load Images
    All_images=dir(ImageProposals);
    All_images=All_images(3:end);
    Image_CNN=[];
    
    for im=1:length(All_images)
  im
       ImagefilePath=[ImageProposals,'/',All_images(im).name]; 
       load(ImagefilePath)
       % We used top two image propoposals. More can also be used.
       Image_CNN=[Image_CNN;Image_BBX_CNN(1:min(size(Image_BBX_CNN,1),2),:)];
    end
    
   
    Image_CNN = Image_CNN ./ (repmat(sqrt(sum(Image_CNN.^2,2)), 1, size(Image_CNN,2)) + eps);
    
    
 
    
    for iv=Video_ID

        
        
      VideofilePath=[Video_Proposals_CNN_Path,'/',All_Videos(iv-1).name]; 
      load(VideofilePath)
        

      
       mu=2^-4;
       lamda_1=0.1;
    
       Prop_dist=zeros(1,length(Proposal));
       Prop_len=zeros(1,length(Proposal));
     
tic
      for ip=1:length(Proposal)
          ip
        
          Proposal_CNN=Proposal(ip).CNN;
          Proposal_CNN = Proposal_CNN ./ (repmat(sqrt(sum(Proposal_CNN.^2,2)), 1, size(Proposal_CNN,2)) + eps);
          Proposal_CNN=double(Proposal_CNN);
          
          D=Image_CNN';
          X=Proposal_CNN';
          y=D;
          
          funObj = @(w)DoveError1(w,X,y,mu);
          nVars=size(X,2)*size(D,2);
          w_init= 0.1*rand(nVars,1);

           % Initial value for iterative optimizer
           
          lambda = lamda_1*ones(nVars,1);
           
          fprintf('\nComputing LASSO Coefficients...\n');

          wLASSO = L1General2_PSSgb(funObj,w_init,lambda);

          W=reshape(wLASSO,[size(D,2),size(X,2)]);
          fprintf('Number of non-zero variables in LASSO solution: %d\n',nnz(wLASSO));
         
          Prop_dist(ip)=norm(X-(D*W),'fro');
          Prop_len(ip)=size(Proposal_CNN,1);
     
          clear Proposal_CNN  W X D wLASSO
      
      end
      toc

        save(ResultVideoPath,'Prop_dist','Prop_len');
        
        clear Prop_dist
      

    end
