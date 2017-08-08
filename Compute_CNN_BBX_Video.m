function  Compute_CNN_BBX_Video(Video_ID,Videos_frame_Path,Video_Proposal_Path,Video_Proposals_CNN_Path,Motion_Salient_Proposals,MatConvNet_Path)
%AL: Compute CNN features within Bounding Boxes of Video proposals.

  if ~exist(Video_Proposals_CNN_Path,'dir')

     mkdir(Video_Proposals_CNN_Path)
 
 end
 

 
AllFiles=dir(Videos_frame_Path);
AllFiles=AllFiles(3:end);
  
for ifile=Video_ID

     
     fprintf('Extracting CNN feats in video proposals (%d/%d) done\n', ifile, numel(AllFiles));
           FramesFilePath=[Videos_frame_Path,'/', AllFiles(ifile).name];
           BBX_Salient_Indexfile=[Motion_Salient_Proposals,'/', AllFiles(ifile).name];
           ResultFilePath=[Video_Proposals_CNN_Path,'/', AllFiles(ifile).name];
           ProposalFilePath=[Video_Proposal_Path,'/', AllFiles(ifile).name];

           if exist( [ResultFilePath,'.mat'],'file')

               continue;

           else
           
          %   waqas=1;
           %  save( ResultFilePath,'waqas');  
               
               
           end
           
           
           
           
           
           
           curr_path=pwd;
cd(MatConvNet_Path)
run matlab/vl_setupnn
net = load('imagenet-vgg-verydeep-16.mat') ;
cd(curr_path)
           
           
           
           
           
      % Load Salient Proposal_Index
      
       load(BBX_Salient_Indexfile)
      
      % Load All proposals
      clear BBX
      load(ProposalFilePath)    
           
  
        % Load Video
        
        AllFrames=dir([FramesFilePath,'/*.jpg']);
        if length(AllFrames)==0
          AllFrames=dir([FramesFilePath,'/*.ppm']);
        end
        
        
        Video=zeros(240,320,3,length(AllFrames));
        
        for iv=1:length(AllFrames)
            
           Video(:,:,:,iv)=imresize(imread([FramesFilePath,'/',AllFrames(iv).name]),[240, 320]);
            
        end
        
        load(BBX_Salient_Indexfile)
        Proposal=struct;
        
%         if length(Final_idx1)~=50
%             
%            error('?????') 
%         end
            
        for ibbx=1:length(Final_idx1)
            
            ibbx
          
            
            idx=Final_idx1(ibbx);
           
            BBX1= BBX(:,:,idx);
            BBX_CNN=zeros(length(1:10:size(BBX1,1)),4096);
            gg=0;
            for y=1:10:size(BBX1,1) 

                BBX_loc=BBX1(y,:);
                c1=BBX_loc(1);
                c2=min(320,BBX_loc(1)+BBX_loc(3));
                r1=BBX_loc(2);
           
      r2=min(240,BBX_loc(2)+BBX_loc(4));
                
                c2=max(c2,10);
                r2=max(r2,10);
                
                BBX_Image=Video(r1:r2,c1:c2,:,y);
   %             figure(1), imshow(uint8(BBX_Image));
                im_ = single(BBX_Image) ; % note: 255 range
                im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
                
                im_ = im_ - net.normalization.averageImage ;
                
               
                res = vl_simplenn(net, im_) ;
                gg=gg+1;
                BBX_CNN(gg,:)=reshape(res(36).x,[1 4096]);
                clear res
            
            end
            
           BBX_CNN=BBX_CNN(1:gg,:);
           Proposal(ibbx).CNN=single(BBX_CNN);
           Proposal(ibbx).BBX=BBX1;
            
        end
        save( ResultFilePath,'Proposal');
 end