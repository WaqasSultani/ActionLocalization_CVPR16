function  DrawTopRankedProposals(Video_ID,Video_Proposals_Recons_Path,Video_Proposal_Path,Motion_Salient_Proposals,GT_Path,Visulization_Path)

% AL

if ~exist(Visulization_Path,'dir')
    mkdir(Visulization_Path);
end


     All_Files_BBX=dir(Video_Proposal_Path);
     All_Files_BBX=All_Files_BBX(3:end);
 
    for ifile=Video_ID
        %:length(All_Files_BBX)
     
       FilePath_BBX=[Video_Proposal_Path,'/',All_Files_BBX(ifile).name];
       FilePath_BBX_Mo=[Motion_Salient_Proposals,'/',All_Files_BBX(ifile).name];
       FilePath_GT=[GT_Path,'/',All_Files_BBX(ifile).name(1:end-4)];
       Recons_file=[Video_Proposals_Recons_Path,'/',All_Files_BBX(ifile).name];
       load(FilePath_BBX_Mo);
       load(FilePath_BBX)
       load(Recons_file)
       
        
       All_images=dir([FilePath_GT,'/*.jpg']);
        if length(All_images)==0
          All_images=dir([FilePath_GT,'/*.ppm']);
        end
        
        GT1=zeros(length(All_images),4);
        Image_Path=[FilePath_GT,'/',All_images(10).name];
        I=imread(Image_Path);
       
            
        All_gt_files1=dir([FilePath_GT,'/gt']);     
        All_gt_files1=All_gt_files1(3:end);
        
        if length(All_gt_files1)~=0
       
           for im=1:1:length(All_images)
            
                gt_path=[FilePath_GT,'/gt/',All_gt_files1(im).name];

                fid=fopen(gt_path); 
                A=[];
                for i=1:4
                    A(i)=str2num(fscanf(fid,'%s/n'));
                end
    
                col_factor=size(I,2)/320;
                row_factor=size(I,1)/240;
                B(1)=A(1)/col_factor;
                B(2)=A(2)/row_factor;
                B(3)=A(3)/col_factor;
                B(4)=A(4)/row_factor;
                B=round(B);
                B(B==0)=1;
                GT1(im,:)=B;
                fclose(fid);
            
            end
       end
       
       clear B
       

            
             nBBX=1;%length(Final_idx1);
            
             Similarity_Measured=zeros(nBBX,1);
     
             
              [xx,yy]=sort(Prop_dist);
              GT1=zero2one(GT1);
    
                 Top_Ranked_Index=Final_idx1(yy(1));
                 
                 it=0;
                 for itub=Top_Ranked_Index
                 
                    
                     A1=round(BBX(:,:,Top_Ranked_Index));
                     S1=zeros(length(BBX(:,:,Top_Ranked_Index)),1);
                     [BBX_Frames,~]=find(A1(:,1));

                     D=diff(BBX_Frames);
                     f=find(D>5);

                     min_Tublet_Frames=BBX_Frames(1);
                     max_Tublet_Frames=BBX_Frames(end);
            
            
                     if ~isempty(f)
                          max_Tublet_Frames=BBX_Frames(f);  
                     end

                     TubletLength=max_Tublet_Frames- min_Tublet_Frames+1;
            
            
                    Max_Frames=length(BBX(:,:,Top_Ranked_Index));
                    Used_Frames=min(length(GT1), length(BBX(:,:,Top_Ranked_Index)));

                    start_frame=1;

               for ii=start_frame:1:Used_Frames
                 
                       A=A1(ii,:);
                       B=GT1(ii,:);
                       intersectionArea=rectint(A,B);
                       unionArea=A(3)*A(4)+B(3)*B(4)-intersectionArea;
                       overlapArea=intersectionArea/unionArea;
                       S1(ii)=overlapArea;
                                         
               end
               
               Max_A1GT=max(TubletLength,size(GT1,1));
               
               Over=sum(S1)/length(start_frame:Max_A1GT);
               it=it+1;
               Similarity_Measured(it)=Over;
            
            end
            
         end
              
          
          
         [Measured_val,Manual_idx]=sort(Similarity_Measured,'descend');
          
          MABO(ifile)=Measured_val(1);

%% Draw BBOXES

          Top_Ranked_Index=Final_idx1(yy(1));
       mm=0;
       for itub=1
            
             
              A1=round(BBX(:,:,Top_Ranked_Index));
              [BBX_Frames,~]=find(A1(:,1));
              
              Used_Frames=min(length(GT1), length(BBX(:,:,(Top_Ranked_Index))));
            
               for ii=1:2:Used_Frames  
                     
                     Image_Path=[  FilePath_GT,'/',All_images(ii).name];
                     I=imresize(imread(Image_Path),[240,320]);
                     imshow(I);
                     
                     
                     rectangle('Position',[A1(ii,1),A1(ii,2),A1(ii,3),A1(ii,4)],'EdgeColor','g','LineWidth',8)
                     rectangle('Position',[GT1(ii,1),GT1(ii,2),GT1(ii,3),GT1(ii,4)],'EdgeColor','r','LineWidth',8)
                     
                     pause(0.1)
               end
               
        end


end