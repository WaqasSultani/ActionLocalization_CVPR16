function RandomWalk_NoisyImages(Image_Dataset_Path,Image_CNNPath,Noisy_Image_Dataset_Path)
%AL: This code perform randowm walk over the CNN features of images, we
%remove images which are less consistant with rest of images.

AllActions=dir(Image_CNNPath);
AllActions=AllActions(3:end);

AllActions_Image=dir(Image_Dataset_Path);
AllActions_Image=AllActions_Image(3:end);

for iaction=1:length(AllActions)
    
    
    fprintf('Action (%d/%d) done\n', iaction, numel(AllActions));
    
    
    ActionPath=[Image_CNNPath,'/',AllActions(iaction).name];
    ActionImagePath=[Image_Dataset_Path,'/',AllActions_Image(iaction).name];
    NoisyImagePath=[Noisy_Image_Dataset_Path,'/',AllActions_Image(iaction).name];
    
    if ~exist(NoisyImagePath,'dir')
        
   
        mkdir(NoisyImagePath);
    
    
    end
    
    
    
    
    
    All_files=dir(ActionPath);
    All_files=All_files(3:end);
    
    
    
    All_features=zeros(length(All_files),4096);
    for im=1:length(All_files)
    
        Pfilename=[ActionPath,'/',All_files(im).name];
    
        load(Pfilename)
        All_features(im,:)=res/sum(res);

    
    end
    
    %% You can change the parameters of random walk, number of Iteration and may get better results.
    
    dist_Mat=distance(All_features',All_features');
    sim_mat=exp(-100*dist_Mat);
    
    for col=1:size(sim_mat,2)
    
        sim_mat(:,col)=sim_mat(:,col)/sum(sim_mat(:,col));
    
    end
    
    init_scores=ones(size(sim_mat,2),1)/size(sim_mat,2);
    
    relevance_scores=init_scores;
    alpha=0.99;
    
    for it=1:1000
       %curr_score=alpha*Transition_Matrix* prev_score+(1-alpha)*prev_assign;
        New_relevance_scores=alpha*(sim_mat*relevance_scores)+(1-alpha)*init_scores;
        relevance_scores=New_relevance_scores;
    end
    
    
    
    
    
       all_dist=sum(dist_Mat);
       [val,idx]=sort(all_dist);
       [val,idx]=sort(relevance_scores,'descend');
       
    
       % We remove 30% of images. You can change this parameter. 
       per_remove_image=0.3;
       ss=length(idx);
       ss_idx=ss-round(ss*per_remove_image);
      
       
       
       
       idx=idx(ss_idx:end);
       ss=length(idx);
    
    
       for ii=1:length(idx)
       
           
           
        Im_Path=[ActionImagePath,'/',All_files(idx(ii)).name(1:end-4),'.jpg'];
        N_Path =[NoisyImagePath,'/',All_files(idx(ii)).name(1:end-4),'.jpg'];
        
        if ~exist(N_Path,'file')
        movefile(Im_Path,N_Path);
        end
       % subplot(round(ss/10),10,ii); imshow(I)
       
       
    
       end
    
    
    
    
        
end