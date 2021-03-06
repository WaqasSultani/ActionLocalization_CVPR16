function [f,g,H,T] = DoveError1(W,X,D,mu)

    W=reshape(W,[size(D,2),size(X,2)]);
    %mu=2^-4;
    
    res1 = (X-D*W);
    f1 = sum(sum(res1.^2));
    res2=W-repmat(mean(W,2),1,size(W,2));
    f2=sum(sum(res2.^2));
    f3=sum(abs(W(:)));
    f= f1 + mu*f2 + mu*f3;
    
       
        One_Mat = ones(size(X,2),size(X,2));
        
 
        g=-2*(D'*X)+2*(D'*D)*W+  mu*eye(size(D,2))*W -mu*(1/size(X,2)*W*One_Mat);
        
