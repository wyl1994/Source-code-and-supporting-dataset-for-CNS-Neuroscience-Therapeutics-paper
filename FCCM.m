


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%这里用全局权重系数的方案，即全收敛交叉映射


function [causality_x_y_corr] = FCCM(x, y, E,tau,delay)   
    L = length(x);
    if L ~= length(y)
        error('series should have same size')
    end
    M_x = shadow(x, tau, E);
    M_y = shadow(y, tau, E);
    M_x = M_x(1:end-delay,:);
    M_y= M_y(1+delay:end,:);
    t0 = 1+(E-1)*tau+delay; 
    x_r = zeros(1, L-t0+1);
    y_r = zeros(1, L-t0+1);
    x_o = x(1:L-t0+1);
    y_o = y(t0:L);
    
    for t=1:L-t0+1   
        x_r(t) = reconstruct(t, x_o, M_y);    
    end
    causality_x_y_corr=corr(x_o',x_r');
end


function M = shadow(v, tau, E)  
    t0 = 1+(E-1)*tau;
    L = length(v);
    M = zeros(L-t0+1, floor(E/tau)); 
    
    for t = t0:L
        M(t-t0+1, :) = v(t-(E-1)*tau:tau:t);
    end  
end



function r = reconstruct(t, v, M)
    [d, n] = sort(sqrt(sum((M-repmat(M(t,:), [size(M,1) 1])).^2, 2)));  
    E = size(M,2);
    d(d==0) = [];
    n=n((length(n)-length(d)+1):end);
    n = n(2:end);
    d = d(2:end);
    u = exp(-d./d(1))';
    w = u ./ sum(u);
    r = sum(w .* v(n));
end


