function [] = CompareCond()
clear all; close all; clc;

%fix K and take derivatives (taken from Palmer's code)
epsilon = 200;
K = @(x, point) exp(-epsilon.*(x-point).^2);
%D1K = @(x,center) ( -2.*epsilon.*(x-center).*K(x,center) );
D2K = @(x,center) ( 2.*epsilon.*(2.*epsilon.*((x-center).^2)-1).*K(x,center));

hold on;

Nstart = 1;
Nend = 100;

condPlotK = zeros(1, Nend - Nstart);
condPlotKTilda = zeros(1, Nend - Nstart);
condPlotL = zeros(1, Nend - Nstart);
condPlotLTilda = zeros(1, Nend - Nstart);
condPlotV = zeros(1, Nend - Nstart);
condPlotP = zeros(1, Nend - Nstart);

for N = Nstart : 1 : Nend
    samplePoints = linspace(0, 1, N);
    temp = repmat(samplePoints, N, 1);
    %Kernel matrix
    KMatrix = K(temp', temp);
    
    KMatrixTilda = [K(temp', temp) ones(N, 1) samplePoints';
                    K(0, samplePoints) 1 0;
                    K(1, samplePoints) 1 1;];

    %For this example, L = 2nd derivative
    LMatrix = D2K(temp', temp);
    LMatrixTilda = [D2K(temp', temp) zeros(N, 2);
               K(0, samplePoints) 1 0;
               K(1, samplePoints) 1 1];
    cond(KMatrix);
    condPlotK(N-Nstart+1) = cond(KMatrix);
    condPlotKTilda(N-Nstart+1) = cond(KMatrixTilda);
    condPlotL(N-Nstart+1) = cond(LMatrix);
    condPlotLTilda(N-Nstart+1) = cond(LMatrixTilda);
    
    %%Newton Basis
    [B, VMatrix] = calculate_beta_v(KMatrix, N, samplePoints, K);
    condPlotV(N-Nstart+1) = cond(VMatrix);
    
    %P = L * M' = L * (K^-1 * V)' = L*(K\V)'
    PMatrix = LMatrix*(KMatrix\VMatrix)'; %may cause problems
    
    condPlotP(N-Nstart+1) = cond(PMatrix);

end

    plot(Nstart:1:Nend, log(condPlotK), ':b')
    plot(Nstart:1:Nend, log(condPlotKTilda), 'b')
    plot(Nstart:1:Nend, log(condPlotL), ':g')
    plot(Nstart:1:Nend, log(condPlotLTilda), 'g')
    plot(Nstart:1:Nend, log(condPlotV), ':r')
    plot(Nstart:1:Nend, log(condPlotP), ':c')
    
    title('log of Condition Number vs. Number of points sampled');
    legend('K Matrix', 'K~ Matrix', 'L Matrix', 'L~ Matrix', 'V Matrix', 'P Matrix', 'Location', 'NorthWest');
    ylabel('log of Condition Number');
    xlabel('Number of points sampled');

end


function [B, V] = calculate_beta_v(KM, N, xs, K)
% here we alternately calculate betas and vs since they depend on each
% other (see paper for relationship). B should be lower triangular,
% and V should be unit upper triangular s.t. KM = B*V
    B = zeros(N,N);
    V = eye(N);
    for c=1:(N-1)
        for i=c:N
            B(i,c) = calculate_single_beta(B,V,i,c,K,xs);
        end
        % not sure if this is bad when KM is ill-cond.
        V(1:c,c+1) = B(1:c,1:c)\KM(1:c,c+1);
    end
    B(N,N) = calculate_single_beta(B,V,N,N,K,xs);
end

function [res] = calculate_single_beta(B, V, i, j, K, xs)
res = K(xs(j), xs(i));
    for k=1:j-1
        res = res - B(i,k).*V(k,j);
    end
end