clear all
close all

m = 132;
n = 103 ;
S = randsparse(m,n);
f = 10;
X = randi(5,m,f);
T = randi(5,f,n);
tol = 10E-6;
error = 1;
L = 0.5;
iter = 0;
error_before = 0;

while abs(error-error_before) > tol
    error_before = error;
    [AX, BX] = getAB(S, X, T, L);
    j = 1;
    while j < m+1
        x = rand(f,1);
        X1(j,:) = CGmethod(AX{j,1},BX{j,1},x)';
        j = j + 1;
    end
    X = X1;
    [AT,BT] = getAB(S', T', X', L);
    j = 1;
    while j < n+1
        t = rand(f,1);
        T1(:,j) = CGmethod(AT{j,1},BT{j,1},t);
        j = j + 1;
    end
    T = T1;
    E = S-X*T;
    error = norm(E(S>0))
    iter = iter + 1;
end
iter

    
