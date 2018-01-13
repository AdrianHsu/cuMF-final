%input:  R:m-by-n
%        X:m-by-f
%        theta:f-by-n
function [A,B] = getAB(R, X, theta, lambda)
n = length((theta));
m = length(X);
A = cell(m,1);
B = cell(m,1);

j = 1;
while j < m+1
    Ru = R(j,:);
    nonzero = nnz(Ru);
    [row,column] = find(Ru > 0,nonzero);
    jj = 1;
    while jj < nonzero+1
        if jj == 1
            A{j,1} = theta(:,column(jj))*theta(:,column(jj))';
        else
            A{j,1} = A{j,1} + theta(:,column(jj))*theta(:,column(jj))';
        end
        A{j,1} = A{j,1} + lambda*eye(length(A{j,1}));
        jj = jj + 1;
    end
    j = j + 1;
end 
j = 1;
while j < m+1
    B{j,1} = theta * R(j,:)';
    j = j + 1;
end
