function S = randsparse(m,n)
S = randi(15,m,n);
S(S>10) = 0;
S = sparse(S);
end
