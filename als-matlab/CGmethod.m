%function [y,iter,residhist,Anorm] = CGmethod(A,b,x,tol,sol)
function y = CGmethod(A,b,x)
r = b-A*x;
p = r;
iter = 0;

while iter < length(A)%max(abs(r)) > tol %iter < length(A) %iter < 50000
    al = (r'*r)/(p'*A*p);
    x = x+al*p;
    u = r;
    r = r-al*A*p;
    beta = (r'*r)/(u'*u);
    p = r + beta*p;
    iter = iter +1;
end
y = x;