
function M = MassAssembler1D(x)
    n = length(x)-1; % number of subintervals
    M = zeros(n+1,n+1); % allocate mass matrix
    for i = 1:n % loop over subintervals
        h = x(i+1) - x(i); % interval length
        M(i,i) = M(i,i) + h/3; % add h/3 to M(i,i)
        M(i,i+1) = M(i,i+1) + h/6;
        M(i+1,i) = M(i+1,i) + h/6;
        M(i+1,i+1) = M(i+1,i+1) + h/3;
    end
end

function b = LoadAssembler1D(x,f)
    n = length(x)-1;
    b = zeros(n+1,1);
    for i = 1:n
        h = x(i+1) - x(i);
        b(i) = b(i) + f(x(i))*h/2;
        b(i+1) = b(i+1) + f(x(i+1))*h/2;
    end
end

m = MassAssembler1D([1,2,3,4])
