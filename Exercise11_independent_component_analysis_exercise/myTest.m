function [cost, grad] = myTest(w,x)
e = 0.00001;
l = length(w);
w = reshape(w,2,l/2);
cost = sum(sum(sqrt((w*x).^2+e)));
grad = zeros(size(w));
grad(1,:) = w(1,:)*x(:,1)*x(:,1)'/sqrt((w(1,:)*x(:,1))^2+e)+ w(1,:)*x(:,2)*x(:,2)'/sqrt((w(1,:)*x(:,2))^2+e);
grad(2,:) = w(2,:)*x(:,1)*x(:,1)'/sqrt((w(2,:)*x(:,1))^2+e)+ w(2,:)*x(:,2)*x(:,2)'/sqrt((w(2,:)*x(:,2))^2+e);
grad = grad(:);
end