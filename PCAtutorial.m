%SVD/PCA psuedo tutorial
clear; close all;
set(0, 'defaultlinelinewidth', 2, 'defaultaxeslinewidth', 2, 'defaultaxesfontsize', 14);

sample_size = 501;

% Make some straight line data
theta0 = 1;
theta1 = 2;
% alternatively write
theta = zeros(2, 1); %similar to fortran, indexing starts at 1 (unlike Fortran can't change it)
theta(1) = theta0;
theta(2) = theta1;
x = linspace(0, 4, sample_size);
x = transpose(x); %make a column vector, matlab default is row vector(?) that's confusing given it is a column major order language
y = theta0 + theta1*x;
y = y + randn(size(y))*2;
plot(x, y, '.b'); title('Noisy linear relation');

% least squares cost with L2 regularization 
% (@ just means anonymous function and makes for easy readability)
%J = @(A, th, y, lambda) (y-A*th)'*(y-A*th) + lambda/length(x)*sum(th(2:end).^2); %penalize all except theta0 as bias is needed
J = @(A, th, y, lambda) (y-A*th)'*(y-A*th) + lambda/length(x)*sum(th(3:end).^2); %if absolutely know first two theta terms are correct then regularize beyond
poly_vals = 2; %how many higher order polynomial terms to fit
initial_theta = ones(poly_vals+1, 1); %initialize theta guess for minimization
lambda_val = 1e8; % changing this higher will kill the higher theta terms because only way to make cost approach zero is to have very small higher theta terms
X_poly = repmat(x, [1, poly_vals]);
X_poly = bsxfun(@power, X_poly, 1:poly_vals);
A = [ones(length(x), 1), X_poly]; %y = col1*theta0 + [col2, .., poly_vals]*x^poly_vals 
[theta_est, cost] = fminunc(@(t) (J(A, t, y, lambda_val)), initial_theta); %minimizing t given A, y, and lambda
y_est = A*theta_est;
figure;
plot(x, y, '.b', x, y_est, '-r', 'linewidth', 2); title('Noisy linear relation fit');

% now PCA
X = [x, y];
% need to get (X-mean(X))/std(X), std(X) not a requirement but a good idea whereas mean subtraction is required by definition of cov
mean_X = mean(X);
X_norm = bsxfun(@minus, X, mean_X);
sigma_X = std(X);
X_norm = bsxfun(@rdivide, X_norm, sigma_X);
[U, S, V] = svd(X_norm); 
% V(:, 1) is the singular vector 
figure;
plot(X_norm(:, 1), X_norm(:, 2), '.b'); hold on;
plot([0, V(1, 1)], [0, V(2, 1)], 'linestyle', '-', 'color', 'r', 'linewidth', 3); hold off;
title('Scaled feature set');
% scale back to original content
PCscore = X_norm*V; %alternatively since X_norm = U*S*V^T then PCscore = U*S;
dim_keep = 1;
X_reduce = PCscore(:, 1:dim_keep)*V(:, 1:dim_keep)';
X_reduce = bsxfun(@times, X_reduce, sigma_X); 
X_reduce = bsxfun(@plus, X_reduce, mean_X);
figure;
plot(X(:, 1), X(:, 2), '*g'); hold on;
plot(X_reduce(:, 1), X_reduce(:, 2), '.b'); hold on;
xlim([0, 4]); ylim([-5, 15]);
title('Data mapped to PCA basis (note: PCA != Least Squares)'); %PCA is not least squares, in low noise they could be similar

% PCA for ellipse
% make the ellipse
a = 3;
b = 8;
x0 = 2;
y0 = 1;
span = max([a, b]);
xvals = -span-2:.1:span+2;
yvals = xvals; % make square region
[Xg, Yg] = meshgrid(xvals, yvals);
theta = 45;
Xg_rot = Xg.*cosd(theta) - Yg.*sind(theta);
Yg_rot = Xg.*sind(theta) + Yg.*cosd(theta);
ellipse = zeros(size(Xg));
ellipse_map = ((Xg_rot-x0).^2/a^2 + (Yg_rot-y0).^2/b^2) <= 1;
ellipse(ellipse_map) = 1;
figure; imagesc(Xg, Yg, ellipse);
set(gca, 'ydir', 'normal')
[ye, xe] = find(ellipse); % get the coordinates of the ellipse
X = [ye, xe];
mean_X = mean(X);
X_norm = bsxfun(@minus, X, mean_X);
sigma_X = std(X);
X_norm = bsxfun(@rdivide, X_norm, sigma_X);
[U, S, V] = svd(X_norm); 
% V(:, 1) is the singular vector 
figure;
plot(X_norm(:, 1), X_norm(:, 2), '.b'); hold on;
plot([0, V(1, 1)], [0, V(2, 1)], 'linestyle', '-', 'color', 'r', 'linewidth', 3); 
plot([0, V(1, 2)], [0, V(2, 2)], 'linestyle', '-', 'color', 'c', 'linewidth', 3); hold off;
title('Scaled feature set');
% scale back to original content
PCscore = X_norm*V; %alternatively since X_norm = U*S*V^T then PCscore = U*S;
dim_keep = 1;
X_reduce = PCscore(:, 1:dim_keep)*V(:, 1:dim_keep)';
X_reduce = bsxfun(@times, X_reduce, sigma_X); 
X_reduce = bsxfun(@plus, X_reduce, mean_X);
%figure;
%plot(X(:, 1), X(:, 2), '*g'); hold on;
%plot([mean_X(1), mean_X(1)+V(1, 1)*b/2], [mean_X(2), mean_X(2)+V(2, 1)*b/2], 'linestyle', '-', 'color', 'r', 'linewidth', 3);
%plot([mean_X(1), mean_X(1)+V(1, 2)*a/2], [mean_X(2), mean_X(2)+V(2, 2)*a/2], 'linestyle', '-', 'color', 'b', 'linewidth', 3); hold off;
