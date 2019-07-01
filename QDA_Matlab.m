clear; close all force; clc;
rng(1);
%% Construção da base
% BASE 1
% load fisheriris;
% data1 = [meas(1:50, 3).'; meas(1:50, 4).'];
% data2 = [meas(51:100, 3).'; meas(51:100, 4).'];
% data3 = [meas(101:end, 3).'; meas(101:end, 4).'];

% BASE 2
DP = sqrt(0.2); 
x1 = [14; 10] * ones(1, 200) + [0, 2; 1, 1] * randn(2, 200) * DP;
x2 = [12; 12] * ones(1, 200) + [0, 4; 1, 1] * randn(2, 200) * DP;
x3 = [17; 12] * ones(1, 200) + [0, -4; -1, 1] * randn(2, 200) * DP;
x4 = [10; 15] * ones(1, 250) + [1, 4; 1, 1] * randn(2, 250);
x5 = [20; 15] * ones(1, 250) + [1, -4; -1, 1] * randn(2, 250);
data1 = [x1];
data2 = [x2, x3];
data3 = [x4, x5];

% BASE 3
% t1 = randn(1, 100);
% t2 = 2 * (randn(1, 100) + 5);
% t3 = 2 * (randn(1, 100) + 10);
% r1 = 2 * pi * rand(1, 100);
% r2 = 2 * pi * rand(1, 100);
% r3 = 2 * pi * rand(1, 100);
% data1 = [t1 .* cos(r1); t1 .* sin(r1)];
% data2 = [t2 .* cos(r2); t2 .* sin(r2)];
% data3 = [t3 .* cos(r3); t3 .* sin(r3)];

%% Divisão da base
X = [data1, data2, data3];
C = [ones(1, size(data1, 2)), 2*ones(1, size(data2, 2)), 3*ones(1, size(data3, 2))];

mx = floor(min(min(X(1, :))));
Mx = ceil(max(max(X(1, :))));
my = floor(min(min(X(2, :))));
My = ceil(max(max(X(2, :))));

figure;
h1 = gscatter(X(1, :), X(2, :), C, 'rgb', 'vdo', [], 'off');
hold on
%% Treinamento e Teste
MdlQuadratic = fitcdiscr(X.', C, 'DiscrimType', 'quadratic');

% Classe 1 vs 2
MdlQuadratic.ClassNames([1 2]);

K = MdlQuadratic.Coeffs(1, 2).Const;
L = MdlQuadratic.Coeffs(1, 2).Linear; 
Q = MdlQuadratic.Coeffs(1, 2).Quadratic;

f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
    (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h2 = fimplicit(f,[mx Mx my My]);
h2.Color = 'k';
h2.LineWidth = 2;

% Classe 2 vs 3
MdlQuadratic.ClassNames([2 3]);

K = MdlQuadratic.Coeffs(2, 3).Const;
L = MdlQuadratic.Coeffs(2, 3).Linear; 
Q = MdlQuadratic.Coeffs(2, 3).Quadratic;

f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
    (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h2 = fimplicit(f,[mx Mx my My]);
h2.Color = 'm';
h2.LineWidth = 2;
