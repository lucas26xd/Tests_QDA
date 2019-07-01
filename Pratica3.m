clear all; close all force; clc;
rng(1);

granulidade = 0.1;
KERNEL = 'polynomial'; % Kernel utilizado no SVM (rbf, linear, polynomial)
K = 10; % Números de folds no cross validation
%% Construção da base
% BASE 1
% load fisheriris;
% data1 = [meas(1:50, 3).'; meas(1:50, 4).'];
% data2 = [meas(51:100, 3).'; meas(51:100, 4).'];
% data3 = [meas(101:end, 3).'; meas(101:end, 4).'];

% BASE 2 (Principal)
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

%%  QDA e SVM (1 vs 1)
accQDA = 0;
accSVM = 0;
tmpQDA = 0;
tmpSVM = 0;

% Cross Validation - K-Fold
CVP1 = cvpartition(data1(1, :), 'KFold', K);
CVP2 = cvpartition(data2(1, :), 'KFold', K);
CVP3 = cvpartition(data3(1, :), 'KFold', K);

for i = 1:K
    fprintf("Fold %d:", i);
    % Divisão da base (Treino)
    idx = training(CVP1, i);
    B1 = data1(:, idx);
    idx = training(CVP2, i);
    B2 = data2(:, idx);
    idx = training(CVP3, i);
    B3 = data3(:, idx);
    % Divisão da base (Teste)
    idx = test(CVP1, i);
    T1 = data1(:, idx);
    idx = test(CVP2, i);
    T2 = data2(:, idx);
    idx = test(CVP3, i);
    T3 = data3(:, idx);
    
    % Treino e Teste
    % D1 vs D2
    C12 = [ones(1, size(T1, 2)), zeros(1, size(T2, 2))];
    Tst1 = [T1(1, :), T2(1, :)];
    Tst2 = [T1(2, :), T2(2, :)];
    t = tic;
    TQDA12 = QDA(B1, B2, Tst1, Tst2);
    tmpQDA = tmpQDA + toc(t);
    t = tic;
    TSVM12 = SVM(B1, B2, Tst1, Tst2, KERNEL);
    tmpSVM = tmpSVM + toc(t);
    
    % D2 vs D3
    C23 = [ones(1, size(T2, 2)), zeros(1, size(T3, 2))];
    Tst1 = [T2(1, :), T3(1, :)];
    Tst2 = [T2(2, :), T3(2, :)];
    TQDA23 = QDA(B2, B3, Tst1, Tst2);
    TSVM23 = SVM(B2, B3, Tst1, Tst2, KERNEL);
    
    % Cálculo das acurácias
    Ac12 = C12 == TQDA12;
    Ac23 = C23 == TQDA23;
    Ac = ((size(find(Ac12 == 1), 2) / size(Ac12, 2)) + (size(find(Ac23 == 1), 2) / size(Ac23, 2))) / 2;
    accQDA = accQDA + Ac;
    fprintf(" QDA = %.2f%%. ", Ac * 100);
    
    Ac12 = C12 == TSVM12;
    Ac23 = C23 == TSVM23;
    Ac = ((size(find(Ac12 == 1), 2) / size(Ac12, 2)) + (size(find(Ac23 == 1), 2) / size(Ac23, 2))) / 2;
    accSVM = accSVM + Ac;
    fprintf(" SVM = %.2f%%.\n", Ac * 100);    
    
%     P = repartition(P);
end
fprintf("Acurácia Média: QDA = %.2f%%.  SVM = %.2f%%.\n", (accQDA/K)*100, (accSVM/K)*100)
fprintf("Tempo Médio: QDA = %.2fms.  SVM = %.2fms.\n", (tmpQDA/K)*1000, (tmpSVM/K)*1000)

%% Definição da grade para testes
base = [data1, data2, data3];
mx = floor(min(base(1, :)));
Mx = ceil(max(base(1, :)));
my = floor(min(base(2, :)));
My = ceil(max(base(2, :)));
[X, Y] = meshgrid(mx:granulidade:Mx, my:granulidade:My);

%% Testando vários pontos da grid para plotar a superfíce de decisão
%(D1 vs D2)
Test1_QDA_1v1 = QDA(data1, data2, X, Y);
Test1_SVM_1v1 = SVM(data1, data2, X, Y, KERNEL);

%(D2 vs D3)
Test2_QDA_1v1 = QDA(data2, data3, X, Y);
Test2_SVM_1v1 = SVM(data2, data3, X, Y, KERNEL);

%% Plotagem da base e superfíce de decisão
figure, plot(data1(1, :), data1(2, :), 'rv', data2(1, :), data2(2, :), 'gd', data3(1, :), data3(2, :), 'bo');
hold on, contour(X, Y, Test1_QDA_1v1, 1, '-k', 'LineWidth', 3);
hold on, contour(X, Y, Test2_QDA_1v1, 1, '-m', 'LineWidth', 3);
title('Base e Surperfície separadora pelo {\bf QDA}');
legend('D1', 'D2', 'D3', 'D1 x D2', 'D2 x D3', 'Location', 'bestoutside');

figure, plot(data1(1, :), data1(2, :), 'rv', data2(1, :), data2(2, :), 'gd', data3(1, :), data3(2, :), 'bo');
hold on, contour(X, Y, Test1_SVM_1v1, 1, '-k', 'LineWidth', 3);
hold on, contour(X, Y, Test2_SVM_1v1, 1, '-m', 'LineWidth', 3);
title("Base e Surperfície separadora pelo {\bf SVM com Kernel = " + upper(KERNEL) + "}");
legend('D1', 'D2', 'D3', 'D1 x D2', 'D2 x D3', 'Location', 'bestoutside');
