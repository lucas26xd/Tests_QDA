function T = QDA(X1, X2, X, Y)
    %% TREINAMENTO
    % Pegando probabilidade de cada classe
    S = size(X1, 2) + size(X2, 2);
    PX1 = size(X1, 2) / S;
    PX2 = size(X2, 2) / S;

    % Média de cada classe
    U1 = mean(X1, 2);
    U2 = mean(X2, 2);

    % Matriz de covariâncias de cada classe
    S1 = cov(X1.');
    S2 = cov(X2.');

    %% TESTE
    % MODO MATRICIAL (Foi necessário dividir em multiplicações menores pois a matriz de pontos ultrapassava o limite de multiplicação entre matrizes no MATLAB)
    D = [X(:), Y(:)].';
    A1 = (D - U1).' * inv(S1);
    A2 = (D - U2).' * inv(S2);
    for i = 1:size(A1, 1)
        B1(i) = A1(i, :) * (D(:, i) - U1);
        B2(i) = A2(i, :) * (D(:, i) - U2);
    end
    
    P1 = log(norm(S1)) + B1 - 2 * log(PX1);
    P2 = log(norm(S2)) + B2 - 2 * log(PX2);

%   MODO ITERATIVO
%     for i = 1:size(X, 1)
%         for j = 1:size(X, 2)
%             x = [X(i, j); Y(i, j)];
%             P1(i, j) = log(norm(S1)) + (x - U1).' * inv(S1) * (x - U1) - 2 * log(PX1);
%             P2(i, j) = log(norm(S2)) + (x - U2).' * inv(S2) * (x - U2) - 2 * log(PX2);
%         end
%     end

    T = P1 < P2;
    T = vec2mat(T, size(X, 1)).'; % Remoldando vetor para matriz (Necessário apenas no modo matricial)
end