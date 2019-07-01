function T = SVM(X1, X2, X, Y, KERNEL)
    % Setando Base e Classe
    D = [X1, X2].';
    C = [ones(1, size(X1, 2)), zeros(1, size(X2, 2))].';
    % Treinamento
    if strcmp(KERNEL, 'polynomial')
        M = fitcsvm(D, C, 'Standardize', true, 'KernelFunction', KERNEL, 'PolynomialOrder', 2);
    else
        M = fitcsvm(D, C, 'Standardize', true, 'KernelFunction', KERNEL);
    end
    % Teste
    Test  = [X(:), Y(:)];
    L = predict(M, Test);
    T = vec2mat(L, size(X, 1)).';
end