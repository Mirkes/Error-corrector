load('ErrorData.mat');

%%
% Debug of createNeedCorrection
tic;
res = createErrorCorrection(TP_vectors_new, FP_vectors_new,...
        'whitening', true, 'Space', 'Reduced');
toc

% Test of stat forming
stat = statsErrorCorrector(res, 'All');
