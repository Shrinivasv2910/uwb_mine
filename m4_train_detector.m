function net = m4_train_detector(cfg)
% M4_TRAIN_DETECTOR  Build and train the LSTM NLOS detector.
%   net = m4_train_detector(cfg)
% Saves the trained network to nlos_detector.mat and a training-curve figure.
%
% Requires: Deep Learning Toolbox.

fprintf('=== Building datasets ===\n');
trainSeeds = cfg.seedOffset + (1:cfg.nTrainSeeds);
valSeeds   = cfg.seedOffset + 100 + (1:cfg.nValSeeds);

[Xtr, Ytr] = m4_build_dataset(cfg, trainSeeds);
[Xva, Yva] = m4_build_dataset(cfg, valSeeds);

% ---- Class weighting: NLOS is the minority class ----
classes = categories(Ytr);
counts  = countcats(Ytr);
w = sum(counts) ./ (numel(counts)*counts);   % inverse-frequency weights

% ---- Network: small bidirectional LSTM over the 9-sample window ----
layers = [
    sequenceInputLayer(cfg.nFeat, 'Name','in')
    bilstmLayer(48, 'OutputMode','last', 'Name','bilstm')
    dropoutLayer(0.3, 'Name','drop')
    fullyConnectedLayer(32, 'Name','fc1')
    reluLayer('Name','relu')
    fullyConnectedLayer(2, 'Name','fc2')
    softmaxLayer('Name','softmax')
    classificationLayer('Classes',classes,'ClassWeights',w,'Name','out')
    ];

opts = trainingOptions('adam', ...
    'MaxEpochs',           25, ...
    'MiniBatchSize',       256, ...
    'InitialLearnRate',    1e-3, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'GradientThreshold',   1, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      {Xva, Yva}, ...
    'ValidationFrequency', 100, ...
    'Plots',               'training-progress', ...
    'Verbose',             true, ...
    'ExecutionEnvironment','auto');

fprintf('=== Training detector ===\n');
net = trainNetwork(Xtr, Ytr, layers, opts);

save(fullfile(pwd,'nlos_detector.mat'),'net','cfg');
fprintf('Saved nlos_detector.mat\n');

% Save the training-progress figure if open
figs = findall(0,'Type','figure');
if ~isempty(figs)
    try
        exportgraphics(figs(1), fullfile(cfg.figdir,'fig_training_progress.png'), ...
            'Resolution',150);
    catch
    end
end
end