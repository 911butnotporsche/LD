%% Kalibravimas: PPD42NS + BME280 -> Palas Fidas Frog (PM2.5 ir PM10.0)
% Atkartoja Sensors 2025, 25(5), 1614 eksperimentus (Table 2 ir Table 3)
% autoriaus schema is:
% https://github.com/gokulbalagopal/Calibration-of-LoRaNodes-using-Super-Learners
%
% Priklausomybe: Statistics and Machine Learning Toolbox.
% Duomenys: calibrate_data.csv tame paciame kataloge kaip sis skriptas.
%
% Autoriaus notebook (calibrationFinal.ipynb):
%   - pozymiai: 9 LoRa stulpeliai (*_loRa)
%   - tikslai: Palas pm2_5Palas, pm10Palas (ir kiti, cia neimami)
%   - filtras: temperatura 0..45 C, slegis 30000..110000 Pa, dregme 0..100 %
%   - padalijimas: train_test_split(test_size=0.20, random_state=20) Table 2
%   - LR, medis, RF, bagging: be mastelio
%   - KNN ir MLP: StandardScaler X ir y (sklearn numatytieji hiperparametrai)
%   - RF: RandomForestRegressor(random_state=42) -> 100 medziu
%   - Bagging: BaggingRegressor(random_state=42) -> 10 medziu (n_estimators=10)
%   - Table 3 PM2.5: bazes KNN, medis, bagging, MLP; meta = Random Forest
%   - Table 3 PM10.0: meta = neuroninis tinklas
%     (XGB ir LightGBM toolbox neturi, juos pakeicia medis ir bagging)

clear; clc; close all;

%% 1. Duomenu paruosimas
rng(42, "twister");

scriptDir = fileparts(mfilename("fullpath"));
csvPath = fullfile(scriptDir, "calibrate_data.csv");
if ~isfile(csvPath)
    error("Nerastas failas: %s", csvPath);
end

raw = readtable(csvPath, "TextType", "string");
raw = rmmissing(raw);

featureNames = ["P1_lpo_loRa", "P1_ratio_loRa", "P1_conc_loRa", ...
    "P2_lpo_loRa", "P2_ratio_loRa", "P2_conc_loRa", ...
    "Temperature_loRa", "Pressure_loRa", "Humidity_loRa"];
targetNames = ["pm2_5Palas", "pm10Palas"];
targetLabels = ["PM2.5", "PM10.0"];

Xall = raw{:, featureNames};
Yall = raw{:, targetNames};

% Autoriaus data_checker ribos (Palas 0..100 mg/m3 = 100000 ug/m3).
tempOk = Xall(:, 7) >= 0 & Xall(:, 7) <= 45;
pressOk = Xall(:, 8) >= 300 * 100 & Xall(:, 8) <= 1100 * 100;
humidOk = Xall(:, 9) >= 0 & Xall(:, 9) <= 100;
pmOk = all(Yall >= 0 & Yall <= 100000, 2);
keep = tempOk & pressOk & humidOk & pmOk;
Xall = Xall(keep, :);
Yall = Yall(keep, :);

n = size(Xall, 1);
holdout = cvpartition(n, "HoldOut", 0.2);
idxTrain = training(holdout);
idxTest = test(holdout);

fprintf("Irasyta eiluciu po NaN ir filtro: %d\n", n);
fprintf("Treniravimas: %d (%.1f%%), testavimas: %d (%.1f%%)\n", ...
    sum(idxTrain), 100 * mean(idxTrain), sum(idxTest), 100 * mean(idxTest));
fprintf("Padalijimas: cvpartition HoldOut 0.2, rng(42).\n");
fprintf("Straipsnio Table 2 naudotas sklearn random_state=20, todel R^2 nebus tapatus.\n\n");

% Straipsnio Table 2 R^2 (train / test) palyginimui.
paperR2 = struct( ...
    "PM25", dictionary( ...
        ["LinearRegression", "DecisionTree", "RandomForest", ...
         "EnsembleBagging", "KNN", "MLP"], ...
        {[0.51, 0.50], [1.00, 0.98], [1.00, 0.98], ...
         [1.00, 0.97], [0.96, 0.87], [0.94, 0.88]}), ...
    "PM10", dictionary( ...
        ["LinearRegression", "DecisionTree", "RandomForest", ...
         "EnsembleBagging", "KNN", "MLP"], ...
        {[0.16, 0.17], [1.00, 0.81], [0.99, 0.89], ...
         [0.98, 0.90], [0.82, 0.59], [0.63, 0.59]}));

modelOrder = ["LinearRegression", "DecisionTree", "RandomForest", ...
    "EnsembleBagging", "KNN", "MLP"];

%% 2. Baziniai modeliai (Table 2 poaibis, kuris yra Statistics and ML Toolbox)
baseStore = struct();
for t = 1:numel(targetLabels)
    label = targetLabels(t);
    y = Yall(:, t);
    yTrain = y(idxTrain);
    yTest = y(idxTest);
    XTrain = Xall(idxTrain, :);
    XTest = Xall(idxTest, :);

    fprintf("========== %s | baziniai modeliai ==========\n", label);
    fprintf("%-18s %10s %10s %12s %12s %18s\n", ...
        "Model", "R2_train", "R2_test", "RMSE_train", "RMSE_test", "Paper R2 tr/te");

    metrics = struct();
    predictions = struct();
    for m = 1:numel(modelOrder)
        name = modelOrder(m);
        fitted = trainModel(name, XTrain, yTrain);
        predTrain = predictModel(fitted, XTrain);
        predTest = predictModel(fitted, XTest);
        [r2tr, rmseTr] = regressionScores(yTrain, predTrain);
        [r2te, rmseTe] = regressionScores(yTest, predTest);

        if label == "PM2.5"
            ref = paperR2.PM25(name);
        else
            ref = paperR2.PM10(name);
        end
        if iscell(ref)
            ref = ref{1};
        end
        fprintf("%-18s %10.4f %10.4f %12.4f %12.4f %10.2f / %5.2f\n", ...
            name, r2tr, r2te, rmseTr, rmseTe, ref(1), ref(2));

        metrics.(name) = struct("R2Train", r2tr, "R2Test", r2te, ...
            "RMSETrain", rmseTr, "RMSETest", rmseTe);
        predictions.(name) = predTest;
    end
    fprintf("\n");

    if label == "PM2.5"
        key = "PM25";
    else
        key = "PM10";
    end
    baseStore.(key) = struct("metrics", metrics, "yTest", yTest, ...
        "predTest", predictions);
end

%% 3. Super Learner (5-fold out-of-fold stakingas)
% PM2.5 atitinka Table 3: KNN + medis + bagging + MLP, meta = Random Forest.
% PM10.0 straipsnyje meta = MLP, bazes KNN + RF + XGB + LightGBM.
% XGB ir LightGBM nera toolbox, todel bazes: KNN, RF, medis, bagging.
slPlan = struct( ...
    "PM25", struct( ...
        "bases", ["KNN", "DecisionTree", "EnsembleBagging", "MLP"], ...
        "meta", "RandomForest", ...
        "paperTrain", 1.00, "paperTest", 0.99), ...
    "PM10", struct( ...
        "bases", ["KNN", "RandomForest", "DecisionTree", "EnsembleBagging"], ...
        "meta", "MLP", ...
        "paperTrain", 0.98, "paperTest", 0.91));

slStore = struct();
fprintf("========== Super Learner (Table 3 schema) ==========\n");
fprintf("%-8s %-42s %-16s %10s %10s %12s %12s %18s\n", ...
    "Target", "Base learners", "Meta", "R2_train", "R2_test", ...
    "RMSE_train", "RMSE_test", "Paper R2 tr/te");

for t = 1:numel(targetLabels)
    label = targetLabels(t);
    if label == "PM2.5"
        plan = slPlan.PM25;
    else
        plan = slPlan.PM10;
    end

    y = Yall(:, t);
    sl = fitSuperLearner(Xall(idxTrain, :), y(idxTrain), ...
        Xall(idxTest, :), plan.bases, plan.meta);

    [r2tr, rmseTr] = regressionScores(y(idxTrain), sl.yhatTrain);
    [r2te, rmseTe] = regressionScores(y(idxTest), sl.yhatTest);

    fprintf("%-8s %-42s %-16s %10.4f %10.4f %12.4f %12.4f %10.2f / %5.2f\n", ...
        label, strjoin(plan.bases, ", "), plan.meta, ...
        r2tr, r2te, rmseTr, rmseTe, plan.paperTrain, plan.paperTest);

    if label == "PM2.5"
        key = "PM25";
    else
        key = "PM10";
    end
    slStore.(key) = struct("yTest", y(idxTest), "yhatTest", sl.yhatTest, ...
        "R2Test", r2te, "RMSETest", rmseTe, "plan", plan);
end
fprintf("\nTrain R^2 skaiciuotas taip, kaip sklearn StackingRegressor:\n");
fprintf("bazes permokytos ant visos treniravimo imties, tada meta-modelis.\n");
fprintf("Tai optimistinis train ivertis (straipsnyje artimas 1.00).\n");

%% 4. Vizualizacija: testavimo imtis, Super Learner
figure("Name", "Super Learner kalibravimas", "Color", "w", ...
    "Position", [80 80 1100 720]);
keys = ["PM25", "PM10"];
titles = ["PM2.5", "PM10.0"];
for t = 1:2
    yTest = slStore.(keys(t)).yTest;
    yhat = slStore.(keys(t)).yhatTest;
    residual = yTest - yhat;

    subplot(2, 2, t);
    scatter(yTest, yhat, 18, [0.15 0.35 0.70], "filled", ...
        "MarkerFaceAlpha", 0.45);
    hold on;
    lo = min([yTest; yhat]);
    hi = max([yTest; yhat]);
    plot([lo hi], [lo hi], "k--", "LineWidth", 1.2);
    hold off;
    grid on;
    axis tight;
    xlabel(sprintf("Palas %s (ug/m^3)", titles(t)));
    ylabel("Super Learner prognoze");
    title(sprintf("%s testas  R^2 = %.3f, RMSE = %.3f", ...
        titles(t), slStore.(keys(t)).R2Test, slStore.(keys(t)).RMSETest));
    legend({"Testas", "1:1"}, "Location", "northwest");

    subplot(2, 2, t + 2);
    histogram(residual, 30, "FaceColor", [0.75 0.20 0.20], "EdgeColor", "none");
    hold on;
    xline(0, "k-", "LineWidth", 1.1);
    xline(-5, "k--");
    xline(5, "k--");
    hold off;
    grid on;
    xlabel("Liekamoji paklaida (faktas - prognoze)");
    ylabel("Daznis");
    title(sprintf("%s liekanos (slenkstis \\pm5, kaip Fig. 8)", titles(t)));
end

%% Lokalios funkcijos
function fitted = trainModel(name, X, y)
    switch name
        case "LinearRegression"
            % sklearn LinearRegression: OLS, intercept=True. Be mastelio.
            fitted = struct("name", name, "model", fitlm(X, y));

        case "DecisionTree"
            % sklearn DecisionTreeRegressor(random_state=42):
            % min_samples_split=2, min_samples_leaf=1, be geniojimo prognozej.
            fitted = struct("name", name, "model", fitrtree(X, y, ...
                "MinParentSize", 2, ...
                "MinLeafSize", 1, ...
                "Surrogate", "off"));

        case "RandomForest"
            % sklearn RandomForestRegressor(): 100 medziu, visi pozymiai.
            fitted = struct("name", name, "model", fitBaggedForest(X, y, 100));

        case "EnsembleBagging"
            % sklearn BaggingRegressor(): 10 sprendimu medziu, bootstrap.
            fitted = struct("name", name, "model", fitBaggedForest(X, y, 10));

        case "KNN"
            % sklearn KNeighborsRegressor(): k=5, euclidean, vienodi svoriai.
            % StandardScaler (populiacijos std, ddof=0) tik ant treniravimo.
            % R2026a nebeturi fitrknn; jei funkcija yra, ji naudojama,
            % kitaip tas pats algoritmas per knnsearch.
            [Xs, muX, sigX] = zscoreFit(X);
            [ys, muY, sigY] = zscoreFit(y);
            k = 5;
            if exist("fitrknn", "file") == 2
                model = fitrknn(Xs, ys, ...
                    "NumNeighbors", k, ...
                    "Distance", "euclidean", ...
                    "Standardize", false, ...
                    "DistanceWeight", "equal");
                backend = "fitrknn";
            else
                model = struct("X", Xs, "y", ys, "k", k);
                backend = "knnsearch";
            end
            fitted = struct("name", name, "model", model, "backend", backend, ...
                "muX", muX, "sigX", sigX, "muY", muY, "sigY", sigY);

        case "MLP"
            % sklearn MLPRegressor(random_state=42): 1 sluoksnis, 100 neuronu,
            % relu, alpha=1e-4, max_iter=200, Adam.
            % fitrnet naudoja L-BFGS, todel NN skaiciai gali skirtis.
            [Xs, muX, sigX] = zscoreFit(X);
            [ys, muY, sigY] = zscoreFit(y);
            model = fitrnet(Xs, ys, ...
                "LayerSizes", 100, ...
                "Activations", "relu", ...
                "Standardize", false, ...
                "Lambda", 1e-4, ...
                "IterationLimit", 200, ...
                "Verbose", 0);
            fitted = struct("name", name, "model", model, ...
                "muX", muX, "sigX", sigX, "muY", muY, "sigY", sigY);

        otherwise
            error("Nezinomas modelis: %s", name);
    end
end

function yhat = predictModel(fitted, X)
    switch fitted.name
        case {"LinearRegression", "DecisionTree", "RandomForest", "EnsembleBagging"}
            yhat = predict(fitted.model, X);
        case "KNN"
            Xs = zscoreApply(X, fitted.muX, fitted.sigX);
            if fitted.backend == "fitrknn"
                yhat = predict(fitted.model, Xs);
            else
                k = min(fitted.model.k, size(fitted.model.X, 1));
                idx = knnsearch(fitted.model.X, Xs, "K", k, "Distance", "euclidean");
                yhat = mean(fitted.model.y(idx), 2);
            end
            yhat = yhat .* fitted.sigY + fitted.muY;
        case "MLP"
            Xs = zscoreApply(X, fitted.muX, fitted.sigX);
            yhat = predict(fitted.model, Xs);
            yhat = yhat .* fitted.sigY + fitted.muY;
        otherwise
            error("Nezinomas modelis: %s", fitted.name);
    end
    yhat = yhat(:);
end

function model = fitBaggedForest(X, y, nTrees)
    % Bag + tree. NumVariablesToSample="all" atitinka sklearn max_features=1.0.
    learner = templateTree( ...
        "MinLeafSize", 1, ...
        "MinParentSize", 2, ...
        "NumVariablesToSample", "all", ...
        "Reproducible", true, ...
        "Surrogate", "off");
    model = fitrensemble(X, y, ...
        "Method", "Bag", ...
        "NumLearningCycles", nTrees, ...
        "Learners", learner);
end

function sl = fitSuperLearner(XTrain, yTrain, XTest, baseNames, metaName)
    n = size(XTrain, 1);
    nBase = numel(baseNames);
    inner = cvpartition(n, "KFold", 5);
    oof = zeros(n, nBase);

    for fold = 1:inner.NumTestSets
        tr = training(inner, fold);
        te = test(inner, fold);
        for b = 1:nBase
            fitted = trainModel(baseNames(b), XTrain(tr, :), yTrain(tr));
            pred = predictModel(fitted, XTrain(te, :));
            oof(te, b) = pred;
        end
    end

    meta = trainModel(metaName, oof, yTrain);

    fullPredTrain = zeros(n, nBase);
    fullPredTest = zeros(size(XTest, 1), nBase);
    for b = 1:nBase
        fitted = trainModel(baseNames(b), XTrain, yTrain);
        fullPredTrain(:, b) = predictModel(fitted, XTrain);
        fullPredTest(:, b) = predictModel(fitted, XTest);
    end

    sl = struct();
    sl.yhatTrain = predictModel(meta, fullPredTrain);
    sl.yhatTest = predictModel(meta, fullPredTest);
    sl.oof = oof;
end

function [mu, sigma] = columnStats(X)
    mu = mean(X, 1);
    sigma = std(X, 1, 1); % ddof=0, kaip sklearn StandardScaler
    sigma(sigma == 0) = 1;
end

function [Xs, mu, sigma] = zscoreFit(X)
    X = double(X);
    if isvector(X)
        X = X(:);
    end
    [mu, sigma] = columnStats(X);
    Xs = (X - mu) ./ sigma;
end

function Xs = zscoreApply(X, mu, sigma)
    X = double(X);
    if isvector(X)
        X = X(:);
    end
    Xs = (X - mu) ./ sigma;
end

function [r2, rmse] = regressionScores(yTrue, yPred)
    yTrue = yTrue(:);
    yPred = yPred(:);
    resid = yTrue - yPred;
    ssRes = sum(resid .^ 2);
    ssTot = sum((yTrue - mean(yTrue)) .^ 2);
    r2 = 1 - ssRes / ssTot;
    rmse = sqrt(mean(resid .^ 2));
end
