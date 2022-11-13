function res = statsErrorCorrector(data, varargin)
%statsErrorCorrector calculate statistics for models formed for error
%correction by createErrorCorrection function.
%
% Iputs:
%   data is structure formed by createErrorCorrection
%   Optional parameters defined statistics to calculate:
%   ROC curves:
%       'ROCTrain+' is assumed by default and calculate ROC curve for
%           training set. Method of forming classifier for clusters is
%           defined by parameters described in Statistics for many
%           clusters. if 'DoNotSaveFigures' is not specified then figure
%           will be saved into file "ROCTrain.png" in folder specified in
%           argument 'Dir' (see below).
%       'ROCTrain-' do not form figure with ROC curve for trainig set.
%       'ROCTest+' is assumed by default and calculate ROC curve for test
%           set. Method of forming classifier for clusters is defined by
%           parameters described in Statistics for many clusters. if
%           'DoNotSaveFigures' is not specified then figure will be saved
%           into file "ROCTrain.png" in folder specified in argument 'Dir'
%           (see below). 
%       'ROCTest-' do not form figure with ROC curve for test set.
%
%   Distributions:
%       'DistrTraining' draw histograms for corectly and wrongly recognised
%           dataset without clustering (with 1 cluster). Graph also
%           contains the best threshold for Balanced Accuracy. if
%           'DoNotSaveFigures' is not specified then figure will be saved
%           into file "DistrTraining.png" in folder specified in argument
%           'Dir' (see below). 
%       'DistrTest' draw histograms for corectly and wrongly recognised
%           dataset without clustering (with 1 cluster). Graph also
%           contains the best threshold for Balanced Accuracy. if
%           'DoNotSaveFigures' is not specified then figure will be saved
%           into file "DistrTest.png" in folder specified in argument
%           'Dir' (see below). 
%       'DistrTrainingClust' draw histograms for corectly and wrongly
%           recognised dataset with many clusters. Graph also contains the
%           best threshold for Balanced Accuracy. Calculation of projection
%           for clusters defined by Statistics for many clusters. if
%           'DoNotSaveFigures' is not specified then figure will be saved
%           into file "DistrTrainingClust_xxx.png" in folder specified in
%           argument 'Dir' (see below), where xxx is number of cluster. 
%       'DistrTestClust' draw histograms for corectly and wrongly
%           recognised dataset with many clusters. Graph also contains the
%           best threshold for Balanced Accuracy. Calculation of projection
%           for clusters defined by Statistics for many clusters. if
%           'DoNotSaveFigures' is not specified then figure will be saved
%           into file "DistrTestClust_xxx.png" in folder specified in
%           argument 'Dir' (see below), where xxx is number of cluster. 
%       'DrawClusters' requires drawing histograms for each cluster
%           separately. if 'DoNotSaveFigures' is not specified then figure
%           will be saved into file "DistrClust_xxx.png" in folder
%           specified in argument 'Dir' (see below), where xxx is number of
%           cluster.
%
%   Special argument:
%       'All' can be used instead of 'ROCTrain+', 'ROCTest+',
%       'DistrTraining', 'DistrTest', 'DistrTrainingClust',
%       'DistrTestClust', 'DrawClusters'.
%
%   General parameters:
%       'Bins' is indicator that the next value will be number of bins to
%           form distribitions. Default value is 200.
%       'Dir' is indicator that the next value is directory name or
%           directory path to save figures. Default value is 'figures'.
%           Note that there must not be "/" before or after directory name.
%           Examples: 'figure', 'c:/my directory/fig', '../figs'. Forms
%           '/figure' or 'figure/' sre wrong.
%       'DoNotSaveFigures' prevents saveing of figures.
%       'CloseFigures' close each figure after save. It is not recommended
%           to close figures if 'DoNotSaveFigures' is specified. It is very
%           reasonable to close figures if 'DrawClusters' is specified.
%
%   Statistics for many clusters:
%       By default assumed unit length of Fisher's direction vectors and
%           the same threshold for all clusters.
%       'STDScale' all Fisher's direction vectors are scaled for unit
%           standard deviations of projections. Projections also shifted to
%           zero mean. STD and mean calculated for training set for this
%           cluster.
%       'RangeScale' all Fisher's direction vectors are scaled and
%           projections are shifted to the interval [-1, 1] for trainig
%           set.
%
% Outputs:
%   Function return one structure which contains following fields:
%       oneClusterTrainingStat if specified 'DistrTraining'. This field
%           contains array with elements described in Distribution
%           statistics.
%       oneClusterTestStat if specified 'DistrTest'. If 'DistrTraining' was
%           specified then this field contains array with elements
%           described in Extended distribution statistics. Otherwise this
%           field contains array with elements described in Distribution
%           statistics. 
%       clustersTrainingStat and clustersTrainingStat if specified
%           'DrawClusters'. This fields contain following information:
%           'clustersTrainingStat' is numClust-by-6 matrix with columns
%               described in Distribution statistics.
%           'clustersTrestStat' is numClust-by-9 matrix with columns
%               described in Extended distribution statistics.
%       ROCTrain and ROCTest if not specified 'ROCTrain-' and 'ROCTest-'.
%           These fields contains area under the ROC curve for one
%           corrector (one cluster) and for corrector with many clusters.

%       'DistrTrainingClustStat' if specified 'DistrTrainingClust'. This
%           field contains array with elements described in Distribution
%           statistics.
%       'DistrTestClustStat' if specified 'DistrTestClust'. If
%           'DistrTrainingClust' was specified then this field contains
%           array with elements described in Extended distribution
%           statistics. Otherwise this field contains array with elements
%           described in Distribution statistics. 
%
%   Distribution statistics. This array contains following 6 elements
%       res(1) is number of cases of the first class
%       res(2) is number of cases of the second class
%       res(3) is the best threshold for balanced accuracy:
%           half of sencsitivity + specificity or 
%               0.5*(TP/Pos+TN/Neg), where 
%           TP is true positive or the number of correctly recognised
%               casses of the first class (, 
%           Pos is the number of casses of the first class (res(1)),
%           TN is true negative or the number of correctly recognised
%               casses of the secong class, 
%           Neg is the number of casses of the second class.
%       res(4) is true positive rate or TP/Pos
%       res(5) is true negative rate or TN/Neg
%       res(6) is Balanced accuracy
%
%   Extended distribution statistics. This array is formed for test sets if
%   threshold for training set is specified. This array contains following
%   9 elements  
%       res(1) is number of cases of the first class
%       res(2) is number of cases of the second class
%       res(3) is the best threshold for balanced accuracy:
%           half of sencsitivity + specificity or 
%               0.5*(TP/Pos+TN/Neg), where 
%           TP is true positive or the number of correctly recognised
%               casses of the first class (, 
%           Pos is the number of casses of the first class (res(1)),
%           TN is true negative or the number of correctly recognised
%               casses of the secong class, 
%           Neg is the number of casses of the second class.
%       res(4) is true positive rate or TP/Pos
%       res(5) is true negative rate or TN/Neg
%       res(6) is Balanced accuracy
%       res(7) is true positive rate or TP/Pos for predefined threshold
%       res(8) is true negative rate or TN/Neg for predefined threshold
%       res(9) is Balanced accuracy for predefined threshold
%
% References
% This work is based on the theory and algorithms presented in:
% [1]   Gorban, A.N. and Tyukin, I.Y., 2018. Blessing of dimensionality:
%       mathematical foundations of the statistical physics of data.
%       Philosophical Transactions of the Royal Society A: Mathematical,
%       Physical and Engineering Sciences, 376(2118), p.20170237.
% [2]   Tyukin, I.Y., Gorban, A.N., Sofeykov, K.I. and Romanenko, I., 2018.
%       Knowledge transfer between artificial intelligence systems.
%       Frontiers in neurorobotics, 12, p.49.
% [3]   Gorban, A.N., Golubkov, A., Grechuk, B., Mirkes, E.M. and Tyukin,
%       I.Y., 2018. Correction of AI systems by linear discriminants:
%       Probabilistic foundations. Information Sciences, 466, pp.303-322.
% [4]   Gorban, A.N., Grechuk, B., Mirkes, E.M., Stasenko, S.V. and Tyukin,
%       I.Y., 2021. High-dimensional separability for one-and few-shot
%       learning. Entropy, 23(8), p.1090. https://doi.org/10.3390/e23081090
% [5]   Grechuk, B., Gorban, A.N., and Tyukin, I.Y., 2021. General
%       stochastic separation theorems with optimal bounds.  Neural
%       Networks, 38, pp.33-56, https://doi.org/10.1016/j.neunet.2021.01.034 
% [6]   Tyukin, I.Y., Gorban, A.N., McEwan, A.A., Meshkinfamfard, S., and
%       Tang, L., 2021. Blessing of dimensionality at the edge and geometry
%       of few-shot learning. Information Sciences, 564, pp.124-143.
%       https://doi.org/10.1016/j.ins.2021.01.022    
%
% Please cite as:
% Gorban, A.N., Tyukin, I.Y., Mirkes, E.M., and Stasenko, S.V., 2022. Error
% corrector, Available online https://github.com/Mirkes/Error-corrector
%
% (c) Alexander N. Gorban, Ivan Y. Tyukin, Evgeny M. Mirkes, Sergey V. Stasenko


    % Analysis of optional variables
    % Default values
    ROCTrain = true;
    ROCTest = true;
    DistrTraining = false;
    DistrTest = false;
    DistrTrainingClust = false;
    DistrTestClust = false;
    DrawClusters = false;
    SaveFigures = true;
    CloseFigures = false;
    nBins = 200;
    dirs = 'figures\';
    scaler = 0;
    % Check optional arguments
    k = 1;
    while k <= length(varargin)
        s = varargin{k};
        if ~(isstring(s) || ischar(s))
            error("Inacceptable type of input argument %s", s);
        end
        if strcmpi(s, 'ROCTrain+')
            ROCTrain = true;
        elseif strcmpi(s, 'ROCTrain-')
            ROCTrain = false;
        elseif strcmpi(s, 'ROCTest+')
            ROCTest = true;
        elseif strcmpi(s, 'ROCTest-')
            ROCTest = false;
        elseif strcmpi(s, 'DistrTraining')
            DistrTraining = true;
        elseif strcmpi(s, 'DistrTest')
            DistrTest = true;
        elseif strcmpi(s, 'DistrTrainingClust')
            DistrTrainingClust = true;
        elseif strcmpi(s, 'DistrTestClust')
            DistrTestClust = true;
        elseif strcmpi(s, 'DrawClusters')
            DrawClusters = true;
        elseif strcmpi(s, 'All')
            ROCTrain = true;
            ROCTest = true;
            DistrTraining = true;
            DistrTest = true;
            DistrTrainingClust = true;
            DistrTestClust = true;
            DrawClusters = true;
        elseif strcmpi(s, 'DoNotSaveFigures')
            SaveFigures = false;
        elseif strcmpi(s, 'CloseFigures')
            CloseFigures = true;
        elseif strcmpi(s, 'STDScale')
            scaler = 1;
        elseif strcmpi(s, 'RangeScale')
            scaler = 2;
        elseif strcmpi(s, 'Bins')
            tmp = varargin{k};
            if ~isnumeric(tmp) || tmp < 2
                error("Number of bins must be integer greater than 1. Value %s is inappropriate", tmp);
            end
            nBins = round(tmp);
            % Skip used value
            k = k + 1;
        elseif strcmpi(s, 'Dir')
            dirs = [char(varargin{k}), "\"];
            % Skip used value
            k = k + 1;
        else
            error("Inacceptable argument '%s'", s)
        end
        % Next argument
        k = k + 1;
    end

    % calculate centralised datasets
    CRLSC = data.CRLS - data.centre;
    CRTSC = data.CRTS - data.centre;
    WRLSC = data.WRLS - data.centre;
    WRTSC = data.WRTS - data.centre;

    % Project to reduced space if necessary
    if strcmpi(data.space, 'reduced')
        CRLSC = CRLSC * data.project;
        CRTSC = CRTSC * data.project;
        WRLSC = WRLSC * data.project;
        WRTSC = WRTSC * data.project;
    else
        % Apply "whitening" in original space
        CRLSC = CRLSC ./ data.project;
        CRTSC = CRTSC ./ data.project;
        WRLSC = WRLSC ./ data.project;
        WRTSC = WRTSC ./ data.project;
    end
    
    % Calcuate Fisher projections for one cluster (without clustering)
    CRLProj = CRLSC * data.FD1;
    CRTProj = CRTSC * data.FD1;
    WRLProj = WRLSC * data.FD1;
    WRTProj = WRTSC * data.FD1;

    % Assign all points to clusters and calculate projections
    nClust = size(data.centroids, 1);
    CRLProjClust = splitToClusters(CRLSC, data.centroids, data.FD);
    CRTProjClust = splitToClusters(CRTSC, data.centroids, data.FD);
    WRLProjClust = splitToClusters(WRLSC, data.centroids, data.FD);
    WRTProjClust = splitToClusters(WRTSC, data.centroids, data.FD);

    res = struct();

    % Form figures for individual clusters and for total data
    if DistrTraining
        res.oneClusterTrainingStat = oneClusterGraph(CRLProj, WRLProj,...
            nBins, 'Distribution of training set');
        if SaveFigures
            saveFigures([dirs, 'DistrTraining.png']);
        end
        if CloseFigures
            close;
        end
    end

    if DistrTest
        if DistrTraining
            res.oneClusterTestStat = oneClusterGraph(CRTProj, WRTProj,...
                nBins, 'Distribution of test set',...
                res.oneClusterTrainingStat(3));
        else
            res.oneClusterTestStat = oneClusterGraph(CRTProj, WRTProj,...
                nBins, 'Distribution of test set');
        end
        if SaveFigures
            saveFigures([dirs, 'DistrTest.png']);
        end
        if CloseFigures
            close;
        end
    end

    if DrawClusters
        % Allocate arrays for statistics
        res.clustersTrainingStat = zeros(nClust, 6);
        res.clustersTestStat = zeros(nClust, 9);
        for k = 1:nClust
            res.clustersTrainingStat(k, :) = oneClusterGraph(CRLProjClust{k},...
                WRLProjClust{k}, round(nBins / 10),...
                sprintf("Distribution of training set for cluster %03d", k));
            if res.clustersTrainingStat(k, 1) > 0 &&...
                    res.clustersTrainingStat(k, 2) > 0
                if SaveFigures
                    saveFigures(sprintf("%sDistrTrainingClust_%03d.png", dirs, k));
                end
                if CloseFigures
                    close;
                end
            end
            res.clustersTestStat(k, :) = oneClusterGraph(CRTProjClust{k},...
                WRTProjClust{k}, round(nBins / 10),...
                sprintf("Distribution of test set for cluster %03d", k),...
                res.clustersTrainingStat(k, 3));
            if res.clustersTestStat(k, 1) > 0 &&...
                    res.clustersTestStat(k, 2) > 0
                if SaveFigures
                    saveFigures(sprintf("%sDistrTestClust_%03d.png", dirs, k));
                end
                if CloseFigures
                    close;
                end
            end
        end
    end

    % Preprocessing fisher's discriminant projections for ROCCurves (FDD)
    % Preallocate arrays for scaled data
    CRLProjSc = zeros(size(CRLProj));
    CRTProjSc = zeros(size(CRTProj));
    WRLProjSc = zeros(size(WRLProj));
    WRTProjSc = zeros(size(WRTProj));
    nextCL = 1;
    nextCT = 1;
    nextWL = 1;
    nextWT = 1;
    % Go through clusters
    for k = 1:nClust
        if scaler == 1
            % Skip if the training set data are empty.
            if (length(CRLProjClust{k}) + length(CRTProjClust{k})) == 0
                continue
            end
            scale = std([CRLProjClust{k}; CRTProjClust{k}]);
            shift = mean([CRLProjClust{k}; CRTProjClust{k}]);
            n = length(CRLProjClust{k});
            CRLProjSc(nextCL:nextCL+n-1) = (CRLProjClust{k} - shift) / scale;
            nextCL = nextCL + n;
            n = length(CRTProjClust{k});
            CRTProjSc(nextCT:nextCT+n-1) = (CRTProjClust{k} - shift) / scale;
            nextCT = nextCT + n;
            n = length(WRLProjClust{k});
            WRLProjSc(nextWL:nextWL+n-1) = (WRLProjClust{k} - shift) / scale;
            nextWL = nextWL + n;
            n = length(WRTProjClust{k});
            WRTProjSc(nextWT:nextWT+n-1) = (WRTProjClust{k} - shift) / scale;
            nextWT = nextWT + n;
        elseif scaler == 3
            % Skip if the training set data are empty.
            if (length(CRLProjClust{k}) + length(CRTProjClust{k})) == 0
                continue
            end
            ma = max([CRLProjClust{k}; CRTProjClust{k}]);
            mi = min([CRLProjClust{k}; CRTProjClust{k}]);
            scale = ma - mi;
            sfift = mi + ma;
            n = length(CRLProjClust{k});
            CRLProjSc(nextCL:nextCL+n-1) = (2 * CRLProjClust{k} - shift) / scale;
            nextCL = nextCL + n;
            n = length(CRTProjClust{k});
            CRTProjSc(nextCT:nextCT+n-1) = (2 * CRTProjClust{k} - shift) / scale;
            nextCT = nextCT + n;
            n = length(WRLProjClust{k});
            WRLProjSc(nextWL:nextWL+n-1) = (2 * WRLProjClust{k} - shift) / scale;
            nextWL = nextWL + n;
            n = length(WRTProjClust{k});
            WRTProjSc(nextWT:nextWT+n-1) = (2 * WRTProjClust{k} - shift) / scale;
            nextWT = nextWT + n;
        else
            n = length(CRLProjClust{k});
            CRLProjSc(nextCL:nextCL+n-1) = CRLProjClust{k};
            nextCL = nextCL + n;
            n = length(CRTProjClust{k});
            CRTProjSc(nextCT:nextCT+n-1) = CRTProjClust{k};
            nextCT = nextCT + n;
            n = length(WRLProjClust{k});
            WRLProjSc(nextWL:nextWL+n-1) = WRLProjClust{k};
            nextWL = nextWL + n;
            n = length(WRTProjClust{k});
            WRTProjSc(nextWT:nextWT+n-1) = WRTProjClust{k};
            nextWT = nextWT + n;
        end
    end
    % Remove unused fragments
    if nextCL < length(CRLProjSc)
        CRLProjSc(nextCL:end) = [];
    end
    if nextCT < length(CRTProjSc)
        CRTProjSc(nextCT:end) = [];
    end
    if nextWL < length(WRLProjSc)
        WRLProjSc(nextWL:end) = [];
    end
    if nextWT < length(WRTProjSc)
        WRTProjSc(nextWT:end) = [];
    end
    % Draw ROC if it is required
    if ROCTrain
        res.ROCTrain = ROCDraw(CRLProj, WRLProj, CRLProjSc, WRLProjSc,...
            'Training set');
        if SaveFigures
            saveFigures(sprintf("%sROCTraining.png", dirs));
        end
        if CloseFigures
            close;
        end
    end
    if ROCTest
        res.ROCTest = ROCDraw(CRTProj, WRTProj, CRTProjSc, WRTProjSc,...
            'Test set');
        if SaveFigures
            saveFigures(sprintf("%sROCTest.png", dirs));
        end
        if CloseFigures
            close;
        end
    end

    if DistrTrainingClust
        res.DistrTrainingClustStat = oneClusterGraph(CRLProjSc, WRLProjSc,...
            nBins, 'Distribution of training set with clusters');
        if SaveFigures
            saveFigures([dirs, 'DistrTrainingClust.png']);
        end
        if CloseFigures
            close;
        end
    end

    if DistrTestClust
        if DistrTrainingClust
            res.DistrTestClustStat = oneClusterGraph(CRTProjSc, WRTProjSc,...
                nBins, 'Distribution of test set',...
                res.DistrTrainingClustStat(3));
        else
            res.DistrTestClustStat = oneClusterGraph(CRTProjSc, WRTProjSc,...
                nBins, 'Distribution of test set with clusters');
        end
        if SaveFigures
            saveFigures([dirs, 'DistrTestClust.png']);
        end
        if CloseFigures
            close;
        end
    end
end

function res = splitToClusters(data, centroids, FD)
% Inputs:
%   data is set of data points, one point in row.
%   centroids matrix of centroids for clusters with one centroid per row
%   FD is matrix of Fisher's discriminant directions. One direction per
%       column.
% Outputs:
%   res is cell array. Each cell contains projections of datapoints of one
%       cluster onto corresponding FD of this cluster.

    %  Create array for output
    nClust = size(FD, 2);
    res = cell(nClust, 1);
    % Calculate distances to centroids
    dist = sum(data .^ 2, 2) + sum(centroids' .^ 2) - 2 * data * centroids';
    % Find the minimal distance
    [~, lab] = min(dist, [], 2);
    % Identify all points with the closest centroids and calculate
    % projections on FD
    for k = 1:nClust
        ind = lab == k;
        proj = data(ind, :) * FD(:, k);
        res(k) = {proj};
    end
end

function res = oneClusterGraph(x, y, nBins, name, prThres)
%oneDClass applied classification with one input attribute by searching
%the best threshold.
%
% Inputs:
%   x contains values for the first class
%   y contains values for the second class
%   nBins is number of bins to use
%   name is string with title of axis
%   prThres is predefined threshold.
%
% Outputs:
%   res is row vector with 6 values (9 if predefined threshold is
%       specified): 
%       res(1) is number of cases of the first class
%       res(2) is number of cases of the second class
%       res(3) is the best threshold for balanced accuracy:
%           half of sencsitivity + specificity or 
%               0.5*(TP/Pos+TN/Neg), where 
%           TP is true positive or the number of correctly recognised
%               casses of the first class (, 
%           Pos is the number of casses of the first class (res(1)),
%           TN is true negative or the number of correctly recognised
%               casses of the secong class, 
%           Neg is the number of casses of the second class.
%       res(4) is true positive rate or TP/Pos
%       res(5) is true negative rate or TN/Neg
%       res(6) is Balanced accuracy
%       res(7) is true positive rate or TP/Pos for predefined threshold
%       res(8) is true negative rate or TN/Neg for predefined threshold
%       res(9) is Balanced accuracy for predefined threshold
%

    % Create Array for result
    if nargin == 5
        res = zeros(1, 9);
    else
        res = zeros(1, 6);
    end

    % Define numbers of cases
    Pos = length(x);
    Neg = length(y);
    res(1) = Pos;
    res(2) = Neg;

    % do we have both classes?
    if Pos == 0 || Neg == 0
        if Pos + Neg == 0
            return
        end
        if Pos > 0
            res(3) = min(x);
            res(3) = res(3) - 0.001 * abs(res(3));
            res(4) = 1;
            if nargin == 5
                res(7) = sum(x > prThres) / Pos;
            end
        else
            res(3) = max(y);
            res(3) = res(3) + 0.001 * abs(res(3));
            res(5) = 1;
            if nargin == 5
                res(8) = sum(y > prThres) / Neg;
            end
        end
        res(6) = 1;
        return;
    end
        
    % Define set of unique values
    thr = unique([x; y])';
    % Add two boders
    thr = [thr(1) - 0.0001 * abs(thr(1)), (thr(2:end) + thr(1:end - 1)) / 2,...
        thr(end) + 0.0001 * abs(thr(end))];
    accs = zeros(1, length(thr));
    
    % Define meaning of "class 1"
    xLt =  mean(x) > mean(y);
    
    % Define variabled to search
    bestAcc = 0;
    bestT = -Inf;
    bestTPR = 0;
    bestTNR = 0;
    %Check each threshold
    for k = 1:length(thr)
        t = thr(k);
        nX = sum(x < t);
        nY = sum(y >= t);
        if xLt
            nX = Pos - nX;
            nY = Neg - nY;
        end
        acc = (nX / Pos + nY / Neg) / 2;
        if acc > bestAcc
            bestAcc = acc;
            bestT = t;
            bestTPR = nX / Pos;
            bestTNR = nY / Neg;
        end
        accs(k) = acc;
    end

    res(3) = bestT;
    res(4) = bestTPR;
    res(5) = bestTNR;
    res(6) = bestAcc;

    if nargin == 5
        nX = sum(x < prThres);
        nY = sum(y >= prThres);
        if xLt
            nX = Pos - nX;
            nY = Neg - nY;
        end
        res(7) = nX / Pos;
        res(8) = nY / Neg;
        res(9) = (nX / Pos + nY / Neg) / 2;
    end

    % Form figure
    %Define min and max to form bines
    mi = min([x; y]);
    ma = max([x; y]);
    edges = mi:(ma-mi)/nBins:ma;

    %Draw histograms
    figure;
    nPos = histcounts(x,edges) / Pos;
    nNeg = histcounts(y,edges) / Neg;
    tmp = (edges(1:end-1) + edges(2:end)) / 2;
    plot(tmp, nPos);
    hold on;
    plot(tmp, nNeg);

    title(name);
    xlabel('Fisher''s discriminant projection');
    ylabel('Fraction of cases');

    %Draw graph of errors
    sizes = axis();
    plot(thr, accs * sizes(4), 'g');
    %Draw the best threshold
    plot([bestT, bestT], sizes(3:4), 'k', 'LineWidth', 2);
    if nargin == 5
        plot([prThres, prThres], sizes(3:4), 'm--', 'LineWidth', 2);
        legend('Correctly recognised', 'Wrongly recognised',...
            'Balanced accuracy * Ymax', 'Threshold', 'Predefined threshold',...
            'Location', 'southoutside');
    else
        legend('Correctly recognised', 'Wrongly recognised',...
            'Balanced accuracy * Ymax', 'Threshold',...
            'Location', 'southoutside');
    end
end

function saveFigures(fName, pos)
    set(gcf, 'PaperPositionMode', 'auto');
    if nargin > 1
        set(gcf, 'pos', pos);
    end
    [~, ~, ext] = fileparts(fName);
    ext = char(ext);
    if strcmpi(ext, '.eps')
        dr = '-depsc2';
    elseif strcmpi(ext, '.jpg')
        dr = '-djpeg';
    else
        dr = ['-d', char(ext(2:end))];
    end
    print(gcf, dr, '-noui', '-loose', fName);
end

function res = ROCDraw(x, y, xx, yy, name)
%ROCDraw formed one ROC Curve image with two ROC curves
%
% Inputs:
%   x contains values for the first class
%   y contains values for the second class
%   xx contains values for the first class for multiple clusters
%   yy contains values for the second class for multiple clusters
%   name is string with title of axis
%
% Outputs:
%   res is row vector with 6 values (9 if predefined threshold is
%       specified): 
%       res(1) Area under the curve for one cluster
%       res(2) Area under the curve for multiple clusters

    res = zeros(1, 2);
    % Create figure
    figure;
    % Calculate data for ROC Curve for one cluster
    [~, TPR, FPR, AUC] = calcROC(x, y);
    res(1) = AUC;
    plot(FPR, TPR, '-');
    hold on
    % Calculate data for ROC Curve for multiple clusters
    [~, TPR, FPR, AUC] = calcROC(xx, yy);
    res(2) = AUC;
    plot(FPR, TPR, '-');
    % Decoration
    title(name);
    xlabel('False positive rate');
    ylabel('True positive rate');
    legend("One cluster", "Multiple clusters", 'Location','southeast');
    % AUC
    text(0.5, 0.5, sprintf("AUC for one cluster is %5.3f", res(1)));
    text(0.5, 0.4, sprintf("AUC for multiple clusters is %5.3f", res(2)));
end

function [thr, TPR, FPR, AUC] = calcROC(x, y)
%ROCDraw formed one ROC Curve image with two ROC curves
%
% Inputs:
%   x contains values for the first class
%   y contains values for the second class
%
% Outputs:
%   thres is row vector of thresholds
%   TPR is row vector with TPR for each threshold
%   FPR is row vector with FPR for each threshold
%   AUC is Area under the curve

    % Define numbers of cases
    Pos = length(x);
    Neg = length(y);

    % Define set of unique values
    thr = unique([x(:); y(:)])';
    % Add two boders
    thr = [thr(1) - 0.0001 * abs(thr(1)), (thr(2:end) + thr(1:end - 1)) / 2,...
        thr(end) + 0.0001 * abs(thr(end))];

    TPR = zeros(1, length(thr));
    FPR = TPR;

    %Check each threshold
    for k = 1:length(thr)
        t = thr(k);
        TPR(k) = sum(x > t);
        FPR(k) = sum(y > t);
    end
    TPR = TPR / Pos;
    FPR = FPR / Neg;
    AUC = sum((TPR(1:end-1) + TPR(2:end)) .* (FPR(1:end-1) - FPR(2:end))) / 2;
end
