function res = createErrorCorrection(CR, WR, varargin)
%createErrorCorrection creates a classifier to identify data points
%corresponding to erroneous or otherwise other undesirable
%outputs/instances of behaviour. 
%The application of the classifier enables to filter these undesirable
%points or highlight/communicate their presence.
%
% Inputs:
%   CR (correctly recognised) is N-by-D matrix with N examples of correctly
%       recognised records. It can be true positive, true negative and
%       mixture of this two sets. It also can be set of points for
%       regression problem for which deviation of predicted value from
%       observed is appropriate (small enough). D MUST be the same for CR
%       and WR. To specify training (learning) and test sets this matrix
%       should be empty.
%   WR (wrongly recognised) is M-by-D matrix with M examples of wrongly
%       recognised records. It can be false positive, false negative and
%       mixture of this two sets. It also can be set of points for
%       regression problem for which deviation of predicted value from
%       observed is inappropriate (too big). D MUST be the same for CR and
%       WR. To specify training (learning) and test sets this matrix
%       should be empty.
%   Optional Name Value pairs for dataset deffining and spliting
%       'fTS' is is fraction of CR and fraction of WR sets which will be
%           used as training set. This value must be greater than zero and
%           less than one. Default value is 0.5 (50%). This value used to
%           split randomly CR into CRLS (learning set of correctly
%           recognised) and CRTS (test set of correctly recognised) and WR
%           into WRLS (learning set of wrongly recognised) and WRTS (test
%           set of wrongly recognised).
%       'CRLS' is NL-by-D matrix with training set for correctly recognised
%           data. D MUST be the same for all specified matrix CR, CRLS,
%           CRTS, WR, WRLS, WRTS. Matrices CRLS and CRTS must be both
%           specified or omitted. This matrix will be taken into account
%           only for empty WR.
%       'CRTS' is NT-by-D matrix with training set for correctly recognised
%           data. D MUST be the same for all specified matrix CR, CRLS,
%           CRTS, WR, WRLS, WRTS.  Matrices CRLS and CRTS must be both
%           specified or omitted. This matrix will be taken into account
%           only for empty CR.
%       'WRLS' is NL-by-D matrix with training set for wrongly recognised
%           data. D MUST be the same for all specified matrix CR, CRLS,
%           CRTS, WR, WRLS, WRTS. Matrices CRLS and CRTS must be both
%           specified or omitted. This matrix will be taken into account
%           only for empty WR.
%       'WRTS' is NT-by-D matrix with training set for wrongly recognised
%           data. D MUST be the same for all specified matrix CR, CRLS,
%           CRTS, WR, WRLS, WRTS. Matrices WRLS and WRTS must be both
%           specified or omitted. This matrix will be taken into account
%           only for empty WR.
%
%   Optional Name Value pairs for space selection and centralisation
%       'Space' can have one of two values:
%           'Original' means search of clusters and Fisher's discriminant
%               in original space. In this case all arguments related to
%               dimensionality reduction are ignored exclude 'PCDataset'.
%           'Reduced' means search of clusters  and Fisher's discriminant
%               in reduced space. 
%           Default value is 'Reduced'.
%       'centre' is vector to use for data centering. Default value is [].
%           If argument is omitted then this vector is calculated as mean
%           of dataset used for projector calculation (see 'PCDataset').
%
%   Optional Name Value pairs for dimensionality reduction. Dimensionality
%       reduction is implemented as subtraction of 'centre' and projection
%       onto subspace defined by column vectors in matrix 'project': 
%           X = (X - centre) * project;
%       'project' is D-by-numPC matrix with basis of reduced space in
%           columns. Each column will be normalised to unit length. Default
%           value is []. If argument is omitted then this matrix will be
%           completed by PCs of specified dataset (see 'PCDataset').
%       'PCDataset' is dataset used to calculate principle components.
%           There are three possible value:
%           'All': PCs should be calculated on base of CRLS and WRLS
%           'CR': PCs should be calculated on base of CRLS
%           'WR': PCs should be calculated on base of WRLS
%           Default value is 'CR'.
%       'numPC' is number of PCs to use. Default value is 200.
%           positive integer is number of PCS to use. 
%           0 means that number of PCs will be defined by Kaiser rule (all
%               PC with eigenvalue greater than average).
%           -1 means that number of PCs will be defined by Broken stick
%               rule.
%           other negative values means usage of conditional number rule:
%               all PCs with eigen values greater than maximal eigenvalue
%               divided by -numPCA will be considered as informative.
%           if project is specified then numPC is number of columns in
%               marix project.
%       'Whitening' can be true (whitening) or false (without whitening).
%           Default value true.
%
%   Optional Name Value pairs for clustering
%       'numClusters' is number of clusters to use. It must be positive
%           integer value. Default value is 100
%
% Outputs:
%   res is structure with following fields:
%        CRLS is matrix m1-by-D which is training set for correctly
%           recognised cases. 
%        CRTS is matrix m2-by-D which is test set for correctly recognised
%           cases. 
%        WRLS is matrix m3-by-D which is training set for wrongly
%           recognised cases. 
%        WRTS is matrix m4-by-D which is test set for wrongly recognised
%           cases.
%        centre	is vector to subtract from data for centralisation. For PC
%           projection it is mean vector of training set. 
%        space can have one of two values:
%           'Original' means search of clusters and Fisher's discriminant
%               in original space. In this case all arguments related to
%               dimensionality reduction are ignored exclude 'PCDataset'.
%           'Reduced' means search of clusters  and Fisher's discriminant
%               in reduced space. 
%        project depends on value of 'space': 
%           if 'space' is 'reduced' then project is D-by-numPC matrix with
%               basis of reduced (low dimensional) space. For PC projection
%               it is matrix with specified number of PCs in columns. If
%               data whitened then projection vectors (PCs) are whitened.
%           if 'space' is 'original' then project is 1-by-D vector of
%               standard deviations calculated for subset specified by
%               'PCDataset' for each attribute for whitening and vector of
%               ones if whitening is not required. 
%        centroids is numClust-by-D or numClust-by-numPC matrix contains 
%           centroids of clusters in corresponding 'space'.
%        centroids1 is row vector with D or numPC elements. This vector is 
%           centroid of one cluster in corresponding 'space'.
%        DF1 is column vector with D or numPC elements which is Fisher's
%           discriminant direction in corresponding 'space' for one cluster.
%        DF is D-by-numClust or or numPC-by-numClust matrix which contains 
%           Fisher's discriminant directions in corresponding 'space' for 
%           all clusters.
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


    % Default values for optional arguments
    fTS = 0.5;
    CRLS = [];
    CRTS = [];
    WRLS = [];
    WRTS = [];
    setPC = 2;
    centre = [];
    project = [];
    whitening = false;
    space = false;
    numClust = 100;
    numPC = 200;
    debugInfo = [];
    % Check optional arguments
    for k = 1:2:length(varargin)
        s = varargin{k};
        if ~(isstring(s) || ischar(s))
            error("Inacceptable type of input argument %s", s);
        end
        tmp = varargin{k + 1};
        if strcmpi(s, 'fTS')
            fTS = tmp;
            if ~isnumeric(fTS) || fTS <= 0 || fTS >= 1
                error("Value of fTS must be between 0 and 1 (exclusevely). Value %s is inappropriate", fTS);
            end
        elseif strcmpi(s, 'CRLS')
            if ~isnumeric(tmp) || ~ismatrix(tmp)
                error("Value for 'CRLS' must be matrix with training set of correctly recognised data");
            end
            CRLS = tmp;
        elseif strcmpi(s, 'CRTS')
            if ~isnumeric(tmp) || ~ismatrix(tmp)
                error("Value for 'CRTS' must be matrix with test set of correctly recognised data");
            end
            CRTS = tmp;
        elseif strcmpi(s, 'WRLS')
            if ~isnumeric(tmp) || ~ismatrix(tmp)
                error("Value for 'CRLS' must be matrix with training set of wrongly recognised data");
            end
            WRLS = tmp;
        elseif strcmpi(s, 'WRTS')
            if ~isnumeric(tmp) || ~ismatrix(tmp)
                error("Value for 'CRTS' must be matrix with test set of wrongly recognised data");
            end
            WRTS = tmp;
        elseif strcmpi(s, 'centre')
            if ~isnumeric(tmp) || ~isvector(tmp)
                error("Value for 'centre' must be vector for data centering.");
            end
            centre = tmp;
        elseif strcmpi(s, 'project')
            if ~isnumeric(tmp) || ~ismatrix(tmp)
                error("Value for 'project' must be matrix of basis of reduced space in columns");
            end
            project = tmp;
        elseif strcmpi(s, 'PCDataset')
            switch lower(tmp)
                case 'all'
                    setPC = 1;
                case 'cr'
                    setPC = 2;
                case 'wr'                    
                    setPC = 3;
                otherwise
                    error("For 'PCDataset' appropriate value is one of the following: 'All', 'CR', 'WR'. Value '%s' is not appropriate.", tmp);
            end
        elseif strcmpi(s, 'numPC')
            if ~isnumeric(tmp)
                error("Argument 'numPC' must be numeric");
            end
            numPC = tmp;
        elseif strcmpi(s, 'whitening')
            if islogical(tmp)
                whitening = tmp;
            elseif isnumeric(tmp)
                whitening = tmp ~= 0;
            else
                error("Value %s is inacceptable for argument 'whitening'. It must be true (whitening) or false (without whitening).", tmp);
            end
        elseif strcmpi(s, 'Space')
            switch lower(tmp)
                case 'original'
                    space = true;
                case 'reduced'
                    space = false;
                otherwise
                    error("For 'ClusteringSpace' appropriate value is 'Original' or 'Reduced'. Value '%s' is not appropriate.", tmp);
            end
        elseif strcmpi(s, 'numClust')
            if ~isnumeric(tmp) || tmp < 1
                error("Argument 'numClust' must be positive integer. Value %s is inacceptable", tmp);
            end
            numClust = round(tmp);
%         elseif strcmpi(s, 'debug') % These two rows are saved for debugging only
%             debugInfo = tmp;
        else
            error("Inacceptable argument '%s'", s)
        end
    end

    % Check dimensions specification of all matrix
    tmp = "The second dimension of all specified matrix CR, CRLS, CRTS, WR, WRLS, WRTS must be the same.";
    if isempty(CR)
        if isempty(CRTS) || isempty(CRLS)
            error("If matrix CR is omitted then both CRTS and CRLS must be specified");
        end
        D = size(CRLS, 2);
        if D ~= size(CRTS)
            error(tmp)
        end
    else
        D = size(CR, 2);
    end
    if isempty(WR)
        if isempty(WRTS) || isempty(WRLS)
            error("If matrix WR is omitted then both WRTS and WRLS must be specified");
        end
        if D ~= size(WRLS, 2) || D ~= size(WRTS, 2)
            error(tmp)
        end
    else
        if size(WR, 2) ~= D
            error(tmp)
        end
    end
    % Split datasets if necessary
    if ~isempty(CR)
        ind = rand(size(CR, 1), 1) > fTS;
        CRLS = CR(ind, :);
        CRTS = CR(~ind, :);
    end
    if ~isempty(WR)
        ind = rand(size(WR, 1), 1) > fTS;
        WRLS = WR(ind, :);
        WRTS = WR(~ind, :);
    end
    % Save all sets to res
    res.CRLS = CRLS;
    res.CRTS = CRTS;
    res.WRLS = WRLS;
    res.WRTS = WRTS;

    % Centre identification
    if isempty(centre)
        % Calculate central point, centralise data and calculate Principal components
        switch setPC
            case 1
                centre = mean([CRLS; WRLS]);
            case 2
                centre = mean(CRLS);
            case 3
                centre = mean(WRLS);
        end
    else
        if length(centre) ~= D
            error("Number of elements in vector 'centre' must be the same as number of column in all data matrices");
        end
        centre = (centre(:))';
    end
    res.centre = centre;
    CRLS = CRLS - res.centre;
    WRLS = WRLS - res.centre;
    if space
        res.space = 'Original';
    else
        res.space = 'Reduced';
    end

    if space
        if whitening
            switch setPC
                case 1
                    project = std([CRLS; WRLS]);
                case 2
                    project = std(CRLS);
                case 3
                    project = std(WRLS);
            end
        else
            project = ones(1, D);
        end
        res.project = project;
        % Rescale (whitening) data
        CRLSR = CRLS ./ project;
        WRLSR = WRLS ./ project;
    else
        % If we used reduced space
        % Calculate PCs if it is necessary
        if isempty(project)
            switch setPC
                case 1
                    [project, ~, ev] = pca([CRLS; WRLS], 'Centered',false);
                case 2
                    [project, ~, ev] = pca(CRLS, 'Centered',false);
                case 3
                    [project, ~, ev] = pca(WRLS, 'Centered',false);
            end
            % Define number of PCs
            if numPC > 0
                numPC = round(numPC);
                if numPC > size(CRLS, 1)
                    error("Requested number of PCs %d is greater than dimension of space %d.", numPC, size(CRLS, 1));
                end
            elseif numPC == 0
                numPC = sum(ev > mean(ev));

            elseif numPC == -1
                numPC = brokenStick(ev);
            else
                numPC = sum(ev > (ev(1) / (-numPC)));
            end
            % Apply whitening, if necessary
            if whitening
                project = project(:, 1:numPC) * diag(1 ./ sqrt(ev(1:numPC)));
            else
                project = project(:, 1:numPC);
            end
        else
            % Check the dimension
            if size(project, 1) ~= D
                error("Number of rows in matrix 'project' must be the same as number of column in all data matrices");
            end
            % Just in case normalise columns in project
            numPC = size(project, 2);
            project = bsxfun(@rdivide, project, sqrt(sum(project .^ 2)));
            % Apply whitening, if necessary
            if whitening
                % Calculate preliminary projection for pseudo whitening
                CRLSR = CRLS * project;
                WRLSR = WRLS * project;
                % Calculate variance of projections
                switch setPC
                    case 1
                        stds = std([CRLSR; WRLSR]);
                    case 2
                        stds = std(CRLSR);
                    case 3
                        stds = std(WRLSR);
                end
                project = project * diag(1 ./ sqrt(stds));
            end
        end
        res.project = project;
        % Calculate projections of training sets to project
        CRLSR = CRLS * project;
        WRLSR = WRLS * project;
    end
    
    % Clustering
    % Form data for one cluster 
    % Centroid of one cluster is vector of zeros for reduced space and
    % for original space because we use cetered data only!
    res.centroids1 = zeros(1, size(WRLSR, 2));
    if numClust > 1
% if isempty(debugInfo)
        % Reduced space. Search clusters and store centroids
        [labs, centroids] = kmeans(WRLSR, numClust);
% else  % These lines were used to reproduce the same clusters. It was necessary for debugging only.
%     labs = debugInfo.ind;
%     centroids = debugInfo.centres;
% end
        res.centroids = centroids;
        % Remove empty clusters
        for k = numClust:-1:1
            tmp = sum(labs == k);
            if tmp == 0
                res.centroidsReduced(k, :) = [];
                res.centroids(k, :) = [];
                labs(labs > k) = labs(labs > k) - 1;
            end
        end
        % Recalculate number of clusters
        numClust = size(res.centroids, 1);
    else
        % There is no clusters
        % Centroid of one cluster is vector of zeros for reduced space and
        % for original space because we use cetered data only!
        res.centroids = res.centroid1;
    end

    % Form Fisher's Discriminant (FDDirect) directions
    % Auxiliary values for all calculations
    covCRLSR = cov(CRLSR);
    meanCRLSR = mean(CRLSR);
    % Firstly form for one cluster
    res.FD1 = ((covCRLSR + cov(WRLSR)) \ (meanCRLSR - mean(WRLSR))');
    % Should be perhaps removed
    res.FD1 = res.FD1 / sqrt(sum(res.FD1 .^ 2)); 
    % Now consider clusters
    if numClust > 1
        res.FD = zeros(numPC, numClust);
        for k = 1:numClust
            % Get required fragment of WRLSR
            tmp = WRLSR(labs == k, :);
            if size(tmp, 1) == 1
                cov1 = 0;
            else
                cov1 = cov(tmp);
            end
            res.FD(:, k) = (covCRLSR + cov1) \ (meanCRLSR - mean(tmp, 1))';
        end
        res.FD = res.FD ./ sqrt(sum(res.FD .^ 2));
    else
        res.FD = res.FD1;
    end
end
