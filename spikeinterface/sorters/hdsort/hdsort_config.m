% Set HDsort parameters

P = struct();

% preprocess
P.filter = {filter};
P.parfor = {parfor};
P.hpf = {hpf};
P.hpf = {lpf};

% leg creation
P.legs.maxElPerGroup = {max_el_per_group};
P.legs.minElPerGroup = {min_el_per_group};
P.legs.addIfNearerThan = {add_if_nearer_than}; % always add direct neighbors
P.legs.maxDistanceWithinGroup = {max_distance_within_group};

% spike detection
P.spikeDetection.method = '-';
P.spikeDetection.thr = {detect_threshold};
P.artefactDetection.use = 0;

% pre-clustering
P.noiseEstimation.minDistFromSpikes = 80;
P.spikeAlignment.initAlignment = '-';
P.spikeAlignment.maxSpikes = 50000;     % so many spikes will be clustered
P.featureExtraction.nDims = {n_pc_dims}; %6
P.clustering.maxSpikes = 50000;  % dont align spikes you dont cluster...
P.clustering.meanShiftBandWidthFactor = 1.8;
%P.clustering.meanShiftBandWidth = sqrt(1.8*6); % todo: check this!

% template matching
P.botm.run = 0;
P.botm.Tf = 75;
P.botm.cutLeft = 20;
P.spikeCutting.maxSpikes = 200000000000; % Set this to basically inf
P.spikeCutting.blockwise = false;
P.templateEstimation.cutLeft = 10;
P.templateEstimation.Tf = 55;
P.templateEstimation.maxSpikes = 100;

% merging
P.mergeTemplates.merge = 1;
P.mergeTemplates.upsampleFactor = 3;
P.mergeTemplates.atCorrelation = .93; % DONT SET THIS TOO LOW! USE OTHER ELECTRODES ON FULL FOOTPRINT TO MERGE
P.mergeTemplates.ifMaxRelDistSmallerPercent = 30;
