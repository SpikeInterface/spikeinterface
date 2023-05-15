function waveclus_snippets_master(outputFolder, waveclusPath)
    if ~isdeployed
        addpath(genpath(waveclusPath));
    end

    % par.mat file should contain a variable called par_input
    load(fullfile(outputFolder, 'par_input.mat'));

    % Default parameters
    S_par = set_parameters_ss();

    % Update with custom parameters
    S_par = update_parameters(S_par, par_input, 'relevant');

    try
        cd(outputFolder);
		vcFile_spikes = fullfile(outputFolder, 'results_spikes.mat');
		vcFile_cluster = fullfile(outputFolder, 'times_results.mat');
        Do_clustering(vcFile_spikes, 'make_plots', false,'save_spikes',false,'par',S_par);

        if ~exist(vcFile_cluster,'file')
            load(vcFile_spikes,'par');
            par = update_parameters(S_par, par, 'relevant');
            cluster_class = zeros(0,2);
            save(vcFile_cluster,'cluster_class','par');
        end
    catch
        fprintf('----------------------------------------');
        fprintf(lasterr());
        quit(1);
    end
    quit(0);
end


function par = set_parameters_ss()

    % SPC PARAMETERS
    par.mintemp = 0.00;                  % minimum temperature for SPC
    par.maxtemp = 0.251;                 % maximum temperature for SPC
    par.tempstep = 0.01;                 % temperature steps
    par.SWCycles = 100;                  % SPC iterations for each temperature (default 100)
    par.KNearNeighb = 11;                % number of nearest neighbors for SPC
    par.min_clus = 20;                   % minimum size of a cluster (default 60)
    par.max_clus = 200;                   % maximum number of clusters allowed (default 200)
    par.randomseed = 0;                  % if 0, random seed is taken as the clock value (default 0)
    %par.randomseed = 147;               % If not 0, random seed
    %par.temp_plot = 'lin';              % temperature plot in linear scale
    par.temp_plot = 'log';               % temperature plot in log scale

    par.c_ov = 0.7;                      % Overlapping coefficient to use for the inclusion criterion.
    par.elbow_min  = 0.4;                %Thr_border parameter for regime border detection.

    % FEATURES PARAMETERS
    par.min_inputs = 10;         % number of inputs to the clustering
    par.max_inputs = 0.75;       % number of inputs to the clustering. if < 1 it will the that proportion of the maximum.
    par.scales = 4;                        % number of scales for the wavelet decomposition
    par.features = 'wav';                % type of feature ('wav' or 'pca')
    %par.features = 'pca'


    % FORCE MEMBERSHIP PARAMETERS
    par.template_sdnum = 3;             % max radius of cluster in std devs.
    par.template_k = 10;                % # of nearest neighbors
    par.template_k_min = 10;            % min # of nn for vote
    %par.template_type = 'mahal';       % nn, center, ml, mahal
    par.template_type = 'center';       % nn, center, ml, mahal
    par.force_feature = 'spk';          % feature use for forcing (whole spike shape)
    %par.force_feature = 'wav';         % feature use for forcing (wavelet coefficients).
    par.force_auto = true;              %automatically force membership (only for batch scripts).

    % TEMPLATE MATCHING
    par.match = 'y';                    % for template matching
    %par.match = 'n';                   % for no template matching
    par.max_spk = 40000;                % max. # of spikes before starting templ. match.
    par.permut = 'y';                   % for selection of random 'par.max_spk' spikes before starting templ. match.
    % par.permut = 'n';                 % for selection of the first 'par.max_spk' spikes before starting templ. match.
end
