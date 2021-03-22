try
    % prepare for kilosort execution
    addpath(genpath('{kilosort2_5_path}'));

    % set file path
    fpath = '{output_folder}';

    % add npy-matlab functions (copied in the output folder)
    addpath(genpath(fpath));

    % create channel map file
    run(fullfile('{channel_path}'));

    % Run the configuration file, it builds the structure of options (ops)
    run(fullfile('{config_path}'))

    ops.trange = [0 Inf]; % time range to sort

    % preprocess data to create temp_wh.dat
    rez = preprocessDataSub(ops);

    % NEW STEP TO DO DATA REGISTRATION
    rez = datashift2(rez, 1); % last input is for shifting data

    % ORDER OF BATCHES IS NOW RANDOM, controlled by random number generator
    iseed = 1;

    % main tracking and template matching algorithm
    rez = learnAndSolve8b(rez, iseed);

    % OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
    % See issue 29: https://github.com/MouseLand/Kilosort/issues/29
    %rez = remove_ks2_duplicate_spikes(rez);

    % final merges
    rez = find_merges(rez, 1);

    % final splits by SVD
    rez = splitAllClusters(rez, 1);

    % decide on cutoff
    rez = set_cutoff(rez);
    % eliminate widely spread waveforms (likely noise)
    rez.good = get_good_units(rez);

    fprintf('found %d good units \n', sum(rez.good>0))

    % write to Phy
    fprintf('Saving results to Phy  \n')
    rezToPhy(rez, fullfile(fpath));
catch
    fprintf('----------------------------------------');
    fprintf(lasterr());
    quit(1);
end
quit(0);
