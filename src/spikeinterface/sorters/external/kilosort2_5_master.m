function kilosort2_5_master(fpath, kilosortPath)
    try
        set(groot,'defaultFigureVisible', 'off');

        if ~isdeployed
            % prepare for kilosort execution
            addpath(genpath(kilosortPath));

            % add npy-matlab functions (copied in the output folder)
            addpath(genpath(fpath));
        end

        % Load channel map file
        load(fullfile(fpath, 'chanMap.mat'));

        % Load the configuration file, it builds the structure of options (ops)
        load(fullfile(fpath, 'ops.mat'));

        % NEW STEP TO SKIP KS PREPROCESSING
        if isfield(ops, 'skip_kilosort_preprocessing')
            skip_kilosort_preprocessing = ops.skip_kilosort_preprocessing;
        else
            skip_kilosort_preprocessing = 0;
        end

        if skip_kilosort_preprocessing
            % hack to skip the internal preprocessing
            % this mimic the preprocessDataSub() function
            fprintf("Skipping kilosort2.5 preprocessing\n");

            ops.nt0 	  = getOr(ops, {'nt0'}, 61); % number of time samples for the templates (has to be <=81 due to GPU shared memory)
            ops.nt0min  = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61)); % time sample where the negative peak should be aligned
            NT       = ops.NT ; % number of timepoints per batch
            NchanTOT = ops.NchanTOT; % total number of channels in the raw binary file, including dead, auxiliary etc
            bytes       = get_file_size(ops.fbinary); % size in bytes of raw binary
            nTimepoints = floor(bytes/NchanTOT/2); % number of total timepoints
            ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
            ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
            ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
            ops.twind = ops.tstart * NchanTOT*2; % skip this many bytes at the start
            [chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap); % function to load channel map file

            ops.igood = true(size(chanMap));
            ops.Nchan = numel(chanMap); % total number of good channels that we will spike sort
            ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan; % upper bound on the number of templates we can have

            rez.xc = xc; % for historical reasons, make all these copies of the channel coordinates
            rez.yc = yc;
            rez.xcoords = xc;
            rez.ycoords = yc;
            Nbatch      = ceil(ops.sampsToRead / ops.NT); % number of data batches
            ops.Nbatch = Nbatch;
            NTbuff      = ops.NT + 3*ops.ntbuff; % we need buffers on both sides for filtering

            rez.Wrot    = eye(ops.Nchan); % fake whitenning
            rez.temp.Nbatch = Nbatch;
            % fproc (preprocessed output) is the same as the fbinary (unprocessed input)
            % because we are skipping preprocessing.
            ops.fproc = ops.fbinary;

            rez.ops = ops; % memorize ops
            rez.ops.chanMap = chanMap;
            rez.ops.kcoords = kcoords;
            rez.ops.Nbatch = Nbatch;
            rez.ops.NTbuff = NTbuff;

            tic;  % tocs are supplied in other parts of KS code

        else
            % preprocess data to create temp_wh.dat
            rez = preprocessDataSub(ops);
        end

        % NEW STEP TO DO DATA REGISTRATION
        if isfield(ops, 'do_correction')
            do_correction = ops.do_correction;
        else
            do_correction = 1;
        end

        if do_correction
            fprintf("Drift correction ENABLED\n");
        else
            fprintf("Drift correction DISABLED\n");
        end

        rez = datashift2(rez, do_correction); % last input is for shifting data


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

        % save the motion vector. Done after rezToPhy because it delete the entire folder
        if do_correction
            writeNPY(rez.dshift, fullfile(fpath, 'motion.npy'))
        end

        % NEW STEP TO SAVE REZ TO MAT
        if isfield(ops, 'save_rez_to_mat')
            save_rez_to_mat = ops.save_rez_to_mat;
        else
            save_rez_to_mat = 0;
        end
        % save rez
        if save_rez_to_mat
            save(fullfile(fpath, 'rez.mat'), 'rez', '-v7')
        end

    catch
        fprintf('----------------------------------------');
        fprintf(lasterr());
        quit(1);
    end
    quit(0);
end
