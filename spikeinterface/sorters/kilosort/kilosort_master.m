try
    useGPU = {useGPU};

    % prepare for kilosort execution
    addpath(genpath('{kilosort_path}'));

    % set file path
    fpath = '{output_folder}';

    % create channel map file
    run(fullfile('{channel_path}'));

    % Run the configuration file, it builds the structure of options (ops)
    run(fullfile('{config_path}'))

    % This part runs the normal Kilosort processing on the simulated data
    [rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
    rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
    rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)

    rez = merge_posthoc2(rez);
    fprintf('merge_posthoc2 error. Reporting pre-merge result\n');

    % save python results file for Phy
    rezToPhy(rez, fullfile(fpath));
catch
    fprintf('----------------------------------------');
    fprintf(lasterr());
    quit(1);
end
quit(0);



