function p_waveclus(vcDir_temp, nChans, par_input)
    % Arguments
    % -----
    % nChans: number of channels, if more than 1 the polytrode
    % par_input: wave_clus parameters defined by the user
    % version of wave_clus will be applied

    S_par = set_parameters_ss();
    S_par = update_parameters(S_par,par_input,'relevant');

    cd(vcDir_temp);

    for nch = 1: nChans
        vcFile_mat{nch} = fullfile(vcDir_temp, ['raw' int2str(nch) '.mat']);
    end
    if nChans==1
        % Run waveclus batch mode. supply parameter file (set sampling rate)
        Get_spikes(vcFile_mat{1}, 'par', S_par);
        vcFile_spikes = strrep(vcFile_mat{1}, '.mat', '_spikes.mat');
        Do_clustering(vcFile_spikes, 'make_plots', false,'save_spikes',false);
        [vcDir_, vcFile_, vcExt_] = fileparts(vcFile_mat{1});
        vcFile_cluster = fullfile(vcDir_, ['times_', vcFile_, vcExt_]);
    else
        % Run waveclus batch mode. supply parameter file (set sampling rate)
        pol_file = fopen('polytrode1.txt','w');
        cellfun(@(x) fprintf(pol_file ,'%s\n',x),vcFile_mat);
        fclose(pol_file);
        Get_spikes_pol(1, 'par', S_par);
        Do_clustering('polytrode1_spikes.mat', 'make_plots', false,'save_spikes',false);
        [vcDir_, ~, vcExt_] = fileparts(vcFile_mat{1});
        vcFile_cluster = fullfile(vcDir_, ['times_polytrode1', vcExt_]);
    end
    
    movefile(vcFile_cluster,fullfile(vcDir_, 'times_results.mat'),'f')
end