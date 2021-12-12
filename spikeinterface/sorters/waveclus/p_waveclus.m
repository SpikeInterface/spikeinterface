function p_waveclus(vcDir_temp, par_input)
    % Arguments
    % -----
    % nch: number of channels, if more than 1 the polytrode
    % par_input: wave_clus parameters defined by the user
    % version of wave_clus will be applied

    S_par = set_parameters_ss();
    S_par = update_parameters(S_par,par_input,'relevant');

    cd(vcDir_temp);
    nch = 0;
    while true
        aux_filename = fullfile(vcDir_temp, ['raw' int2str(nch+1) '.h5']);
        if exist(aux_filename,'file')
            nch = nch +1;
            vcFile_mat{nch} = aux_filename;
        else
            break
        end
    end
    if nch==1
        % Run waveclus batch mode.
        Get_spikes(vcFile_mat{1}, 'par', S_par);
        vcFile_spikes = strrep(vcFile_mat{1}, '.h5', '_spikes.mat');
        Do_clustering(vcFile_spikes, 'make_plots', false,'save_spikes',false);
        [vcDir_, vcFile_, ~] = fileparts(vcFile_mat{1});
        vcFile_cluster = fullfile(vcDir_, ['times_', vcFile_, '.mat']);
    else
        % Run waveclus batch mode.
        pol_file = fopen('polytrode1.txt','w');
        cellfun(@(x) fprintf(pol_file ,'%s\n',x),vcFile_mat);
        fclose(pol_file);
        Get_spikes_pol(1, 'par', S_par);
        vcFile_spikes = 'polytrode1_spikes.mat';
        Do_clustering(vcFile_spikes, 'make_plots', false,'save_spikes',false);
        [vcDir_, ~, ~] = fileparts(vcFile_mat{1});
        vcFile_cluster = fullfile(vcDir_, ['times_polytrode1', '.mat']);
    end
    newfile = fullfile(vcDir_, 'times_results.mat');
    if exist(vcFile_cluster,'file')
        movefile(vcFile_cluster,newfile,'f');
    else
        load(vcFile_spikes,'par');
        par = update_parameters(S_par, par, 'relevant');
        cluster_class = zeros(0,2);
        save(newfile,'cluster_class','par');
    end
end