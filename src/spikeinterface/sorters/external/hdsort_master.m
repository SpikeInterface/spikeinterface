function hdsort_master(outputFolder, hdsortPath)
    try
        if ~isdeployed
            % prepare for hdsort matlab execution
            addpath(genpath(hdsortPath));
        else
            % Running in compiled mode:
            % cding to outputFolder is needed to save output files in the correct place
            cd(fullfile(outputFolder))
        end

        %% Load sorter configs and params
        load(fullfile(outputFolder, 'configsParams.mat'))

        if (strcmp(fileFormat, 'maxwell'))
            RAW = hdsort.file.MaxWellFile(rawFile);
        elseif (strcmp(fileFormat, 'mea1k'))
            RAW = hdsort.file.BELMEAFile(rawFile);
        end

        RAW.restrictToConnectedChannels(); % This line is very important to not sort empty electrodes!

        %% Create the object that performs the sorting:
        mainFolder = '.';
        HDSorting = hdsort.Sorting(RAW, mainFolder, sortingName)

        %% Preprocess:
        % The preprocessor loads data from the file in chunks, filters it, and saves
        % the filtered data into a new hdf5 file that is standardized for all types
        % of input data.
        % It further performs a couple of operations for each local electrode group
        % (LEG) such as spike detection, spike waveform cutting and noise
        % estimation. The result is a folder named group000x for each LEG that
        % contains the data necessary to perform the parallel parts of the sorting.
        % This implementation minimizes the number of time-consuming file access
        % operations and thus speeds up the parallel processes significantly, even
        % allowing the parallel processes to be run on a small desktop computer or
        % laptop.

        chunkSize = chunkSize; % This number depends a lot on the available RAM
        HDSorting.preprocess('chunkSize', chunkSize, 'forceFileDeletionIfExists', true);

        %% Sort each LEG independently:
        HDSorting.sort('sortingMode', loopMode); % (default)
        % Alternative sorting modes are:
        % HDSorting.sort('sortingMode', 'local'); % for loop over each LEG
        % HDSorting.sort('sortingMode', 'grid'); % requires a computer grid architecture

        %% Combine the results of each LEG in the postprocessing step:
        HDSorting.postprocess()

        %% Export the results in an easy to read format:
        [sortedPopulation, sortedPopulation_discarded] = HDSorting.createSortedPopulation(mainFolder);

        %% When the sorting has already been run before, open the results from a file with:
        % sortedPopulation = hdsort.results.Population(HDSorting.files.results);
    catch
        fprintf('----------------------------------------');
        fprintf(lasterr());
        quit(1);
    end
    quit(0);
end
