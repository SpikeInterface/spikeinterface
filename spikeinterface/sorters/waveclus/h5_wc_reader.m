classdef h5_wc_reader < handle
    properties
        sr
        min_index
        max_segments
        raw_filename
        sr_infile
        t0_segments
        max_index
        segmentLength
        spikes_file
    end 
    methods
        function obj = h5_wc_reader(par, raw_filename)
            obj.max_segments = [];
            info = h5info(raw_filename);

            if ~any(strcmp({info.Datasets.Name},'sr')) || ~any(strcmp({info.Datasets.Name},'data'))
                error('Invalid h5 file.')
            end
            obj.raw_filename = raw_filename;
            obj.spikes_file = false;

            obj.sr_infile = true;
            obj.sr = double(h5read(raw_filename,'/sr'));

            data_info = h5info(raw_filename,'/data');
            lts = data_info.Dataspace.Size;
            obj.min_index = floor(par.tmin * obj.sr);
            if strcmp(par.tmax,'all')
                obj.max_index = lts;
            else
                obj.max_index = min(ceil(par.tmax * obj.sr),lts);
            end
            n = obj.max_index -  obj.min_index;

            %Segments the data in par.segments pieces
            obj.max_segments = ceil(n/ obj.sr / ...
                    (par.segments_length * 60));         %number of segments in which data is cutted

            obj.segmentLength = floor (n/obj.max_segments);

            obj.t0_segments = ones(1,obj.max_segments);
            obj.t0_segments(1) = (obj.min_index-1)/obj.sr*1000;
            for i = 2:obj.max_segments
                obj.t0_segments(i) = obj.t0_segments(i-1) + obj.segmentLength/obj.sr*1000;
            end
        end

        function [sr,max_segments,with_raw,with_spikes] = get_info(obj)
            if isempty(obj.max_segments)
                with_raw = false;
                max_segments = 0;
            else
                with_raw = true;
                max_segments = obj.max_segments;
            end

            with_spikes = obj.spikes_file;
            if obj.sr_infile
                sr = obj.sr;
            else
                sr = [];
            end
        end

        function index_ts = index2ts(obj,index,i)
            index_ts = (index)/obj.sr*1000 + obj.t0_segments(i);
        end

        function x = get_segment(obj,i)
            if i ~= obj.max_segments
                x = h5read(obj.raw_filename,'/data',obj.min_index+obj.segmentLength*(i-1)+1 , obj.min_index+obj.segmentLength*i-obj.min_index-obj.segmentLength*(i-1));
            else
                x = h5read(obj.raw_filename,'/data',obj.min_index+obj.segmentLength*(i-1)+1 , obj.max_index-obj.min_index-obj.segmentLength*(i-1));
            end
        end
    end
end
