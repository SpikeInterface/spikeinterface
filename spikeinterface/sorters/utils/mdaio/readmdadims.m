function S=readmdadims(fname)
%READMDADIMS - read only dimensions of a .mda file. MDA stands for multi-dimensional array.
%
% See http://magland.github.io//articles/mda-format/
%
% Syntax: dims=readmdadims(fname)
%
% Inputs:
%    fname - path to the .mda file
%
% Outputs:
%    dims - row vector of dimension sizes of multi-dimensional array
%
% Other m-files required: none
%
% See also: writemda

% Author: Alex Barnett 7/22/16

F=fopen(fname,'rb');

try
code=fread(F,1,'int32');
catch
    error('Problem reading file: %s',fname);
end
if (code>0) 
    num_dims=code;
    code=-1;
else
    fread(F,1,'int32');
    num_dims=fread(F,1,'int32');    
end;
dim_type_str='int32';
if (num_dims<0)
    num_dims=-num_dims;
    dim_type_str='int64';
end;

S=zeros(1,num_dims);
for j=1:num_dims
    S(j)=fread(F,1,dim_type_str);
end;

fclose(F);