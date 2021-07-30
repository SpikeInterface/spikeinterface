function A=readmda_block(fname,index0,size0)
%READMDA - read a subarray from an .mda file. 
%
% Syntax: A=readmda_block(fname,index0,size0)
%
% Inputs:
%    fname - path to the .mda file
%    index0 - the array index, e.g., [1,1,1000], to start reading
%    size0 - the size of the array to use
%    The length of index0 and size0 should equal the number of dimensions
%    of the array represented by the .mda file
%
% Outputs:
%    A - the multi-dimensional array
%
% Other m-files required: none
%
% See also: readmda

% Author: Jeremy Magland
% Jan 2015; Last revision: 15-Feb-2106

if (nargin<1)
    test_readmda_block;
    return;
end;

if (strcmp(fname(end-4:end),'.csv')==1)
    warning('Case of .csv file not supported for readmda_block');
    return;
end

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

if num_dims == 1,
  num_dims=2;
  S=[1,S(1)];
end

% Remove singleton dimensions at end until length(index0) equals num_dims
while ((num_dims>2)&&(num_dims>length(index0))&&(S(end)==1))
    S=S(1:end-1);
    num_dims=num_dims-1;
end;

% check that we can handle the situation
for j=1:num_dims-1
    if (index0(j)~=1)||(size0(j)~=S(j))
        disp(index0);
        disp(size0);
        error('Cannot yet handle this case when reading block from %s',fname);
    end;
end;

A=zeros(size0);
N=prod(size0);
seek_size=prod(S(1:end-1))*(index0(end)-1);

if (code==-1)
    fseek(F,seek_size*8,'cof');
    M=zeros(1,N*2);
    M(:)=fread(F,N*2,'float');
    A(:)=M(1:2:N*2)+i*M(2:2:N*2);
elseif (code==-2)
    fseek(F,seek_size*1,'cof');
    A(:)=fread(F,N,'uchar');
elseif (code==-3)
    fseek(F,seek_size*4,'cof');
    A(:)=fread(F,N,'float');
elseif (code==-4)
    fseek(F,seek_size*2,'cof');
    A(:)=fread(F,N,'int16');
elseif (code==-5)
    fseek(F,seek_size*4,'cof');
    A(:)=fread(F,N,'int32');
elseif (code==-6)
    fseek(F,seek_size*2,'cof');
    A(:)=fread(F,N,'uint16');
elseif (code==-7)
    fseek(F,seek_size*8,'cof');
    A(:)=fread(F,N,'double');
elseif (code==-8)
    fseek(F,seek_size*4,'cof');
    A(:)=fread(F,N,'uint32');
else
    error('Unsupported data type code: %d',code);
end;

fclose(F);

function test_readmda_block
X=rand(10,20,30);
writemda(X,'tmp.mda');
Y=readmda_block('tmp.mda',[1,1,5],[10,20,10]);
tmp=X(:,:,5:5+10-1)-Y;
fprintf('This should be small: %g\n',max(abs(tmp(:))));

X2=floor(X*1000);
writemda16ui(X2,'tmp.mda');
Y=readmda_block('tmp.mda',[1,1,5],[10,20,10]);
tmp=X2(:,:,5:5+10-1)-Y;
fprintf('This should be zero: %g\n',max(abs(tmp(:))));

