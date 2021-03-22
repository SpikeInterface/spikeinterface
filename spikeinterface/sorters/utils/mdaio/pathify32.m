function X = pathify32(X)
% PATHIFY32   if array, write to an MDA & give path, otherwise leave as path.
%
% X = pathify32(X) uses single-precision float MDA files.

% Barnett 6/17/16

if isnumeric(X)
  dir = [tempdir,'/mountainlab/tmp_short_term'];
  if ~exist(dir,'dir'), mkdir(dir); end     % note can handle creation of parents
  fname = [dir,'/',num2str(randi(1e15)),'.mda'];  % random filename
  writemda(X,fname,'float32');
  X = fname;
end
