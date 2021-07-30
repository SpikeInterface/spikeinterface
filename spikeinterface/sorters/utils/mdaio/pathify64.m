function X = pathify64(X)
% PATHIFY64   if array, write to an MDA & give path, otherwise leave as path.
%
% X = pathify64(X) uses double-precision float MDA files.

% Barnett 6/17/16

if isnumeric(X)
  dir = [tempdir,'/mountainlab/tmp_short_term'];
  if ~exist(dir,'dir'), mkdir(dir); end     % note can handle creation of parents
  fname = [dir,'/',num2str(randi(1e15)),'.mda'];  % random filename
  writemda(X,fname,'float64');
  X = fname;
end
