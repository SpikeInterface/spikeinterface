function X = arrayify(X)
% ARRAYIFY - if a string, read in from file, otherwise leave as an array
%
% X = arrayify(X)

% Barnett 6/16/16

if ischar(X), X = readmda(X); end

