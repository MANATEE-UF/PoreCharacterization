%SAUVOLA local thresholding.

function output=sauvola(image, varargin)
% Initialization
numvarargs = length(varargin);      % only want 3 optional inputs at most
if numvarargs > 3
    error('myfuns:somefun2Alt:TooManyInputs', ...
     'Possible parameters are: (image, [m n], threshold, padding)');
end
 
optargs = {[3 3] 0.34 'replicate'}; % set defaults
 
optargs(1:numvarargs) = varargin;   % use memorable variable names
[window, k, padding] = optargs{:};

if ndims(image) ~= 2
    error('The input image must be a two-dimensional array.');
end

% Convert to double
image = double(image);

% Mean value
mean = averagefilter(image, window, padding);

% Standard deviation
meanSquare = averagefilter(image.^2, window, padding);
deviation = (meanSquare - mean.^2).^0.5;

% Sauvola
R = max(deviation(:));
threshold = mean.*(1 + k * (deviation / R-1));
output = (image > threshold);
