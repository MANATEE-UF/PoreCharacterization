iptsetpref('ImshowBorder','tight');
clear all
close all

% Call image
f = imread('image200.tif');

% Convert to grayscale if RGB, otherwise do nothing
Dimension = size(f,3); 
if Dimension >= 3
    f = rgb2gray(f);
end

% Crop to region of interest
% fcrop = imcrop(f,[]);
figure, imshow(f, [])

% Convert to frequency space
F = (fftshift(fft2(double(f))));
S = log(abs(F));
Spectrum = S/max(S(:));
figure, imshow(Spectrum), impixelinfo;

Pixels = size(Spectrum);
XPix = Pixels(1,2);

M = ones(Pixels);
for x = 875:895
    for y = 1:1000
        M(x,y) = 0;
    end
end

for x = 875:895
    for y = 1050:XPix
        M(x,y) = 0;
    end
end

% figure, imshow(M);

Mask = M .* Spectrum
figure, imshow(Mask)

% Apply mask to FFT image and convert back to spatial domain
G = M .* F;
g = real( ifft2( ifftshift(G) ) );
figure, imshow(g,[]);
imwrite( g/max(g(:)) , 'test.tif');
