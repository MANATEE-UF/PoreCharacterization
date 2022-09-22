%fft filtering

%% First Section
im = input('Select image stack: ');
im2(:,:,1) = fftshift(fft2(im(:,:,1)));
imshow(log(1+abs(im2(:,:,1))),[])

remove1 = createMask(drawrectangle);
remove2 = createMask(drawrectangle);

%% Second Section
remove_tot = remove1+remove2;

masked_im(:,:,1) = im2(:,:,1) - (im2(:,:,1).*remove_tot);

filtered(:,:,1) = ifft2(ifftshift(masked_im(:,:,1)));

subplot(2,1,1);
imshow(im(:,:,1));

subplot(2,1,2);
imshow(filtered(:,:,1),[]);

%% Third Section
for i = 2:size(im,[3])
    im2(:,:,i) = fftshift(fft2(im(:,:,i)));
end

for i = 2:size(im,[3])
    masked_im(:,:,i) = im2(:,:,i) - (im2(:,:,i).*remove_tot);
    filtered(:,:,i) = ifft2(ifftshift(masked_im(:,:,i)));
end
filtered = real(uint8(filtered));
%%
%im_in = input('Select image stack: ');
%im = im_in(:,:,1);
%im2=fft2(im);
%im3=fftshift(log(1+abs(im2)));
%imshow(im3,[])

%remove1 = createMask(drawrectangle);
%remove2 = createMask(drawrectangle);

%%
%remove_tot = remove1+remove2;
%mask = im3.*remove_tot;

%masked_im = im3 - mask;

