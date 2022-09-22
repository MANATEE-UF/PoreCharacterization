%% Read in and sort images (gray scale)
path = uigetdir(pwd, 'Select Image Folder'); %Initial read in
cd(path);
img = dir('*.tif');

for i = 1:length(img) %Sorting
    image(:,:,i) = imread(img(i).name);
end

%% Read in and sort images (RGB)
path = uigetdir(pwd, 'Select Image Folder'); %Initial read in
cd(path);
img = dir('*.tif');

for i = 1:length(img) %Sorting
    image(:,:,:,i) = imread(img(i).name);
end

%% Convert color to grayscale
for i = 1:length(img)
    gray_im(:,:,i) = rgb2gray(image(:,:,:,i));
end
image = gray_im;

%% Rotation(calc angle in ImageJ)
%Need to change angle value here
rotated = imrotate(image,-7.7,'bicubic','loose'); %imrotate(image, angle, method, bbox)
                                                  %angle is in degrees in a
                                                  %counterclockwise
                                                  %direction around center
                                                  %point

%Run imshow(rotated(:,:,1)) in command window to check rotation, may need
%to flip +/- value of the angle
                                                  
%% Translate images down (margin calculated in command window)
% run imshow(firstimage), b1=createMask(drawline), regionprops(b1)
%     imshow(lastimage), b2=createMask(drawline), regoinprops(b2)
% calculate shift margin
% multiply shift margin (SM) by i in code (current margin is .6705)

for i = 1:size(rotated,[3])                       %SM%
    trans(:,:,i) = imtranslate(rotated(:,:,i),[0,(.6705*i)]);
end

%Run imshow(trans(:,:,lastimage#)) b3=createMask(drawline), regionprops(b3)
%in command window & compare to regionprops(b1) to check that translation 
%was performed correctly

%% Single Image Stack Cropping
figure('Name', 'Select Region of Interest'),
imshow(trans(:,:,1));

ROI = createMask(drawrectangle);
%% Single Crop pt 2
ROI_crop = regionprops(ROI, 'BoundingBox');
ROI_crop = ROI_crop.BoundingBox;
close gcf,

for i = 1:size(rotated,[3])
    crop(:,:,i) = imcrop(trans(:,:,i),ROI_crop);
end

%% Cropping multiple stacks
figure('Name', 'Select Region of Interest'),
imshow(trans_1(:,:,1)); %check images for one with tightest borders

ROI = createMask(drawrectangle);
%% Multi Crop pt 2
ROI_crop = regionprops(ROI, 'BoundingBox');
ROI_crop = ROI_crop.BoundingBox;
close gcf,

for i = 1:size(trans_1,[3])
    crop_1(:,:,i) = imcrop(trans_1(:,:,i),ROI_crop);
    crop_2(:,:,i) = imcrop(trans_2(:,:,i),ROI_crop);
    crop_3(:,:,i) = imcrop(trans_3(:,:,i),ROI_crop);
end

%% Removes noise (EDS)
for i = 1:size(crop_1,[3])
    filtered_1(:,:,i) = wiener2(crop_1(:,:,i), [5 5]); %Removes noise with adaptive filter
    filtered_2(:,:,i) = wiener2(crop_2(:,:,i), [5 5]);
end

% To remove curtaining artifacts on SE micrographs, use fft_filter.m script

%% Determine threshold
% run resized = filtered_#; in command window
delta_z = input('Delta z: '); %delta_z = slice depth (nm) / pixel size (nm)
nm_per_pix = input('Pixel size (nm): ');
filename = input('File name: '); %What name you wish to save file with
out = uigetdir(pwd, 'Select Output Folder Destination');
cd(out);
new_folder = input('Name of new folder: ');
mkdir(out,new_folder)
cd(new_folder)

smooth = imgaussfilt3(resized);
dims = size(smooth);
smooth = imresize3(smooth, [dims(1), dims(2),round(dims(3)*delta_z,0)]);

thresh = multithresh(smooth,4);

thresh = thresh_feedback(smooth, 0,thresh(1),'pore');

voids = smooth<thresh; %Use > for light selection (EDS). Use < for dark selection (bubbles).
AoI = imcrop(ROI,ROI_crop);
dims = size(smooth);
%% Cleaning mask (EDS threshold cleanup)
h = waitbar(0,'Removing small pores and filling gaps');
for i = 1:dims(3)
    waitbar(i/dims(3));
    temp = voids(:,:,i);
    temp(~AoI) = 0;
    voids(:,:,i) = temp;
    voids(:,:,i) = imopen(voids(:,:,i),strel('disk',2));
    voids(:,:,i) = imclose(voids(:,:,i),strel('disk',5));
    voids(:,:,i) = imfill(voids(:,:,i), 'holes');
end
close(h);

h = waitbar(0, 'Optimizing pore contours for best fit');
for i = 1:dims(3)  
    waitbar(i/dims(3));
    voids(:,:,i) = activecontour(resized(:,:,i),voids(:,:,i),20, 'edge');
end
close(h);

%% Cleaning mask (Bubble threshold cleanup)
for i = 1:size(dims(3))
    temp = imcomplement(imbinarize(smooth(:,:,i),adaptthresh(smooth(:,:,i))));
    temp(~AoI) = 0;
    voids(:,:,i) = temp;
end
voids = imopen(voids, strel('sphere',1));
voids = imfill(voids, 'holes');
voids = imclose(voids, strel('sphere',3));
for i = 1:dims(3)
    voids(:,:,i) = imfill(voids(:,:,i), 'holes');
end
voids = activecontour(smooth,voids,100,'edge');
for i = 1:dims(3)
    voids(:,:,i) = imfill(voids(:,:,i), 'holes');
end
voids = imopen(voids, strel('sphere',1));
voids = logical(voids);
 
parfor i = 1:dims(3)
    thresh = multithresh(smooth(:,:,i),1);
    temp_v = voids(:,:,i);
    temp_c = smooth(:,:,i);
    temp_v(temp_c>thresh) = 0;             %Flip </> for voids vs precips(left on pores)
    voids(:,:,i) = temp_v;
    voids(:,:,i) = imclose(voids(:,:,i),strel('disk', 1));
    voids(:,:,i) = imfill(voids(:,:,i),'holes');
end

%% Volume Rendering, Overlay video, Properties Calculation, and Save
L = bwlabeln(voids);
props = regionprops3(L,'Volume', 'VoxelList','Centroid', 'Orientation','PrincipalAxisLength');
props = table2struct(props);
%Here is where I need to check and remove bad ones
h = waitbar(0, 'Creating pore overlay images');
for i = 1:dims(3)
    waitbar(i/dims(3));
    temp = smooth(:,:,i); 
    temp = imover(temp,voids(:,:,i),[0.4 0 0]);
    overlay(:,:,1,i) = temp(:,:,1);
    overlay(:,:,2,i) = temp(:,:,2);
    overlay(:,:,3,i) = temp(:,:,3);
end
close(h);
backup = voids;
 
% %ONLY RUN THIS TO TAKE OUT BIG BAD ONES
%  CC = bwconncomp(voids);
%  [~, idx] = max(cellfun(@numel, CC.PixelIdxList));
%  bad = voids;
%  bad(voids==1) = 0;
%   bad(CC.PixelIdxList{idx}) = 1;
%  voids(bad)=0;
%  %END PRIOR WARNING
 L = bwlabeln(voids);
props = regionprops3(L,'Volume', 'VoxelList','Centroid', 'Orientation','PrincipalAxisLength');
props = table2struct(props);
%end section
 
 h = waitbar(0, 'Calculating centroid nearest neighbor distances.');
clearvars dist
for i = 1:numel(props)
     waitbar(i/numel(props));
     center1 = (props(i).Centroid);
     for j = 1:numel(props)
         center2 = (props(j).Centroid);
         dist(j) = sum((center1-center2).^2)^.5;
     end
     [dist, idx] = sort(dist);
     props(i).NN1 =dist(2);
     props(i).NN1_obj = idx(2);
     props(i).NN2 =dist(3);
     props(i).NN2_obj = idx(3);
     props(i).NN3 =dist(4);
     props(i).NN3_obj = idx(4);
end
close(h);
h = waitbar(0, 'Calculating surface nearest neighbor distances.');
parfor i = 1:numel(props)
    %waitbar(i/numel(props));
    BW1 = (L==i);
    BW2 = voids;
    BW2(BW1) = 0;
    dist_array = bwdist(BW2);
    dist_list = nonzeros(dist_array(BW1));
    dist(i) = min(dist_list);
    %props(i).SurfaceDistance = dist;
end
for i = 1:numel(props)
    props(i).SurfaceDistance = dist(i);
end
close(h);
props = struct2table(props);
h = waitbar(0, 'Creating 3D pore rendering');
max_vol  = log(max(props.Volume));
min_vol = log(min(props.Volume));
cmap = jet(256);                     %Can change colormap here if desired. 
for i = 1:height(props)
    waitbar(i/height(props));
    list = props.VoxelList{i,1};
    shp = alphaShape(list(:,2),list(:,1),list(:,3));
    shp_list{i,1} = shp;
    cind = floor((log(props.Volume(i))-min_vol)/max_vol*255)+1;                 %NEED TO CHANGE THIS FOR UNIFORMITY BETWEEN IMAGES
    shp_list{i,2} = cmap(cind,:);
end
close(h);
h = waitbar(0, 'Creating pore overlay images');
for i = 1:dims(3)
    waitbar(i/dims(3));
    temp = smooth(:,:,i); 
    temp = imover(temp,voids(:,:,i),[0.4 0 0]);
    overlay(:,:,1,i) = temp(:,:,1);
    overlay(:,:,2,i) = temp(:,:,2);
    overlay(:,:,3,i) = temp(:,:,3);
end
close(h);
v = VideoWriter([filename '_overlay.avi']);
open(v);
writeVideo(v,overlay);
close(v);
overlay(:,:,1,:) = overlay(:,:,2,:);
v = VideoWriter([filename '_processed.avi']);
open(v);
writeVideo(v,overlay);
close(v);
% props = struct2cell(props);
% props = props';
props(:,3) = [];
% props = cell2mat(props);
% props(:,1) = props(:,1) * nm_per_pix^3 / 1000^3;
% props(:,2:7) = props(:,2:7) * nm_per_pix / 1000;
% props(:,3) = props(:,3) * nm_per_pix / 1000;
% xlswrite([filename '_data.xlsx'],props);
writetable(props,[filename '_data.xlsx']);
save([filename '_3D'],'shp_list', 'props');
