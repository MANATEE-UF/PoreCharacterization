clear all
close all

% prompt = {'Enter the path name:'};
% dlg_title = 'Select Path'; 
% num_lines = 1;
% def = {'C:\'};
% imagepath = inputdlg(prompt,dlg_title,num_lines,def);
% nIter = numel(imagepath);
% for n = 1:nIter
%        img = imread(imagepath{nIter});
% end

[filename,pathname] = uigetfile('..\*.*');
image = strcat(pathname, filename);
img = imread(image);
imshow(img)

hold on

M = size(img,1)
N = size(img,2)

% O = sqrt((M * N)/121);

for k = 0:190:M
    x = [1 N];
    y = [k k];
    plot(x,y,'Color','r','LineStyle','-');
    plot(x,y,'Color','b','LineStyle',':');
end

for k = 0:190:N
    x = [k k];
    y = [1 M];
    plot(x,y,'Color','r','LineStyle','-');
    plot(x,y,'Color','b','LineStyle',':');
end

sprintf(filename)

hold off