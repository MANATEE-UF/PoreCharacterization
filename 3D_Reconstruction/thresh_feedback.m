function [high_thresh] = thresh_feedback(img,low_thresh,high_thresh,name)

dims = size(img);
slice = img(:,:,round(dims(3)/5));
slice(1:dims(1),dims(2)+1:dims(2)*2) = img(:,:,round(dims(3)*2/5));
slice(dims(1)+1:dims(1)*2,1:dims(2)) = img(:,:,round(dims(3)*3/5));
slice(dims(1)+1:dims(1)*2,dims(2)+1:dims(2)*2) = img(:,:,round(dims(3)*4/5));
slice_BW = and(slice > low_thresh, slice < high_thresh);
slice_BW = imopen(slice_BW, strel('disk',2));
high_thresh = mat2cell(num2str(high_thresh),1);
kg = true;
while kg
    temp = imover(slice,slice_BW,[.1 -.2 -.2]);
    figure, imshow(temp);
    set(gcf, 'Position', get(0, 'Screensize'));
    button = questdlg(['Accept ' name ' threshold?'], 'Threshold', 'Yes', 'No', 'Yes');
    close(gcf);
    if strcmp(button, 'Yes')
        kg = false;
    else
        high_thresh = inputdlg(['Enter new ' name ' threshold'], 'Threshold', 1, high_thresh);
        slice_BW = and(slice>low_thresh,slice<str2num(high_thresh{1}));
        slice_BW = imopen(slice_BW, strel('disk',2));
    end
end
high_thresh = cell2mat(high_thresh);
high_thresh = str2num(high_thresh);