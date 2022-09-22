function varargout = FFT_Filter_GUI(varargin)
% FFT_FILTER_GUI MATLAB code for FFT_Filter_GUI.fig
%      FFT_FILTER_GUI, by itself, creates a new FFT_FILTER_GUI or raises the existing
%      singleton*.
%
%      H = FFT_FILTER_GUI returns the handle to a new FFT_FILTER_GUI or the handle to
%      the existing singleton*.
%
%      FFT_FILTER_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FFT_FILTER_GUI.M with the given input arguments.
%
%      FFT_FILTER_GUI('Property','Value',...) creates a new FFT_FILTER_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before FFT_Filter_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to FFT_Filter_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help FFT_Filter_GUI

% Last Modified by GUIDE v2.5 12-Aug-2015 11:16:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @FFT_Filter_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @FFT_Filter_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before FFT_Filter_GUI is made visible.
function FFT_Filter_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to FFT_Filter_GUI (see VARARGIN)

% Choose default command line output for FFT_Filter_GUI
handles.output = hObject;
handles.data = handles.Working_Axes;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes FFT_Filter_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = FFT_Filter_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Load_Image.
function Load_Image_Callback(hObject, eventdata, handles)
% hObject    handle to Load_Image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

prompt = {'Enter the path name:'};
dlg_title = 'Select Path'; 
num_lines = 1;
def = {'C:\'};
paths = inputdlg(prompt,dlg_title,num_lines,def);

nIter = numel(paths);
for n = 1:nIter
    [filename,pathname] = uigetfile('..\*.*', '', paths{nIter});
end

ShowImageFile(filename,pathname,handles);

 function ShowImageFile(filename,pathname,handles)
    if ~isequal(filename, 0)
        fn=strcat(pathname,filename);
        I=imread(fn);
        axes(handles.Original_Image);
        imshow(I);
        Dimension = size(I,3); % Convert to grayscale if RGB, otherwise do nothing
         if Dimension >= 3
         I = rgb2gray(I);
         end
        axes(handles.Working_Axes);
        imshow(I),impixelinfo;
        
        ImageInfo = imfinfo(fn);
        set(handles.image, 'string', filename);
        set(handles.path, 'string', pathname);
        
    end


% --- Executes on button press in FFT.
function FFT_Callback(hObject, eventdata, handles)
% hObject    handle to FFT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Convert to frequency space
f = getimage(handles.Working_Axes);
F = (fftshift(fft2(double(f))));
S = log(abs(F));
Freq = S/max(S(:));
axes(handles.Working_Axes)
imshow(Freq), impixelinfo;




% --- Executes on button press in Mask_Bounds.
function Mask_Bounds_Callback(hObject, eventdata, handles)
% hObject    handle to Mask_Bounds (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

prompt = {'Enter Left Boundary of Central Low Frequency Peak:', 'Enter Desired Mask Width:'};
dlg_title = 'Frequency Mask Boundaries'; 
num_lines = 1;
def = {'965','40'};
Bounds = inputdlg(prompt,dlg_title,num_lines,def);
handles.LeftBound = Bounds(1);
handles.MaskWidth = Bounds(2);
guidata(hObject, handles)



% --- Executes on button press in Apply_Mask.
function Apply_Mask_Callback(hObject, eventdata, handles)
% hObject    handle to Apply_Mask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

f = getimage(handles.Original_Image);
F = (fftshift(fft2(double(f))));
S = log(abs(F));
Freq = S/max(S(:));
axes(handles.Working_Axes)
imshow(Freq), impixelinfo;

Pixels = size(Freq);
XPix = Pixels(1,2);
XCenter = round(size(Freq,2) / 2);
YCenter = round(size(Freq,1) / 2);

LeftBound = handles.LeftBound;
MaskWidth = handles.MaskWidth;
LeftBound = str2double(cell2mat(LeftBound));
MaskWidth = str2double(cell2mat(MaskWidth));

MaskTop = YCenter + round(MaskWidth/2);
MaskBottom = YCenter - round(MaskWidth/2);

DistCenter = XCenter - LeftBound;
RightBound = XCenter + DistCenter;


M = ones(Pixels);
for x = MaskBottom:MaskTop
    for y = 1:LeftBound
        M(x,y) = 0;
    end
end

for x = MaskBottom:MaskTop
    for y = RightBound:XPix
        M(x,y) = 0;
    end
end

Mask = M .* Freq;
axes(handles.Working_Axes)
imshow(Mask),impixelinfo




% --- Executes on button press in Reverse_FFT.
function Reverse_FFT_Callback(hObject, eventdata, handles)
% hObject    handle to Reverse_FFT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Apply mask to FFT image and convert back to spatial domain
Mask = getimage(handles.Working_Axes);
Mask = Mask(:,:,1);
Mask = double((Mask>0));
Original = getimage(handles.Original_Image);
F = (fftshift(fft2(double(Original))));
G = Mask .* F;
g = real( ifft2( ifftshift(G) ) );
axes(handles.Working_Axes)
imshow(g,[]), impixelinfo;

% --- Executes on button press in reset_original.
function reset_original_Callback(hObject, eventdata, handles)
% hObject    handle to reset_original (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Image = get(handles.image, 'String');
Path = get(handles.path, 'String');
ImageFileLocation = fullfile(Path, Image);
I = imread(ImageFileLocation);
Dimension = size(I,3); % Convert to grayscale if RGB, otherwise do nothing
if Dimension >= 3
    I = rgb2gray(I);
end
axes(handles.Working_Axes);
imshow(I),impixelinfo;
axes(handles.Original_Image);
imshow(I)



% --- Executes on button press in save_image.
function save_image_Callback(hObject, eventdata, handles)
% hObject    handle to save_image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Notched = getimage(handles.Working_Axes);
Notched = (Notched/max(Notched(:)));

ImageFolder = get(handles.path,'string');
Image = get(handles.image,'string');
fullFileName = fullfile(ImageFolder,Image);

[Pathstr,Name,Ext] = fileparts(fullFileName);
ReLabel = sprintf('%s_Notched%s', Name, Ext);
NewFileName = fullfile(Pathstr,ReLabel);
ImageNameShort = fullfile(Name,Ext);
imwrite(Notched, NewFileName);



function image_Callback(hObject, eventdata, handles)
% hObject    handle to image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of image as text
%        str2double(get(hObject,'String')) returns contents of image as a double


% --- Executes during object creation, after setting all properties.
function image_CreateFcn(hObject, eventdata, handles)
% hObject    handle to image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function path_Callback(hObject, eventdata, handles)
% hObject    handle to path (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of path as text
%        str2double(get(hObject,'String')) returns contents of path as a double


% --- Executes during object creation, after setting all properties.
function path_CreateFcn(hObject, eventdata, handles)
% hObject    handle to path (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in batch_process.
function batch_process_Callback(hObject, eventdata, handles)
% hObject    handle to batch_process (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ImageFolder = uigetdir;
ImageFiles = [dir(fullfile(ImageFolder,'*tif')); dir(fullfile(ImageFolder,'*bmp'))];

for k = 1:length(ImageFiles) % Open sequence loop
    baseFileName = ImageFiles(k).name;
    fullFileName = fullfile(ImageFolder, baseFileName);
    fprintf(1, 'Now reading %s\n', baseFileName);
    
    I = imread(fullFileName); % Read in image.
  
    
    ImageInfo = imfinfo(fullFileName);
    set(handles.image, 'string', baseFileName);
    set(handles.path, 'string', ImageFolder);
    
    Dimension = size(I,3); % Convert to grayscale if RGB, otherwise do nothing
    if Dimension >= 3
        I = rgb2gray(I);
    end
    
    Rectangle = handles.data;
    if Rectangle > 0
        I = imcrop(I, Rectangle);
    else
    end
    
    axes(handles.Original_Image)
    imshow(I)
    axes(handles.Working_Axes)
    imshow(I)
    
    f = getimage(handles.Working_Axes);
    F = (fftshift(fft2(double(f))));
    S = log(abs(F));
    Freq = S/max(S(:));
    axes(handles.Working_Axes)
    imshow(Freq), impixelinfo;
    
    Current_Image = getimage(handles.Working_Axes);
    Pixels = size(Current_Image);
    XPix = Pixels(1,2);
    XCenter = round(size(Current_Image,2) / 2);
    YCenter = round(size(Current_Image,1) / 2);
    
    LeftBound = handles.LeftBound;
    MaskWidth = handles.MaskWidth;
    LeftBound = str2double(cell2mat(LeftBound));
    MaskWidth = str2double(cell2mat(MaskWidth));
    
    MaskTop = YCenter + round(MaskWidth/2);
    MaskBottom = YCenter - round(MaskWidth/2);
    
    DistCenter = XCenter - LeftBound;
    RightBound = XCenter + DistCenter;
    
    
    M = ones(Pixels);
    for x = MaskBottom:MaskTop
        for y = 1:LeftBound
            M(x,y) = 0;
        end
    end
    
    for x = MaskBottom:MaskTop
        for y = RightBound:XPix
            M(x,y) = 0;
        end
    end
    
    Mask = M .* Current_Image;
    axes(handles.Working_Axes)
    imshow(Mask)
    
    Mask = getimage(handles.Working_Axes);
    Mask = Mask(:,:,1);
    Mask = double((Mask>0));
    Original = getimage(handles.Original_Image);
    F = (fftshift(fft2(double(Original))));
    G = Mask .* F;
    g = real( ifft2( ifftshift(G) ) );
    axes(handles.Working_Axes)
    imshow(g,[]);
    
    Notched = getimage(handles.Working_Axes);
    Notched = (Notched/max(Notched(:)));
   
    [Pathstr,Name,Ext] = fileparts(fullFileName);
    ReLabel = sprintf('%s_Notched%s', Name, Ext);
    NewFileName = fullfile(Pathstr,ReLabel);
    ImageNameShort = fullfile(Name,Ext);
    imwrite(Notched, NewFileName);
    
end

  


% --- Executes on button press in crop.
function crop_Callback(hObject, eventdata, handles)
% hObject    handle to crop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[Cropped rect] = imcrop(handles.Working_Axes);

axes(handles.Working_Axes);
imshow(Cropped), impixelinfo;
axes(handles.Original_Image);
imshow(Cropped)

handles.data = rect;
guidata(hObject, handles)
