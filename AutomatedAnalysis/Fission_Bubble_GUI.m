function varargout = Fission_Bubble_GUI(varargin)
% FISSION_BUBBLE_GUI MATLAB code for Fission_Bubble_GUI.fig
%      FISSION_BUBBLE_GUI, by itself, creates a new FISSION_BUBBLE_GUI or raises the existing
%      singleton*.
%
%      H = FISSION_BUBBLE_GUI returns the handle to a new FISSION_BUBBLE_GUI or the handle to
%      the existing singleton*.
%
%      FISSION_BUBBLE_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FISSION_BUBBLE_GUI.M with the given input arguments.
%
%      FISSION_BUBBLE_GUI('Property','Value',...) creates a new FISSION_BUBBLE_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Fission_Bubble_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Fission_Bubble_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Fission_Bubble_GUI

% Last Modified by GUIDE v2.5 19-Jun-2017 12:07:14

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Fission_Bubble_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @Fission_Bubble_GUI_OutputFcn, ...
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

% --- Executes just before Fission_Bubble_GUI is made visible.
function Fission_Bubble_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Fission_Bubble_GUI (see VARARGIN)
set(handles.text_gauss_window, 'String', round(get(handles.gaussian_window_size, 'Value')));
set(handles.text_gauss_sigma, 'String', round(get(handles.gauss_sigma, 'Value')));
set(handles.text_bilateral_size, 'String', round(get(handles.bilateral_size, 'Value')));
set(handles.text_sigma_spatial, 'String', round(get(handles.sigma_spatial, 'Value')));
set(handles.text_sigma_intensity, 'String', get(handles.sigma_intensity, 'Value'));
set(handles.text_thresh_size, 'String', round(get(handles.thresh_size, 'Value')));
set(handles.text_thresh_val, 'String', get(handles.thresh_val, 'Value'));
set(handles.text_median_size, 'String', round(get(handles.median_size, 'Value')));
set(handles.border_conn_comp, 'Value', 1);

handles.data = handles.Working_Axes;
handles.Coordinates = [0 0 0 0];
handles.ScaleBarMicrometers = {'1'};
handles.ScaleBarPixels = {'1'};

% Choose default command line output for Fission_Bubble_GUI
handles.output = hObject;


% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Fission_Bubble_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = Fission_Bubble_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in load_single_image.
function load_single_image_Callback(hObject, eventdata, handles)
% hObject    handle to load_single_image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

prompt = {'Enter the path name:'};
dlg_title = 'Select Path'; 
num_lines = 1;
def = {'C:\'};
paths = inputdlg(prompt,dlg_title,num_lines,def);

nIter = numel(paths);
    for n = 1:nIter
        [filename,pathname] = uigetfile('..\*.*', 'Select', paths{nIter});
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
        imshow(I);
        axes(handles.Histogram);
        imhist(I)  
      
        ImageInfo = imfinfo(fn);
        set(handles.image, 'string', filename);
        set(handles.path, 'string', pathname);
        FileModDate = ImageInfo.FileModDate;
        set(handles.mod_date, 'string', FileModDate);
        FileSize = ImageInfo.FileSize/(10^6);
        set(handles.size_MB, 'string', FileSize);
        Format = ImageInfo.Format;
        set(handles.format, 'string', Format);
        Width = ImageInfo.Width;
        Height = ImageInfo.Height;
        set(handles.dimension, 'string', sprintf('%d x %d', Height, Width));
        BitDepth = ImageInfo.BitDepth;
        set(handles.bit_depth, 'string', BitDepth);
      
        TIF = strfind(Format, 'tif');
        BMP = strfind(Format, 'bmp');
        if TIF >= 1;
            CompressionTIF = ImageInfo.Compression;
            set(handles.compression, 'string', CompressionTIF);
        else
            CompressionBMP = ImageInfo.CompressionType;
            set(handles.compression, 'string', CompressionBMP);
        end
        
    end
    
   
% --- Executes on button press in measure_scale.
function measure_scale_Callback(hObject, eventdata, handles)
% hObject    handle to measure_scale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
imtool(getimage(handles.Working_Axes))

% --- Executes on button press in set_scale.
function set_scale_Callback(hObject, eventdata, handles)
% hObject    handle to set_scale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
prompt = {'Enter length of scale bar in um:','Enter length of scale bar in pixels:'};
dlg_title = 'Scale Bar Input'; 
num_lines = 1;
def = {'5','690'};
ScaleBar = inputdlg(prompt,dlg_title,num_lines,def);
handles.ScaleBarMicrometers = ScaleBar(1);
handles.ScaleBarPixels = ScaleBar(2);
guidata(hObject, handles)

% --- Executes on button press in bilateral_filter.
function bilateral_filter_Callback(hObject, eventdata, handles)
% hObject    handle to bilateral_filter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Current_Image = getimage(handles.Working_Axes);
DP = double(Current_Image)/255;
BilateralSize = round(get(handles.bilateral_size, 'Value'));
SigmaSpatial = round(get(handles.sigma_spatial, 'Value'));
SigmaIntensity = get(handles.sigma_intensity, 'Value');
B = bfilter2((DP), BilateralSize, [SigmaSpatial SigmaIntensity]);
axes(handles.Working_Axes);
imshow(B);
axes(handles.Histogram);
imhist(B);
set(handles.distribution_plot_text,'String', 'Histogram');

handles.data = B;
guidata(hObject, handles)

% --- Executes on button press in Reset_original.
function Reset_original_Callback(hObject, eventdata, handles)
% hObject    handle to Reset_original (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cla(handles.Working_Axes);
handles.Coordinates = [0 0 0 0];

Image = get(handles.image, 'String');
Path = get(handles.path, 'String');
ImageFileLocation = fullfile(Path, Image);
I = imread(ImageFileLocation);
Dimension = size(I,3); % Convert to grayscale if RGB, otherwise do nothing
if Dimension >= 3
    I = rgb2gray(I);
end
axes(handles.Working_Axes);
imshow(I);
axes(handles.Histogram);
imhist(I);
axes(handles.Original_Image);
imshow(I)
set(handles.distribution_plot_text,'String', 'Histogram');

handles.data = I;
guidata(hObject, handles)

% --- Executes on button press in contrast_adjustment.
function contrast_adjustment_Callback(hObject, eventdata, handles)
% hObject    handle to contrast_adjustment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
ContAdj = imadjust(Current_Image, stretchlim(Current_Image, [0.01 0.99]));
imshow(ContAdj);
axes(handles.Histogram);
imhist(ContAdj);
set(handles.distribution_plot_text,'String', 'Histogram');

handles.data = ContAdj;
guidata(hObject, handles)

% --- Executes on button press in gaussian_blur.
function gaussian_blur_Callback(hObject, eventdata, handles)
% hObject    handle to gaussian_blur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
        
Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
BlurSize = round(get(handles.gaussian_window_size, 'Value'));
BlurSigma = round(get(handles.gauss_sigma, 'Value'));
h = fspecial('gaussian', [BlurSize BlurSize], BlurSigma);
Blur = imfilter(Current_Image, h);
imshow(Blur);
axes(handles.Histogram);
imhist(Blur);
set(handles.distribution_plot_text,'String', 'Histogram');

handles.data = Blur;
guidata(hObject, handles)

% --- Executes on button press in sauvola.
function sauvola_Callback(hObject, eventdata, handles)
% hObject    handle to sauvola (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Current_Image = getimage(handles.Working_Axes);
Dimension = size(handles.data,3); % Convert to grayscale if RGB, otherwise do nothing
  if Dimension >= 3
      PP = rgb2gray(handles.data);
  end
axes(handles.Working_Axes);
ThreshSize = round(get(handles.thresh_size, 'Value'));
Val = get(handles.thresh_val, 'Value');
ThreshVal = round(Val*100)/100;
BW = sauvola(handles.data, [ThreshSize ThreshSize], ThreshVal);
BW = imcomplement(BW);
imshow(BW);

Pixels = handles.ScaleBarPixels;
um = handles.ScaleBarMicrometers;
Pixels = str2double(cell2mat(Pixels));
um = str2double(cell2mat(um));
PixeluMRatio = Pixels / um;

Voids = regionprops(BW, 'EquivDiameter'); % Extract area and centroid data for binary blobs
EquivDiameter = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(EquivDiameter, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Count');

% --- Executes on button press in despeckle.
function despeckle_Callback(hObject, eventdata, handles)
% hObject    handle to despeckle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
MedianSize = round(get(handles.median_size, 'Value'));
Despeckle = medfilt2(Current_Image, [MedianSize MedianSize]);
imshow(Despeckle);

Pixels = handles.ScaleBarPixels;
um = handles.ScaleBarMicrometers;
Pixels = str2double(cell2mat(Pixels));
um = str2double(cell2mat(um));
PixeluMRatio = Pixels / um;

Voids = regionprops(Despeckle, 'EquivDiameter'); % Extract area and centroid data for binary blobs
EquivDiameter = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(EquivDiameter, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Count');

% --- Executes on button press in fill_holes.
function fill_holes_Callback(hObject, eventdata, handles)
% hObject    handle to fill_holes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
BWfill = imfill(Current_Image, 'holes');
imshow(BWfill);

Pixels = handles.ScaleBarPixels;
um = handles.ScaleBarMicrometers;
Pixels = str2double(cell2mat(Pixels));
um = str2double(cell2mat(um));
PixeluMRatio = Pixels / um;

Voids = regionprops(BWfill, 'EquivDiameter'); % Extract area and centroid data for binary blobs
EquivDiameter = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(EquivDiameter, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Count');

% --- Executes on button press in clear_border.
function clear_border_Callback(hObject, eventdata, handles)
% hObject    handle to clear_border (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
ConnCompContents = get(handles.border_conn_comp, 'String');
ConnectedComponentChar = ConnCompContents{get(handles.border_conn_comp, 'Value')};
ConnectedComponentNum = str2double(ConnectedComponentChar);
ClearBorder = imclearborder(Current_Image, ConnectedComponentNum);
imshow(ClearBorder);

Pixels = handles.ScaleBarPixels;
um = handles.ScaleBarMicrometers;
Pixels = str2double(cell2mat(Pixels));
um = str2double(cell2mat(um));
PixeluMRatio = Pixels / um;

Voids = regionprops(ClearBorder, 'EquivDiameter'); % Extract area and centroid data for binary blobs
EquivDiameter = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(EquivDiameter, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Count');

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

% --- Executes on button press in all_postprocess.
function all_postprocess_Callback(hObject, eventdata, handles)
% hObject    handle to all_postprocess (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
MedianSize = round(get(handles.median_size, 'Value'));
Despeckle = medfilt2(Current_Image, [MedianSize MedianSize]);
imshow(Despeckle);

axes(handles.Working_Axes);
BWfill = imfill(Current_Image, 'holes');
imshow(BWfill);

axes(handles.Working_Axes);
ConnCompContents = get(handles.border_conn_comp, 'String');
ConnectedComponentChar = ConnCompContents{get(handles.border_conn_comp, 'Value')};
ConnectedComponentNum = str2double(ConnectedComponentChar);
ClearBorder = imclearborder(Current_Image, ConnectedComponentNum);
imshow(ClearBorder);

Pixels = handles.ScaleBarPixels;
um = handles.ScaleBarMicrometers;
Pixels = str2double(cell2mat(Pixels));
um = str2double(cell2mat(um));
PixeluMRatio = Pixels / um;

ClearBorder = regionprops(ClearBorder, 'EquivDiameter'); % Extract area and centroid data for binary blobs
EquivDiameter = (cat(1, ClearBorder.EquivDiameter)) / (PixeluMRatio);

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(EquivDiameter, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Count');

% --- Executes on button press in all_preprocess.
function all_preprocess_Callback(hObject, eventdata, handles)
% hObject    handle to all_preprocess (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
ContAdj = imadjust(Current_Image, stretchlim(Current_Image, [0.01 0.99]));
imshow(ContAdj);
axes(handles.Histogram);
imhist(ContAdj);

Current_Image = getimage(handles.Working_Axes);
DP = double(Current_Image)/255;
BilateralSize = round(get(handles.bilateral_size, 'Value'));
SigmaSpatial = round(get(handles.sigma_spatial, 'Value'));
SigmaIntensity = get(handles.sigma_intensity, 'Value');
B = bfilter2((DP), BilateralSize, [SigmaSpatial SigmaIntensity]);
axes(handles.Working_Axes);
imshow(B);
axes(handles.Histogram);
imhist(B);

Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
BlurSize = round(get(handles.gaussian_window_size, 'Value'));
BlurSigma = round(get(handles.gauss_sigma, 'Value'));
h = fspecial('gaussian', [BlurSize BlurSize], BlurSigma);
Blur = imfilter(Current_Image, h);
imshow(Blur);
axes(handles.Histogram);
imhist(Blur);

handles.data = Blur;
guidata(hObject, handles)


% --- Executes on slider movement.
function gaussian_window_size_Callback(hObject, eventdata, handles)
% hObject    handle to gaussian_window_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
set(handles.text_gauss_window,'String', round(get(handles.gaussian_window_size,'Value')));


% --- Executes during object creation, after setting all properties.
function gaussian_window_size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gaussian_window_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function text_gauss_window_Callback(hObject, eventdata, handles)
% hObject    handle to text_gauss_window (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_gauss_window as text
%        str2double(get(hObject,'String')) returns contents of text_gauss_window as a double
val = str2double(get(hObject, 'String'));
if val > 20
    val = 20;
end
if val < 0
    val = 0;
end
set(handles.gaussian_window_size, 'Value', val);
set(handles.text_gauss_window, 'String', round(get(handles.gaussian_window_size, 'Value')));
guidata(hObject, handles)



% --- Executes during object creation, after setting all properties.
function text_gauss_window_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_gauss_window (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function text_gauss_sigma_Callback(hObject, eventdata, handles)
% hObject    handle to text_gauss_sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_gauss_sigma as text
%        str2double(get(hObject,'String')) returns contents of text_gauss_sigma as a double
val = str2double(get(hObject, 'String'));
if val > 8
   val = 8
end
if val < 0
   val = 0
end
set(handles.gauss_sigma, 'Value', val);
set(handles.text_gauss_sigma,'String', round(get(handles.gauss_sigma,'Value')));
guidata(hObject, handles)



% --- Executes during object creation, after setting all properties.
function text_gauss_sigma_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_gauss_sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function gauss_sigma_Callback(hObject, eventdata, handles)
% hObject    handle to gauss_sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
set(handles.text_gauss_sigma,'String', round(get(handles.gauss_sigma,'Value')));


% --- Executes during object creation, after setting all properties.
function gauss_sigma_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gauss_sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function text_bilateral_size_Callback(hObject, eventdata, handles)
% hObject    handle to text_bilateral_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_bilateral_size as text
%        str2double(get(hObject,'String')) returns contents of text_bilateral_size as a double
val = str2double(get(hObject, 'String'));
if val > 30
    val = 30
end
if val < 0
    val = 0
end
set(handles.bilateral_size, 'Value', val);
set(handles.text_bilateral_size, 'String', round(get(handles.bilateral_size, 'Value')));
guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function text_bilateral_size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_bilateral_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function text_sigma_spatial_Callback(hObject, eventdata, handles)
% hObject    handle to text_sigma_spatial (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_sigma_spatial as text
%        str2double(get(hObject,'String')) returns contents of text_sigma_spatial as a double
val = str2double(get(hObject, 'String'));
if val > 10
    val = 10;
end
if val < 0
    val = 0;
end
set(handles.sigma_spatial, 'Value', val);
set(handles.text_sigma_spatial, 'String', round(get(handles.sigma_spatial, 'Value')));
guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function text_sigma_spatial_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_sigma_spatial (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function text_sigma_intensity_Callback(hObject, eventdata, handles)
% hObject    handle to text_sigma_intensity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_sigma_intensity as text
%        str2double(get(hObject,'String')) returns contents of text_sigma_intensity as a double
val = str2double(get(hObject, 'String'));
if val > 2
    val = 2
end
if val < 0
    val = 0
end
set(handles.sigma_intensity, 'Value', val);
set(handles.text_sigma_intensity, 'String', get(handles.sigma_intensity, 'Value'));
guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function text_sigma_intensity_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_sigma_intensity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function bilateral_size_Callback(hObject, eventdata, handles)
% hObject    handle to bilateral_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
set(handles.text_bilateral_size,'String', round(get(handles.bilateral_size,'Value')));


% --- Executes during object creation, after setting all properties.
function bilateral_size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bilateral_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function sigma_spatial_Callback(hObject, eventdata, handles)
% hObject    handle to sigma_spatial (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
set(handles.text_sigma_spatial,'String', round(get(handles.sigma_spatial,'Value')));


% --- Executes during object creation, after setting all properties.
function sigma_spatial_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sigma_spatial (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function sigma_intensity_Callback(hObject, eventdata, handles)
% hObject    handle to sigma_intensity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
SigmaI = get(handles.sigma_intensity, 'Value');
SigmaIRound = (round(SigmaI*100))/100;
set(handles.text_sigma_intensity,'String', SigmaIRound);

% --- Executes during object creation, after setting all properties.
function sigma_intensity_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sigma_intensity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function text_thresh_size_Callback(hObject, eventdata, handles)
% hObject    handle to text_thresh_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_thresh_size as text
%        str2double(get(hObject,'String')) returns contents of text_thresh_size as a doubleval = str2double(get(hObject, 'String'));
val = str2double(get(hObject, 'String'));
if val > 100
    val = 100
end
if val < 0
    val = 0
end
set(handles.thresh_size, 'Value', val);
set(handles.text_thresh_size, 'String', round(get(handles.thresh_size, 'Value')));
guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function text_thresh_size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_thresh_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function text_thresh_val_Callback(hObject, eventdata, handles)
% hObject    handle to text_thresh_val (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_thresh_val as text
%        str2double(get(hObject,'String')) returns contents of text_thresh_val as a double
val = str2double(get(hObject, 'String'));
if val > 1
    val = 1
end
if val < 0
    val = 0
end
set(handles.thresh_val, 'Value', val);
set(handles.text_thresh_val, 'String', get(handles.thresh_val, 'Value'));
guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function text_thresh_val_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_thresh_val (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function thresh_size_Callback(hObject, eventdata, handles)
% hObject    handle to thresh_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
set(handles.text_thresh_size,'String', round(get(handles.thresh_size,'Value')));

axes(handles.Working_Axes);
ThreshSize = round(get(handles.thresh_size, 'Value'));
Val = get(handles.thresh_val, 'Value');
ThreshVal = round(Val*100)/100;
BW = sauvola(handles.data, [ThreshSize ThreshSize], ThreshVal);
BW = imcomplement(BW);
imshow(BW);

Thresh_Window = regionprops(BW, 'EquivDiameter'); % Extract area and centroid data for binary blobs
EquivDiameter = (cat(1, Thresh_Window.EquivDiameter));

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(EquivDiameter, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
xlabel('Equivalent Diameter (pixels)');
ylabel('Count');


% --- Executes during object creation, after setting all properties.
function thresh_size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to thresh_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function thresh_val_Callback(hObject, eventdata, handles)
% hObject    handle to thresh_val (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
Val = get(handles.thresh_val, 'Value');
Roundto100th = round(Val*100)/100;
set(handles.text_thresh_val,'String',Roundto100th);

axes(handles.Working_Axes);
ThreshSize = round(get(handles.thresh_size, 'Value'));
Val = get(handles.thresh_val, 'Value');
ThreshVal = round(Val*100)/100;
BW = sauvola(handles.data, [ThreshSize ThreshSize], ThreshVal);
BW = imcomplement(BW);
imshow(BW);

Thresh_Value = regionprops(BW, 'EquivDiameter'); % Extract area and centroid data for binary blobs
EquivDiameter = (cat(1, Thresh_Value.EquivDiameter));

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(EquivDiameter, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
xlabel('Equivalent Diameter (pixels)');
ylabel('Count');


% --- Executes during object creation, after setting all properties.
function thresh_val_CreateFcn(hObject, eventdata, handles)
% hObject    handle to thresh_val (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function text_median_size_Callback(hObject, eventdata, handles)
% hObject    handle to text_median_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_median_size as text
%        str2double(get(hObject,'String')) returns contents of text_median_size as a double
val = str2double(get(hObject, 'String'));
if val > 25
    val = 25
end
if val < 0
    val = 0
end
set(handles.median_size, 'Value', val);
set(handles.text_median_size, 'String', round(get(handles.median_size, 'Value')));
guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function text_median_size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_median_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in border_conn_comp.
function border_conn_comp_Callback(hObject, eventdata, handles)
% hObject    handle to border_conn_comp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns border_conn_comp contents as cell array
%        contents{get(hObject,'Value')} returns selected item from border_conn_comp

  


% --- Executes during object creation, after setting all properties.
function border_conn_comp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to border_conn_comp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function median_size_Callback(hObject, eventdata, handles)
% hObject    handle to median_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
set(handles.text_median_size,'String',round(get(handles.median_size, 'Value')));

% --- Executes during object creation, after setting all properties.
function median_size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to median_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in set_defaults.
function set_defaults_Callback(hObject, eventdata, handles)
% hObject    handle to set_defaults (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of set_defaults
set(handles.bilateral_size, 'Value', 8);
set(handles.sigma_spatial, 'Value', 6);
set(handles.sigma_intensity, 'Value', 0.25);
set(handles.gaussian_window_size, 'Value', 8)
set(handles.gauss_sigma,'Value', 3);
set(handles.thresh_size,  'Value', 25);
set(handles.thresh_val, 'Value', 0.07);
set(handles.median_size, 'Value', 5);
set(handles.text_bilateral_size, 'String', 8);
set(handles.text_sigma_spatial, 'String', 6);
set(handles.text_sigma_intensity, 'String', 0.25);
set(handles.text_gauss_window, 'String', 8);
set(handles.text_gauss_sigma, 'String', 3);
set(handles.text_thresh_size, 'String', 25);
set(handles.text_thresh_val, 'String', 0.07);
set(handles.text_median_size, 'String', 5);
set(handles.border_conn_comp, 'Value', 1)


% --- Executes on button press in batch_process.
function batch_process_Callback(hObject, eventdata, handles)
% hObject    handle to batch_process (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ImageFolder = uigetdir;
ImageFiles = [dir(fullfile(ImageFolder,'*tif')); dir(fullfile(ImageFolder,'*bmp'))];

% Dialog box
prompt = {'Name Excel Data File:'};
dlg_title = 'Fission Bubble GUI Data';
num_lines = 1;
def = {'FIB_Data'};
ExcelFile = inputdlg(prompt,dlg_title,num_lines,def);

Pixels = handles.ScaleBarPixels;
um = handles.ScaleBarMicrometers;
Pixels = str2double(cell2mat(Pixels));
um = str2double(cell2mat(um));
PixeluMRatio = Pixels / um;

   
if Pixels > 1
    Pixels = Pixels;
else
    choice = questdlg('The scale bar dimensions were not updated. Feature data will use pixels as the default unit. Continue anyways?','Unit Conversion','Yes','No', 'No');
    switch choice
        case 'Yes'
            Pixels = Pixels;
        case 'No'
            return
    end
end

%%%%%%%%%%%%% Write Parameter Text File %%%%%%%%%%%%%%%%%
Extension = sprintf('%s_Parameter_Notes.txt',ExcelFile{1});
TextFileName = fullfile(ImageFolder, Extension);
fid = fopen(TextFileName, 'wt' );
A1 = get(handles.text_bilateral_size, 'String');
A2 = get(handles.text_sigma_spatial, 'String');
A3 = get(handles.text_sigma_intensity, 'String');
A4 = get(handles.text_gauss_window, 'String');
A5 = get(handles.text_gauss_sigma, 'String');
A6 = get(handles.text_thresh_size, 'String');
A7 = get(handles.text_thresh_val, 'String');
A8 = get(handles.text_median_size, 'String');
ConnContents = get(handles.border_conn_comp,'String'); 
ConnValue = ConnContents{get(handles.border_conn_comp,'Value')};
A9 = ConnValue;
A10 = Pixels;
A11 = um;
fprintf(fid, 'Bilateral Filter Window Size: %s\r\n', A1);
fprintf(fid, 'Bilateral Filter Spatial Std Dev: %s\r\n', A2);
fprintf(fid, 'Bilateral Filter Intensity Std Dev: %s\r\n', A3);
fprintf(fid, 'Gaussian Blur Window Size: %s\r\n', A4);
fprintf(fid, 'Gaussian Blur Std Dev: %s\r\n', A5);
fprintf(fid, 'Sauvola Threshold Window Size: %s\r\n', A6);
fprintf(fid, 'Sauvola Threshold Value: %s\r\n', A7);
fprintf(fid, 'Depeckle Window Size: %s\r\n',A8);
fprintf(fid, 'Border Clear Pixel Connectivity: %s\r\n', A9);
fprintf(fid, 'Scale Bar (pixels, um): %d, %d\r\n', A10, A11);
fclose(fid);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% SEQUENCE LOAD IN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:length(ImageFiles) % Open sequence loop
  baseFileName = ImageFiles(k).name;
  fullFileName = fullfile(ImageFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', baseFileName);
  
  ImageName{k} = baseFileName;
  
  I = imread(fullFileName); % Read in image.
  axes(handles.Original_Image)
  imshow(I)
  
  ImageInfo = imfinfo(fullFileName);
        set(handles.image, 'string', baseFileName);
        set(handles.path, 'string', ImageFolder);
        FileModDate = ImageInfo.FileModDate;
        set(handles.mod_date, 'string', FileModDate);
        FileSize = ImageInfo.FileSize/(10^6);
        set(handles.size_MB, 'string', FileSize);
        Format = ImageInfo.Format;
        set(handles.format, 'string', Format);
        Width = ImageInfo.Width;
        Height = ImageInfo.Height;
        set(handles.dimension, 'string', sprintf('%d x %d', Height, Width));
        BitDepth = ImageInfo.BitDepth;
        set(handles.bit_depth, 'string', BitDepth);
%         Compression = ImageInfo.Compression;       
%         set(handles.compression, 'string', Compression);
         
        
   Dimension = size(I,3); % Convert to grayscale if RGB, otherwise do nothing
  if Dimension >= 3
      I = rgb2gray(I);
  end
  
  Rectangle = handles.Coordinates;
    if Rectangle > 0
        I = imcrop(I, Rectangle);
    else
    end
  axes(handles.Working_Axes)
  imshow(I)
  axes(handles.Histogram)
  imhist(I)
  
%%%%%%%%%%%%%%%%%%%%%%%%%%% PRE-PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
ContAdj = imadjust(Current_Image, stretchlim(Current_Image, [0.01 0.99]));
imshow(ContAdj);
axes(handles.Histogram);
imhist(ContAdj);

Current_Image = getimage(handles.Working_Axes);
DP = double(Current_Image)/255;
BilateralSize = round(get(handles.bilateral_size, 'Value'));
SigmaSpatial = round(get(handles.sigma_spatial, 'Value'));
SigmaIntensity = get(handles.sigma_intensity, 'Value');
B = bfilter2((DP), BilateralSize, [SigmaSpatial SigmaIntensity]);
axes(handles.Working_Axes);
imshow(B);
axes(handles.Histogram);
imhist(B);

Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
BlurSize = round(get(handles.gaussian_window_size, 'Value'));
BlurSigma = round(get(handles.gauss_sigma, 'Value'));
h = fspecial('gaussian', [BlurSize BlurSize], BlurSigma);
Blur = imfilter(Current_Image, h);
imshow(Blur);
axes(handles.Histogram);
imhist(Blur);

%%%%%%%%%%%%%%%%%%%%%%%%%%% SEGMENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
ThreshSize = round(get(handles.thresh_size, 'Value'));
Val = get(handles.thresh_val, 'Value');
ThreshVal = round(Val*100)/100;
BW = sauvola(Current_Image, [ThreshSize ThreshSize], ThreshVal);
BW = imcomplement(BW);
imshow(BW);

Voids = regionprops(BW, 'EquivDiameter'); % Extract area and centroid data for binary blobs
GUIPlot = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(GUIPlot, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Count');

%%%%%%%%%%%%%%%%%%%%%%%%%%% POST-PROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
MedianSize = round(get(handles.median_size, 'Value'));
Despeckle = medfilt2(Current_Image, [MedianSize MedianSize]);
imshow(Despeckle);
axes(handles.Histogram);
imhist(Despeckle);

Voids = regionprops(BW, 'EquivDiameter'); % Extract area and centroid data for binary blobs
GUIPlot = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution binDiameters] = hist(GUIPlot, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Count');

Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
BWfill = imfill(Current_Image, 'holes');
imshow(BWfill);
axes(handles.Histogram);
imhist(BWfill);

Current_Image = getimage(handles.Working_Axes);
axes(handles.Working_Axes);
ConnCompContents = get(handles.border_conn_comp, 'String');
ConnectedComponentChar = ConnCompContents{get(handles.border_conn_comp, 'Value')};
ConnectedComponentNum = str2double(ConnectedComponentChar);
ClearBorder = imclearborder(Current_Image, ConnectedComponentNum);
imshow(ClearBorder);
axes(handles.Histogram);
imhist(ClearBorder);

Voids = regionprops(ClearBorder, 'EquivDiameter'); % Extract area and centroid data for binary blobs
GUIPlot = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);

axes(handles.Histogram);
numberOfBins = 150; % Set histogram bin quantity
[AreaDistribution, binDiameters] = hist(GUIPlot, numberOfBins);
bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Count');

Current_Image = getimage(handles.Working_Axes);
Bubbles = bwlabel(Current_Image); % Label image

%%%%%%%%%%%%%%%%%%%% MORPHOLOGICAL DATA EXTRACTION %%%%%%%%%%%%%%%%%%%%
Voids = regionprops(Bubbles, 'Area', 'Centroid', 'EquivDiameter', 'Eccentricity', 'Perimeter'); % Extract area and centroid data for binary blobs
Centroids = cat(1, Voids.Centroid); % Concatenate into single column matrix
axes(handles.Working_Axes)
hold on
plot(Centroids(:,1), Centroids(:,2), 'b*'); % Overlay centroids onto image
hold off

% Concatenate area data into single column matrix
Area = (cat(1, Voids.Area)); 
Number = size(Area,1);

Areas{k} = (cat(1, Voids.Area)) / (PixeluMRatio*PixeluMRatio);
EquivDiameter{k} = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);
EquivDiameterIndividual = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);
Eccentricity{k} = cat(1,Voids.Eccentricity);
EccentricityIndividual = cat(1,Voids.Eccentricity);

Count(k) = size(Area,1); % Object count
PercentPorousVolume(k) = (sum(Area) / (size(Current_Image,1)*size(Current_Image,2)))*100;
EccentricityMean(k) = mean(EccentricityIndividual);
EquivDiameterMean(k) = mean(EquivDiameterIndividual);
SizeMean(k) = mean(Area)/(PixeluMRatio*PixeluMRatio);

% SizeMean(k) = mean(Areas);
% SizeMedian(k) = median(Areas);
% SizeMode(k) = mode(Areas);
% EquivDiameterMedian(k) = median(EquivDiameter);
% EquivDiameterMode(k) = mode(EquivDiameter);
% Major = (cat(1, Voids.MajorAxisLength));
% MajorMean(k) = mean(Major);
% MajorMedian(k) = median(Major);
% MajorMode(k) = mode(Major);
% Minor = (cat(1, Voids.MinorAxisLength));
% MinorMean(k) = mean(Minor);
% MinorMedian(k) = median(Minor);
% MinorMode(k) = mode(Minor);
% ConvexArea = cat(1, Voids.ConvexArea);
% ConvexAreaMean(k) = mean(ConvexArea);
% ConvexAreaMedian(k) = median(ConvexArea);
% ConvexAreaMode(k) = mode(ConvexArea);

% MATLAB File Data Export


%%%%%%%%%%%%%%%%%%% IMAGE SAVE/WRITE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Pathstr,Name,Ext] = fileparts(fullFileName);
ReLabel = sprintf('%s_Binary%s', Name, Ext);
NewFileName = fullfile(Pathstr,ReLabel);
ImageNameShort = fullfile(Name,Ext);
imwrite(Bubbles, NewFileName);

% SizeDist = figure;
% iptsetpref('ImshowBorder','tight'); % Allow border around image for labeling
% subplot(1,2,2)
% histogram(EccentricityIndividual, 50, 'Normalization','probability');
% Title1 = sprintf('Bubble Morphology Distribution');
% title(Title1,'interpreter','none','FontSize',10); % Original input image
% xlabel('Eccentricity');
% ylabel('Normalized Frequency');
% subplot(1,2,1)
% histogram(EquivDiameterIndividual, 50, 'Normalization', 'probability', 'FaceColor', 'r');
% Title2 = sprintf('Bubble Size Distribution');
% title(Title2,'interpreter','none','FontSize',10); % Final output image
% if Pixels > 1;
% xlabel('Equivalent Diameter (um)');
% else xlabel('Equivalent Diameter (pixels)');
% end
% ylabel('Normalized Frequency');
% set(gcf,'units','normalized','outerposition',[0 0 1 1]);
% Path = fullfile(Pathstr,Name);
% Title3 = sprintf('%s_Size Distributions%s',Path,Ext);
% saveas(gcf, Title3) % Export and save comparitave image
% close(SizeDist)

end

%%%%%%%%%%%%%%%%% DATA EXPORT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DataFile = 'GUITest.mat'; % Name of Data Set. Uncomment for .mat file save
% save(DataFile,'ImageName', 'Count', 'PercentPorousVolume', 'Size')

% Excel Export Destination and Data
FileName = fullfile(ImageFolder, ExcelFile{1});
Extension = sprintf('%s.xlsx',ExcelFile{1});
XLSXFileName = fullfile(ImageFolder, Extension);

SheetNames = {'Data Summary', 'Area (Full Data)', 'Equivalent Diameter (Full Data)', 'Eccentricity (Full Data)'};
xlsheets(SheetNames,FileName)

xlswrite(XLSXFileName,{'Image'},1,'A1') % Column Headers
xlswrite(XLSXFileName,{'Count'},1,'B1')
xlswrite(XLSXFileName,{'Porosity'},1,'C1')
xlswrite(XLSXFileName,{'AreaMean'},1,'D1')
% xlswrite(FileName,{'SizeMedian'},1,'E1')
% xlswrite(FileName,{'SizeMode'},1,'F1')
xlswrite(XLSXFileName,{'EDMean'},1,'E1')
xlswrite(XLSXFileName,{'EccentricityMean'},1,'F1')
% xlswrite(FileName,{'EDMedian'},1,'H1')
% xlswrite(FileName,{'EDMode'},1,'I1')
% xlswrite(FileName,{'MajorMean'},1,'J1')
% xlswrite(FileName,{'MajorMedian'},1,'K1')
% xlswrite(FileName,{'MajorMode'},1,'L1')
% xlswrite(FileName,{'MinorMean'},1,'M1')
% xlswrite(FileName,{'MinorMedian'},1,'N1')
% xlswrite(FileName,{'MinorMode'},1,'O1')
% xlswrite(FileName,{'CAMean'},1,'P1')
% xlswrite(FileName,{'CAMedian'},1,'Q1')
% xlswrite(FileName,{'CAMode'},1,'R1')

xlswrite(XLSXFileName,ImageName',1,'A2') % Data entry
xlswrite(XLSXFileName,Count',1,'B2')
xlswrite(XLSXFileName,PercentPorousVolume',1,'C2')
xlswrite(XLSXFileName,SizeMean',1,'D2')
% xlswrite(FileName,SizeMedian',1,'E2')
% xlswrite(FileName,SizeMode',1,'F2')
xlswrite(XLSXFileName,EquivDiameterMean',1,'E2')
xlswrite(XLSXFileName,EccentricityMean', 1, 'F2')
% xlswrite(FileName,EquivDiameterMedian',1,'H2')
% xlswrite(FileName,EquivDiameterMode',1,'I2')
% xlswrite(FileName,MajorMean',1,'J2')
% xlswrite(FileName,MajorMedian',1,'K2')
% xlswrite(FileName,MajorMode',1,'L2')
% xlswrite(FileName,MinorMean',1,'M2')
% xlswrite(FileName,MinorMedian',1,'N2')
% xlswrite(FileName,MinorMode',1,'O2')
% xlswrite(FileName,ConvexAreaMean',1,'P2')
% xlswrite(FileName,ConvexAreaMedian',1,'Q2')
% xlswrite(FileName,ConvexAreaMode',1,'R2')

AreaMatrix = padcat(Areas{:});
EquivDiameterMatrix = padcat(EquivDiameter{:});
EccentricityMatrix = padcat(Eccentricity{:});
xlswrite(XLSXFileName, ImageName, 2, 'A1')
xlswrite(XLSXFileName, AreaMatrix, 2, 'A2')
xlswrite(XLSXFileName, ImageName, 3, 'A1')
xlswrite(XLSXFileName, EquivDiameterMatrix, 3, 'A2')
xlswrite(XLSXFileName, ImageName, 4, 'A1')
xlswrite(XLSXFileName, EccentricityMatrix, 4, 'A2')

SizeDist = figure;
iptsetpref('ImshowBorder','tight'); % Allow border around image for labeling
subplot(1,2,2)
EccenList = reshape(EccentricityMatrix, prod(size(EccentricityMatrix)), 1);
histogram(EccenList, 50, 'Normalization','probability');
Title1 = sprintf('Bubble Morphology Distribution');
title(Title1,'interpreter','none','FontSize',10); % Original input image
xlabel('Eccentricity');
ylabel('Normalized Frequency');
subplot(1,2,1)
EDList = reshape(EquivDiameterMatrix, prod(size(EquivDiameterMatrix)), 1);
histogram(EDList, 50, 'Normalization', 'probability', 'FaceColor', 'r');
Title2 = sprintf('Bubble Size Distribution');
title(Title2,'interpreter','none','FontSize',10); % Final output image
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Normalized Frequency');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
Path = fullfile(Pathstr,Name);
Title3 = sprintf('%s/%s_Size Distributions%s',Pathstr,ExcelFile{1},Ext);
saveas(gcf, Title3) % Export and save comparitave image
close(SizeDist)



% --- Executes on button press in binary_data.
function binary_data_Callback(hObject, eventdata, handles)
% hObject    handle to binary_data (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ImageFolder = get(handles.path,'string');
Image = get(handles.image,'string');
fullFileName = fullfile(ImageFolder,Image);

prompt = {'Name Excel Data File:'};
dlg_title = 'Fission Bubble GUI Data';
num_lines = 1;
def = {'Single_Image_Data'};
ExcelFile = inputdlg(prompt,dlg_title,num_lines,def);

Pixels = handles.ScaleBarPixels;
um = handles.ScaleBarMicrometers;
Pixels = str2double(cell2mat(Pixels));
um = str2double(cell2mat(um));
PixeluMRatio = Pixels / um;

   
if Pixels > 1
    Pixels = Pixels;
else
    choice = questdlg('The scale bar dimensions were not updated. Feature data will use pixels as the default unit. Continue anyways?','Unit Conversion','Yes','No', 'No');
    switch choice
        case 'Yes'
            Pixels = Pixels;
        case 'No'
            return
    end
end

%%%%%%%%%%%%% Write Parameter Text File %%%%%%%%%%%%%%%%%
Extension = sprintf('%s_Parameter_Notes.txt',ExcelFile{1});
TextFileName = fullfile(ImageFolder, Extension);
fid = fopen(TextFileName, 'wt' );
A1 = get(handles.text_bilateral_size, 'String');
A2 = get(handles.text_sigma_spatial, 'String');
A3 = get(handles.text_sigma_intensity, 'String');
A4 = get(handles.text_gauss_window, 'String');
A5 = get(handles.text_gauss_sigma, 'String');
A6 = get(handles.text_thresh_size, 'String');
A7 = get(handles.text_thresh_val, 'String');
A8 = get(handles.text_median_size, 'String');
ConnContents = get(handles.border_conn_comp,'String'); 
ConnValue = ConnContents{get(handles.border_conn_comp,'Value')};
A9 = ConnValue;
A10 = Pixels;
A11 = um;
fprintf(fid, 'Bilateral Filter Window Size: %s\r\n', A1);
fprintf(fid, 'Bilateral Filter Spatial Std Dev: %s\r\n', A2);
fprintf(fid, 'Bilateral Filter Intensity Std Dev: %s\r\n', A3);
fprintf(fid, 'Gaussian Blur Window Size: %s\r\n', A4);
fprintf(fid, 'Gaussian Blur Std Dev: %s\r\n', A5);
fprintf(fid, 'Sauvola Threshold Window Size: %s\r\n', A6);
fprintf(fid, 'Sauvola Threshold Value: %s\r\n', A7);
fprintf(fid, 'Depeckle Window Size: %s\r\n',A8);
fprintf(fid, 'Border Clear Pixel Connectivity: %s\r\n', A9);
fprintf(fid, 'Scale Bar (pixels, um): %d, %d\r\n', A10, A11);
fclose(fid);

Current_Image = getimage(handles.Working_Axes);
Bubbles = bwlabel(Current_Image); % Label image

%%%%%%%%%%%%%%%%%%%% MORPHOLOGICAL DATA EXTRACTION %%%%%%%%%%%%%%%%%%%%
Voids = regionprops(Bubbles, 'Area', 'Centroid', 'EquivDiameter', 'Eccentricity'); % Extract area and centroid data for binary blobs
Centroids = cat(1, Voids.Centroid); % Concatenate into single column matrix
axes(handles.Working_Axes)
hold on
plot(Centroids(:,1), Centroids(:,2), 'b*'); % Overlay centroids onto image
hold off

Areas = cat(1, Voids.Area); % Concatenate area data into single column matrix
Count = size(Areas,1); % Object count
PercentPorousVolume = (sum(Areas) / (size(Current_Image,1)*size(Current_Image,2)))*100;
Number = size(Areas,1);% Calculate sample porosity
SizeMean = mean(Areas)/(PixeluMRatio*PixeluMRatio);

EquivDiameter = (cat(1, Voids.EquivDiameter)) / (PixeluMRatio);
EquivDiameterMean = mean(EquivDiameter);

Eccentricity = cat(1,Voids.Eccentricity);
EccentricityMean = mean(Eccentricity);

[Pathstr,Name,Ext] = fileparts(fullFileName);
ReLabel = sprintf('%s_Binary%s', Name, Ext);
NewFileName = fullfile(Pathstr,ReLabel);
ImageNameShort = fullfile(Name,Ext);
imwrite(Bubbles, NewFileName);

% MATLAB File Data Export
save('ExcelFile','ImageNameShort','EquivDiameter','Eccentricity');

% Excel Export Destination and Data
FileName = fullfile(ImageFolder, ExcelFile{1});
Extension = sprintf('%s.xlsx',ExcelFile{1});
XLSXFileName = fullfile(ImageFolder, Extension);

SheetNames = {'Data Summary', 'Area (Full Data)', 'Equivalent Diameter (Full Data)', 'Eccentricity (Full Data)'};
xlsheets(SheetNames,FileName)

xlswrite(XLSXFileName,{'Image'},1,'A1') % Column Headers
xlswrite(XLSXFileName,{'Count'},1,'B1')
xlswrite(XLSXFileName,{'Porosity'},1,'C1')
xlswrite(XLSXFileName,{'AreaMean'},1,'D1')
xlswrite(XLSXFileName,{'EDMean'},1,'E1')
xlswrite(XLSXFileName,{'EccentricityMean'},1,'F1')

xlswrite(XLSXFileName,{Image},1,'A2') % Data entry
xlswrite(XLSXFileName,Count,1,'B2')
xlswrite(XLSXFileName,PercentPorousVolume,1,'C2')
xlswrite(XLSXFileName,SizeMean,1,'D2')
xlswrite(XLSXFileName,EquivDiameterMean',1,'E2')
xlswrite(XLSXFileName,EccentricityMean, 1, 'F2')

AreaMatrix = Areas(:)/(PixeluMRatio*PixeluMRatio);
EquivDiameterMatrix = EquivDiameter(:);
EccentricityMatrix = Eccentricity(:);
xlswrite(XLSXFileName, {'Raw Areas'}, 2, 'A2')
xlswrite(XLSXFileName, AreaMatrix, 2, 'A3')
xlswrite(XLSXFileName, {'Raw Equivalent Diameters'}, 3, 'A2')
xlswrite(XLSXFileName, EquivDiameterMatrix, 3, 'A3')
xlswrite(XLSXFileName, {'Raw Eccentricities'}, 4, 'A2')
xlswrite(XLSXFileName, EccentricityMatrix, 4, 'A3')

% numberOfBins = 150; % Set histogram bin quantity
% [AreaDistribution binDiameters] = hist(EquivDiameter, numberOfBins);
% bar(binDiameters, AreaDistribution, 'BarWidth', 1.0);
% set(handles.distribution_plot_text, 'String', 'Fission Gas Bubble Size Distribution');
% xlabel('Equivalent Diameter (pixels)');
% ylabel('Count');

SizeDist = figure;
iptsetpref('ImshowBorder','tight'); % Allow border around image for labeling
subplot(1,2,2)
histogram(Eccentricity, 50, 'Normalization','probability');
Title1 = sprintf('Bubble Morphology Distribution');
title( [Title1],'interpreter','none','FontSize',10); % Original input image
xlabel('Eccentricity');
ylabel('Normalized Frequency');
subplot(1,2,1)
histogram(EquivDiameter, 50, 'Normalization', 'probability', 'FaceColor', 'r');
Title2 = sprintf('Bubble Size Distribution');
title( [Title2],'interpreter','none','FontSize',10); % Final output image
if Pixels > 1;
xlabel('Equivalent Diameter (um)');
else xlabel('Equivalent Diameter (pixels)');
end
ylabel('Normalized Frequency');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
Path = fullfile(Pathstr,Name);
Title3 = sprintf('%s_Size Distributions%s',Path,Ext);
saveas(gcf, Title3) % Export and save comparitave image
set(gcf, 'Visible', 'Off');



% --- Executes on button press in crop.
function crop_Callback(hObject, eventdata, handles)
% hObject    handle to crop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[Cropped rect] = imcrop(handles.Working_Axes);

axes(handles.Working_Axes);
imshow(Cropped)
axes(handles.Histogram);
imhist(Cropped)
axes(handles.Original_Image);
imshow(Cropped)

handles.data = Cropped;
handles.Coordinates = rect;
guidata(hObject, handles)




function size_MB_Callback(hObject, eventdata, handles)
% hObject    handle to size_MB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of size_MB as text
%        str2double(get(hObject,'String')) returns contents of size_MB as a double


% --- Executes during object creation, after setting all properties.
function size_MB_CreateFcn(hObject, eventdata, handles)
% hObject    handle to size_MB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function format_Callback(hObject, eventdata, handles)
% hObject    handle to format (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of format as text
%        str2double(get(hObject,'String')) returns contents of format as a double


% --- Executes during object creation, after setting all properties.
function format_CreateFcn(hObject, eventdata, handles)
% hObject    handle to format (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function dimension_Callback(hObject, eventdata, handles)
% hObject    handle to dimension (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of dimension as text
%        str2double(get(hObject,'String')) returns contents of dimension as a double


% --- Executes during object creation, after setting all properties.
function dimension_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dimension (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function bit_depth_Callback(hObject, eventdata, handles)
% hObject    handle to bit_depth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bit_depth as text
%        str2double(get(hObject,'String')) returns contents of bit_depth as a double


% --- Executes during object creation, after setting all properties.
function bit_depth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bit_depth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function compression_Callback(hObject, eventdata, handles)
% hObject    handle to compression (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of compression as text
%        str2double(get(hObject,'String')) returns contents of compression as a double


% --- Executes during object creation, after setting all properties.
function compression_CreateFcn(hObject, eventdata, handles)
% hObject    handle to compression (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function mod_date_Callback(hObject, eventdata, handles)
% hObject    handle to mod_date (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mod_date as text
%        str2double(get(hObject,'String')) returns contents of mod_date as a double


% --- Executes during object creation, after setting all properties.
function mod_date_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mod_date (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function distribution_plot_text_Callback(hObject, eventdata, handles)
% hObject    handle to distribution_plot_text (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of distribution_plot_text as text
%        str2double(get(hObject,'String')) returns contents of distribution_plot_text as a double


% --- Executes during object creation, after setting all properties.
function distribution_plot_text_CreateFcn(hObject, eventdata, handles)
% hObject    handle to distribution_plot_text (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function binary_data_CreateFcn(hObject, eventdata, handles)
% hObject    handle to binary_data (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
