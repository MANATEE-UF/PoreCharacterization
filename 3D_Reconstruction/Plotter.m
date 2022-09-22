function varargout = Plotter(varargin)
% PLOTTER MATLAB code for Plotter.fig
%      PLOTTER, by itself, creates a new PLOTTER or raises the existing
%      singleton*.
%
%      H = PLOTTER returns the handle to a new PLOTTER or the handle to
%      the existing singleton*.
%
%      PLOTTER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLOTTER.M with the given input arguments.
%
%      PLOTTER('Property','Value',...) creates a new PLOTTER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Plotter_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Plotter_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Plotter

% Last Modified by GUIDE v2.5 11-Oct-2017 10:48:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Plotter_OpeningFcn, ...
                   'gui_OutputFcn',  @Plotter_OutputFcn, ...
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


% --- Executes just before Plotter is made visible.
function Plotter_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Plotter (see VARARGIN)

% Choose default command line output for Plotter
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Plotter wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Plotter_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;





function vid_duration_Callback(hObject, eventdata, handles)

function vid_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to vid_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function vid_framerate_Callback(hObject, eventdata, handles)

function vid_framerate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to vid_framerate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tilt_angles_Callback(hObject, eventdata, handles)

function tilt_angles_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tilt_angles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function rot_angles_Callback(hObject, eventdata, handles)

function rot_angles_CreateFcn(hObject, eventdata, handles)
% hObject    handle to rot_angles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function vid_filename_Callback(hObject, eventdata, handles)

function vid_filename_CreateFcn(hObject, eventdata, handles)
% hObject    handle to vid_filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function SaveVideo_Callback(hObject, eventdata, handles)
% hObject    handle to SaveVideo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
OptionZ.Duration = str2num(get(handles.vid_duration, 'String'));
OptionZ.FrameRate = str2num(get(handles.vid_framerate, 'String'));
OptionZ.Periodic = true;
filename = get(handles.vid_filename, 'String');
tilt = str2num(get(handles.tilt_angles, 'String'));
rot = str2num(get(handles.rot_angles, 'String'));

for i =1:min(length(tilt), length(rot));
    ang(i,1:2) = [rot(i) tilt(i)];
end

set(0,'currentfigure', handles.p_fig);
CaptureFigVid(ang,filename,OptionZ);


function browse_Callback(hObject, eventdata, handles)
% hObject    handle to browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile('*.mat', 'Select data file for imaging');
cd(pathname);
set(handles.filepath,'String', [pathname filename]);


load(filename);



handles.p_fig = figure('Name', 'Pores', 'NumberTitle', 'off');
hold on
for i = 1:length(shp_list)
    plot(shp_list{i,1},'EdgeAlpha', 0.1, 'FaceColor', shp_list{i,2});
end
axis tight
hold off


guidata(hObject,handles);


function filepath_Callback(hObject, eventdata, handles)

function filepath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filepath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function exit_Callback(hObject, eventdata, handles)
% hObject    handle to exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close all force;

% --- Executes on button press in LoadAdjacent.
function LoadAdjacent_Callback(hObject, eventdata, handles)
% hObject    handle to LoadAdjacent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
k_props = handles.k_props;

target = str2double(get(handles.p_obj_1, 'String'));
adj_list = k_props(target).AdjKern;

if length(adj_list)<19
    adj_list(end+1:19) = 1;
end

set(handles.p_obj_2, 'String', num2str(adj_list(1)));
set(handles.p_obj_3, 'String', num2str(adj_list(2)));
set(handles.p_obj_4, 'String', num2str(adj_list(3)));
set(handles.p_obj_5, 'String', num2str(adj_list(4)));
set(handles.p_obj_6, 'String', num2str(adj_list(5)));
set(handles.p_obj_7, 'String', num2str(adj_list(6)));
set(handles.p_obj_8, 'String', num2str(adj_list(7)));
set(handles.p_obj_9, 'String', num2str(adj_list(8)));
set(handles.p_obj_10, 'String', num2str(adj_list(9)));
set(handles.m_obj_1, 'String', num2str(adj_list(10)));
set(handles.m_obj_2, 'String', num2str(adj_list(11)));
set(handles.m_obj_3, 'String', num2str(adj_list(12)));
set(handles.m_obj_4, 'String', num2str(adj_list(13)));
set(handles.m_obj_5, 'String', num2str(adj_list(14)));
set(handles.m_obj_6, 'String', num2str(adj_list(15)));
set(handles.m_obj_7, 'String', num2str(adj_list(16)));
set(handles.m_obj_8, 'String', num2str(adj_list(17)));
set(handles.m_obj_9, 'String', num2str(adj_list(18)));
set(handles.m_obj_10, 'String', num2str(adj_list(19)));


% --- Executes on button press in AutoRGB.
function AutoRGB_Callback(hObject, eventdata, handles)
% hObject    handle to AutoRGB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.p_col_1, 'String', '1 0 0');
set(handles.p_col_2, 'String', '0 1 0');
set(handles.p_col_3, 'String', '0 0 1');
set(handles.p_col_4, 'String', '1 1 0');
set(handles.p_col_5, 'String', '1 0 1');
set(handles.p_col_6, 'String', '0 1 1');
set(handles.p_col_7, 'String', '1 .5 0');
set(handles.p_col_8, 'String', '1 0 .5');
set(handles.p_col_9, 'String', '.5 1 0');
set(handles.p_col_10, 'String', '.5 0 1');
set(handles.m_col_1, 'String', '0 .5 1');
set(handles.m_col_2, 'String', '0 1 .5');
set(handles.m_col_3, 'String', '.5 0 0');
set(handles.m_col_4, 'String', '0 .5 0');
set(handles.m_col_5, 'String', '0 0 .5');
set(handles.m_col_6, 'String', '1 .5 .5');
set(handles.m_col_7, 'String', '.5 .5 1');
set(handles.m_col_8, 'String', '.5 1 .5');
set(handles.m_col_9, 'String', '1 .3 .6');
set(handles.m_col_10, 'String', '0 .5 .5');

set(handles.p_col_1,'BackgroundColor','1 0 0');
set(handles.p_col_2,'BackgroundColor','0 1 0');
set(handles.p_col_3,'BackgroundColor','0 0 1');
set(handles.p_col_4,'BackgroundColor','1 1 0');
set(handles.p_col_5,'BackgroundColor','1 0 1');
set(handles.p_col_6,'BackgroundColor','0 1 1');
set(handles.p_col_7,'BackgroundColor','1 .5 0');
set(handles.p_col_8,'BackgroundColor','1 0 .5');
set(handles.p_col_9,'BackgroundColor','.5 1 0');
set(handles.p_col_10,'BackgroundColor','.5 0 1');
set(handles.m_col_1, 'BackgroundColor', '0 .5 1');
set(handles.m_col_2, 'BackgroundColor', '0 1 .5');
set(handles.m_col_3, 'BackgroundColor', '.5 0 0');
set(handles.m_col_4, 'BackgroundColor', '0 .5 0');
set(handles.m_col_5, 'BackgroundColor', '0 0 .5');
set(handles.m_col_6, 'BackgroundColor', '1 .5 .5');
set(handles.m_col_7, 'BackgroundColor', '.5 .5 1');
set(handles.m_col_8, 'BackgroundColor', '.5 1 .5');
set(handles.m_col_9, 'BackgroundColor', '1 .3 .6');
set(handles.m_col_10, 'BackgroundColor', '0 .5 .5');


% --- Executes on button press in matrix_toggle.
function matrix_toggle_Callback(hObject, eventdata, handles)
toggle = get(hObject,'Value');
set(handles.p_zoc_1, 'Value', toggle);
set(handles.p_zoc_2, 'Value', toggle);
set(handles.p_zoc_3, 'Value', toggle);
set(handles.p_zoc_4, 'Value', toggle);
set(handles.p_zoc_5, 'Value', toggle);
set(handles.p_zoc_6, 'Value', toggle);
set(handles.p_zoc_7, 'Value', toggle);
set(handles.p_zoc_8, 'Value', toggle);
set(handles.p_zoc_9, 'Value', toggle);
set(handles.p_zoc_10, 'Value', toggle);
set(handles.m_par_1, 'Value', toggle);
set(handles.m_par_2, 'Value', toggle);
set(handles.m_par_3, 'Value', toggle);
set(handles.m_par_4, 'Value', toggle);
set(handles.m_par_5, 'Value', toggle);
set(handles.m_par_6, 'Value', toggle);
set(handles.m_par_7, 'Value', toggle);
set(handles.m_par_8, 'Value', toggle);
set(handles.m_par_9, 'Value', toggle);
set(handles.m_par_10, 'Value', toggle);


% --- Executes on button press in visible_toggle.
function visible_toggle_Callback(hObject, eventdata, handles)
toggle = get(hObject,'Value');
set(handles.p_vis_1, 'Value', toggle);
set(handles.p_vis_2, 'Value', toggle);
set(handles.p_vis_3, 'Value', toggle);
set(handles.p_vis_4, 'Value', toggle);
set(handles.p_vis_5, 'Value', toggle);
set(handles.p_vis_6, 'Value', toggle);
set(handles.p_vis_7, 'Value', toggle);
set(handles.p_vis_8, 'Value', toggle);
set(handles.p_vis_9, 'Value', toggle);
set(handles.p_vis_10, 'Value', toggle);
set(handles.m_vis_1, 'Value', toggle);
set(handles.m_vis_2, 'Value', toggle);
set(handles.m_vis_3, 'Value', toggle);
set(handles.m_vis_4, 'Value', toggle);
set(handles.m_vis_5, 'Value', toggle);
set(handles.m_vis_6, 'Value', toggle);
set(handles.m_vis_7, 'Value', toggle);
set(handles.m_vis_8, 'Value', toggle);
set(handles.m_vis_9, 'Value', toggle);
set(handles.m_vis_10, 'Value', toggle);

% --- Executes on button press in show_particles.
function show_particles_Callback(hObject, eventdata, handles)
% hObject    handle to show_particles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of show_particles
