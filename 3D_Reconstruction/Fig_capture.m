%% Capture Figure
[filename, pathname] = uigetfile('*.mat', 'Select data file for imaging');
cd(pathname);
load(filename);

%Set shp_list# = shp_list in command window to save if using multiple
%stacks
%Repeat for each data file

%% For single upload
name = input('Figure name: ');
shp_list = input('Load list number: ');
figure('Name', name, 'NumberTitle', 'off');
hold on
set(gcf, 'color', 'w');
for i = 1:length(shp_list)

    plot(shp_list{i,1},'EdgeAlpha', 0.1, 'FaceColor', shp_list{i,2});

end
axis tight

ax = gca;
xlim = ax.XLim*nm_per_pix/1000
ylim = ax.YLim*nm_per_pix/1000
zlim = ax.ZLim*nm_per_pix/1000

x_ax = input('X Axis values (vector): ');
y_ax = input('Y Axis values (vector): ');
z_ax = input('Z Axis values (vector): ');
ax.XAxis.TickValues = x_ax*1000/nm_per_pix;
ax.XAxis.TickLabels = ax.XAxis.TickValues*nm_per_pix/1000;

ax.YAxis.TickValues = y_ax*1000/nm_per_pix;
ax.YAxis.TickLabels = ax.YAxis.TickValues*nm_per_pix/1000;

ax.ZAxis.TickValues = z_ax*1000/nm_per_pix;
ax.ZAxis.TickLabels = ax.ZAxis.TickValues*nm_per_pix/1000;

view(290, 30);

%For rotation run view(az, el); in command window. view(45, 30);

%Add scale bar

hold off

%% For multiple upload
nm_per_pix = input('Pixel size (nm/pix): ');
figure('Name', 'Reconstruction', 'NumberTitle', 'off');
hold on
for i = 1:length(shp_list1)
    plot(shp_list1{i,1},'EdgeAlpha', 0.1, 'FaceColor', shp_list1{i,2});
end

for i = 1:length(shp_list2)
    plot(shp_list2{i,1},'EdgeAlpha', 0.1, 'FaceColor', shp_list2{i,2});
end

for i = 1:length(shp_list3)
    plot(shp_list3{i,1},'EdgeAlpha', 0.1, 'FaceColor', shp_list3{i,2});
end
axis tight

ax = gca;
xlim = ax.XLim*nm_per_pix/1000
ylim = ax.YLim*nm_per_pix/1000
zlim = ax.ZLim*nm_per_pix/1000

x_ax = input('X Axis values (vector): ');
y_ax = input('Y Axis values (vector): ');
z_ax = input('Z Axis values (vector): ');

ax.XAxis.TickValues = x_ax*1000/nm_per_pix;
ax.XAxis.TickLabels = ax.XAxis.TickValues*nm_per_pix/1000;

ax.YAxis.TickValues = y_ax*1000/nm_per_pix;
ax.YAxis.TickLabels = ax.YAxis.TickValues*nm_per_pix/1000;

ax.ZAxis.TickValues = z_ax*1000/nm_per_pix;
ax.ZAxis.TickLabels = ax.ZAxis.TickValues*nm_per_pix/1000;

view(290, 30);

% Add scale bar

hold off

%% Save
path = uigetdir(pwd, 'Select Save Location');
cd(path);
output = input('Filename: ');
saveas(gcf, output, 'png');