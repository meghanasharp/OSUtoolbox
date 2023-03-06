%Code to run RollingRadon (Nick Holschuh)
%Georgia Carroll 
%3/8/2022

%RollingRadon says: 

% The inputs are as follows:
%
% data_x_or_filename - Either the X-axis or the name of a CReSIS flight
%                      file (x-axis in distance)
% data_y - values for the Y-axis ***(ignored if filename provided)
%          this can be either a twtt or depth
% Data - the data raster         ***(ignored if filename provided)
% window - this defines the size of the rolling window
% angle_thresh - this value is the maximum slope useable;
% plotter - 1, generates the debug plots
% [surface_bottom] - a vector containing the surface and bottom picks
% [movie_flag] - 1, Records the debug plots (must have plotter == 1)
% [max_frequency] - this sets the scale for interpolation, based on the
%                 highest frequency of interest in the data. Can induce
%                 memory problems, and not required.


clc;

%addpath('/Users/crosson/Documents/MATLAB/Radargramming/CSARP_standard');

load Data_20160512_02_009.mat
data_x_or_filename = new_data;
window = 301;
angle_thresh = [5 5]; % maybe in degrees? subtracts 5 at one point??
plotter = 1;
movie_flag = 1;
surface_bottom = 0;
%others


[slopegrid_x,slopegrid_y,slopegrid,opt_x,opt_y,opt_angle]=RollingRadon(new_data,window,angle_thresh,plotter,movie_flag);

%%
% [new_data, shift_amount, depth_axis, surface_elev, bed_elev] = depth_shift(Data,Time,Surface,Elevation,botinter,1);
% % 
% [x,y]=meshgrid(Latitude,depth_axis);
% figure; surf(x,y,log10(new_data),'EdgeColor','interp');
% view(0,90)
% 

