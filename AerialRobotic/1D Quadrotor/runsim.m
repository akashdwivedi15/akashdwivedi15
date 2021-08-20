close all;
clear;

%AkashDwivedi
%Implementation of PD controller to control the height of a quadrotor and tuned gains for 1-D quadrotor 
% Hover
%z_des = 0;

% Step
z_des = 1;

% Given trajectory generator
trajhandle = @(t) fixed_set_point(t, z_des);

% This is your controller
controlhandle = @controller;

% Run simulation with given trajectory generator and controller
[t, z] = height_control(trajhandle, controlhandle);