clear;
close all;
%Akash Dwivedi
%	Implementation of the PD controller to control the motion of the 2-D  quadrotor 
addpath('utils');
addpath('trajectories');

controlhandle = @controller;

% Choose which trajectory you want to test with
% trajhandle = @traj_line;
trajhandle = @traj_sine;

[t, state] = simulation_2d(controlhandle, trajhandle);