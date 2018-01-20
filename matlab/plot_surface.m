clc;
clear all;
close all;

c = 1/20216.335877;
f = @(x1,x2) c * exp(-((x1.*x1).*(x2.*x2) + x1.*x1 + x2.*x2 - 8*x1 - 8*x2)/2);

[X,Y] = meshgrid(-2:.2:8);
Z = f(X,Y);
% s = surf(X,Y,Z,'FaceAlpha',0.9)
sl = surfl(X,Y,Z)
% shading interp