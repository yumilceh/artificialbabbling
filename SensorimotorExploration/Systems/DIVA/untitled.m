function [x,fval,exitflag,output,lambda,grad,hessian] = untitled(MaxIter_Data,TolFun_Data,TolX_Data,TolCon_Data)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = optimoptions('fmincon');
%% Modify options setting
options = optimoptions(options,'Display', 'off');
options = optimoptions(options,'MaxIter', MaxIter_Data);
options = optimoptions(options,'TolFun', TolFun_Data);
options = optimoptions(options,'TolX', TolX_Data);
options = optimoptions(options,'TolCon', TolCon_Data);
[x,fval,exitflag,output,lambda,grad,hessian] = ...
fmincon([],[],[],[],[],[],[],[],[],options);
