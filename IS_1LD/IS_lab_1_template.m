clear all
clc
%Classification using perceptron

%Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

%Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

%Calculate for each image, colour and roundness
%For Apples
%1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
%2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
%3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
%4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
%5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
%6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
%7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
%8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
%9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[x1;x2];

%Desired output vector
T=[1;1;1;-1;-1];

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% calculate wieghted sum with randomly generated parameters
%v1 = <...>; % write your code here
v1 = x1(1)*w1 + x2(1)*w2 + b
% calculate current output of the perceptron 
if v1 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e1 = T(1) - y;

% repeat the same for the rest 4 inputs x1 and x2
% calculate wieghted sum with randomly generated parameters
% v2 = <...> ; % write your code here
v2 = x1(2)*w1 + x2(2)*w2 + b
% calculate current output of the perceptron 
if v2 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e2 = T(2) - y;


%________3__________
v3 = x1(3)*w1 + x2(3)*w2 + b
if v3 > 0
    y = 1;
else
    y = -1;
end
% calculate the error
e3 = T(3) - y;

%_________4_________
v4 = x1(4)*w1 + x2(4)*w2 + b
if v4 > 0
    y = 1;
else
    y = -1;
end
% calculate the error
e4 = T(4) - y;

%________5__________
v5 = x1(5)*w1 + x2(5)*w2 + b
if v5 > 0
    y = 1;
else
    y = -1;
end
% calculate the error
e5 = T(5) - y;


% <...> write the code for another 3 inputs

% calculate the total error for these 5 inputs 
e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5)

eta = 0.1; % mokymosi greitis

% write training algorithm
while e ~= 0 % executes while the total error is not 0

    % --- svoriu atnaujinimas kiekvienam is 5 pvz
    for n = 1:5
        v = x1(n)*w1 + x2(n)*w2 + b;
        if v > 0
            y = 1;
        else
            y = -1;
        end
        e_i(n) = T(n) - y;

        w1 = w1 + eta*e_i(n)*x1(n);
        w2 = w2 + eta*e_i(n)*x2(n);
        b  = b  + eta*e_i(n);
    end

    % --- po atnaujinimo iš naujo skaičiuojame v, y, e su NAUJAIS w1,w2,b 
    for n = 1:5
        v = x1(n)*w1 + x2(n)*w2 + b;
        if v > 0
            y = 1;
        else
            y = -1;
        end
        e_i(n) = T(n) - y;
    end

    % calculate the total error for these 5 inputs
    e = sum(abs(e_i));
end

fprintf('Apmokyta.\n');
fprintf('w1 = %.4f, w2 = %.4f, b = %.4f\n', w1, w2, b);
fprintf('Galutine paklaida e = %d\n', e);

%% Testavimas su nematytais paveikslėliais
test_images = {A4, A5, A6, A7, A8, A9, P3, P4}; % 8 test vaizdai, mokyme nedalyvavę
test_names  = {'A4','A5','A6','A7','A8','A9','P3','P4'};
test_true   = [1, 1, 1, 1, 1, 1, -1, -1]; % tikra klasė: obuolys=1, kriauše=-1
correct_count = 0; % teisingų atsakymų skaitiklis

for i = 1:length(test_images)
    img = test_images{i};
    tx1 = spalva_color(img);       % test vaizdo spalva
    tx2 = apvalumas_roundness(img); % test vaizdo apvalumas
    tv = tx1*w1 + tx2*w2 + b;      % svertinė suma su apmokytais w1,w2,b
    ty = (tv > 0)*2 - 1;           % ty=1 jei tv>0, kitaip ty=-1

    if ty == 1, klase = 'obuolys'; else, klase = 'kriauše'; end
    if ty == test_true(i)
        rez = 'TEISINGAI'; correct_count = correct_count + 1;
    else
        rez = 'KLAIDINGAI';
    end

    fprintf('%s: %d (%s) - %s\n', test_names{i}, ty, klase, rez); % rezultato eilutė
end
