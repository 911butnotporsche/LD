%% IS-Lab2: Daugiasluoksnis perceptronas (2 sluoksniai), mokymas Backpropagation metodu
% Nenaudojami newff / train / sim. Visi svoriai/bias'ai - atskiri skaliariniai
% kintamieji (be masyvu), kaip reikalauja uzduotis.
%
% Architektura:
%   1 ivestis -> 6 pasleptojo sluoksnio neuronai (sigmoide) -> 1 tiesinis isejimo neuronas
%
% Autorius: Kamil (Lab2)

clear; clc; close all;

%% 1. Mokymo duomenys (20 tasku), ivestis intervale [0,1]
x = 0.1 : 1/22 : 1;              % 20 tasku
N = length(x);

% Tiksline funkcija, kuria reikia aproksimuoti (duota kreive)
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;

% figure(1);
% plot(x, d, 'k-o'); hold on;
% xlabel('x'); ylabel('d(x)');
% title('Tiksline kreive, kuria reikia aproksimuoti');
% grid on;

%% 2. Tinklo inicializavimas (6 pasleptojo sluoksnio neuronai, visi atskiri skaliarai)
rng(1);  % rezultatu pakartojamumui

% --- Pasleptojo sluoksnio svoriai ir bias'ai (1..6 neuronas) ---
w11 = randn(1); b11 = randn(1);
w12 = randn(1); b12 = randn(1);
w13 = randn(1); b13 = randn(1);
w14 = randn(1); b14 = randn(1);
w15 = randn(1); b15 = randn(1);
w16 = randn(1); b16 = randn(1);

% --- Isejimo sluoksnio svoriai ir bias (tiesinis neuronas) ---
w21 = randn(1);
w22 = randn(1);
w23 = randn(1);
w24 = randn(1);
w25 = randn(1);
w26 = randn(1);
b21 = randn(1);

%% 3. Mokymo parametrai
eta      = 0.1;      % mokymosi zingsnis
epochs   = 5000;      % epochu skaicius
mse_hist = zeros(1, epochs);

y  = zeros(1, N);     % tinklo isejimo saugojimas
e  = zeros(1, N);     % paklaidos saugojimas

%% 4. Mokymo ciklas (Backpropagation, atnaujinimas po kiekvieno pavyzdzio)
for epoch = 1:epochs
    kvadr_paklaidu_suma = 0;

    for index = 1:N

        %% ---- TIESIOGINIS SKLIDIMAS (FORWARD PASS) ----
        % Pasleptasis sluoksnis (sigmoidine aktyvavimo funkcija)
        y1 = 1 / (1 + exp(-(x(index)*w11 + b11)));
        y2 = 1 / (1 + exp(-(x(index)*w12 + b12)));
        y3 = 1 / (1 + exp(-(x(index)*w13 + b13)));
        y4 = 1 / (1 + exp(-(x(index)*w14 + b14)));
        y5 = 1 / (1 + exp(-(x(index)*w15 + b15)));
        y6 = 1 / (1 + exp(-(x(index)*w16 + b16)));

        % Isejimo sluoksnis (tiesine aktyvavimo funkcija -> tiesiog svertine suma)
        y(index) = y1*w21 + y2*w22 + y3*w23 + y4*w24 + y5*w25 + y6*w26 + b21;

        %% ---- PAKLAIDOS SKAICIAVIMAS ----
        e(index) = d(index) - y(index);
        kvadr_paklaidu_suma = kvadr_paklaidu_suma + e(index)^2;

        %% ---- BACKPROPAGATION: ISEJIMO SLUOKSNIS ----
        % Isejimo neuronas tiesinis => aktyvavimo funkcijos isvestine = 1
        w21 = w21 + eta * e(index) * y1;
        w22 = w22 + eta * e(index) * y2;
        w23 = w23 + eta * e(index) * y3;
        w24 = w24 + eta * e(index) * y4;
        w25 = w25 + eta * e(index) * y5;
        w26 = w26 + eta * e(index) * y6;
        b21 = b21 + eta * e(index) * 1;

        %% ---- BACKPROPAGATION: PASLEPTASIS SLUOKSNIS ----
        % delta_j = phi'(net_j) * e * w2j   cia phi'(net_j) = yj*(1-yj)
        delta1 = y1*(1-y1) * e(index) * w21;
        delta2 = y2*(1-y2) * e(index) * w22;
        delta3 = y3*(1-y3) * e(index) * w23;
        delta4 = y4*(1-y4) * e(index) * w24;
        delta5 = y5*(1-y5) * e(index) * w25;
        delta6 = y6*(1-y6) * e(index) * w26;

        w11 = w11 + eta * delta1 * x(index);
        w12 = w12 + eta * delta2 * x(index);
        w13 = w13 + eta * delta3 * x(index);
        w14 = w14 + eta * delta4 * x(index);
        w15 = w15 + eta * delta5 * x(index);
        w16 = w16 + eta * delta6 * x(index);

        b11 = b11 + eta * delta1;
        b12 = b12 + eta * delta2;
        b13 = b13 + eta * delta3;
        b14 = b14 + eta * delta4;
        b15 = b15 + eta * delta5;
        b16 = b16 + eta * delta6;

    end

    mse_hist(epoch) = kvadr_paklaidu_suma / N;
end

%% 5. Rezultatai

% Perskaiciuojamas galutinis tinklo isejimas po apmokymo
for index = 1:N
    y1 = 1 / (1 + exp(-(x(index)*w11 + b11)));
    y2 = 1 / (1 + exp(-(x(index)*w12 + b12)));
    y3 = 1 / (1 + exp(-(x(index)*w13 + b13)));
    y4 = 1 / (1 + exp(-(x(index)*w14 + b14)));
    y5 = 1 / (1 + exp(-(x(index)*w15 + b15)));
    y6 = 1 / (1 + exp(-(x(index)*w16 + b16)));
    y(index) = y1*w21 + y2*w22 + y3*w23 + y4*w24 + y5*w25 + y6*w26 + b21;
end

figure(2);
plot(x, d, 'k-o', 'LineWidth', 1.5); hold on;
plot(x, y, 'r-x', 'LineWidth', 1.5);
legend('Tiksline reiksme d(x)', 'Tinklo isejimas y(x)');
xlabel('x'); ylabel('reiksme');
title('Tiksline reiksme ir MLP aproksimacija po apmokymo');
grid on;

% figure(3);
% plot(1:epochs, mse_hist, 'b-');
% xlabel('Epocha'); ylabel('MSE');
% title('Mokymo paklaida (MSE) per epochas');
% grid on;

fprintf('Galutine MSE = %.6f\n', mse_hist(end));