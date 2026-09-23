%% IS Lab3 - SBF (RBF) tinklas: 1 ieejimas, 2 Gauso neuronai, tiesinis isejimas
% Mokomi tik w0, w1, w2 (perceptrono/LMS taisykle). Centrai ir spinduliai
% parenkami rankiniu budu ir nesikeicia mokymo metu.

clear all; close all; clc;

%% 1. Duomenys
x = 0.1:1/22:1;                                  % 20 tasku, [0.1 ; 1]
y = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;   % norimas atsakas

N = length(x);

%% 2. Rankiniu budu parinkti SBF parametrai (centrai ir spinduliai)
c1 = 0.18;   r1 = 0.25;
c2 = 0.90;   r2 = 0.20;

%% 3. Pradiniai (atsitiktiniai) isejimo neurono svoriai
w0 = 0.1;
w1 = -0.2;
w2 = 0.3;

eta    = 0.1;     % mokymo zingsnis
epochs = 5000;     % epochu skaicius

mse_history = zeros(1, epochs);

%% 4. Mokymo ciklas (LMS / perceptrono atnaujinimo taisykle)
for epoch = 1:epochs
    err_sum = 0;

    for i = 1:N
        xi = x(i);

        % Gauso baziniu funkciju reiksmes siam taskui
        F1 = exp(-(xi - c1)^2 / (2*r1^2));
        F2 = exp(-(xi - c2)^2 / (2*r2^2));

        % Tinklo isejimas (tiesinis neuronas)
        out = w0 + w1*F1 + w2*F2;

        % Klaida
        e = y(i) - out;

        % Svoriu atnaujinimas (kaip perceptrone/LMS: w += eta*e*ieejimas)
        w0 = w0 + eta * e * 1;   % w0 - poslinkis, "ieejimas" = 1
        w1 = w1 + eta * e * F1;
        w2 = w2 + eta * e * F2;

        err_sum = err_sum + e^2;
    end

    mse_history(epoch) = err_sum / N;
end

%% 5. Rezultatu ivertinimas
y_model = zeros(1, N);
for i = 1:N
    xi = x(i);
    F1 = exp(-(xi - c1)^2 / (2*r1^2));
    F2 = exp(-(xi - c2)^2 / (2*r2^2));
    y_model(i) = w0 + w1*F1 + w2*F2;
end

fprintf('Galutiniai svoriai: w0 = %.4f, w1 = %.4f, w2 = %.4f\n', w0, w1, w2);
fprintf('Galutine MSE: %.6f\n', mse_history(end));

%% 6. Grafikai
figure;
subplot(2,1,1);
plot(x, y, 'b-o', 'LineWidth', 1.2); hold on;
plot(x, y_model, 'r-*', 'LineWidth', 1.2);
xlabel('x'); ylabel('y');
legend('Norimas atsakas y(x)', 'SBF tinklo isejimas', 'Location', 'best');
title('SBF tinklo aproksimacija');
grid on;

subplot(2,1,2);
plot(1:epochs, mse_history, 'g-', 'LineWidth', 1.2);
xlabel('Epocha'); ylabel('MSE');
title('Mokymo paklaidos kitimas');
grid on;