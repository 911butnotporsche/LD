%% ========================================================================
%  IS-Lab4: Raidziu atpazinimas su RBFN
%  Abecele: L M N O P R S T U V Z (11 raidziu, 8 mokymo eilutes)
%  SVARBU: patikrink komandineje eilute su "dir *.png", kad failu vardai
%  zemiau TIKSLIAI atitiktu tavo turimus failus (didziosios/mazosios
%  raides, papildomi skaiciai pavadinime ir t.t.)
% =========================================================================
clear; close all; clc;

%% 1) Raidziu pavyzdziu nuskaitymas ir pozymiu skaiciavimas (mokymui)
pavadinimas = 'train_data.png';
eiluciu_sk_mokymui = 8;
raidziu_sk_eiluteje = 11;    % L M N O P R S T U V Z

pozymiai_tinklo_mokymui = pozymiai_atpazinti(pavadinimas, eiluciu_sk_mokymui);
P = cell2mat(pozymiai_tinklo_mokymui);
T = repmat(eye(raidziu_sk_eiluteje), 1, eiluciu_sk_mokymui);

%% 2) RBFN mokymas su ORIGINALIU (13) ir SUMAZINTU neuronu skaiciumi
goal = 0;
spread = 1;
MN_originalus = 13;
MN_sumazintas = 6;

tinklas_orig   = newrb(P, T, goal, spread, MN_originalus, 1);
tinklas_mazas  = newrb(P, T, goal, spread, MN_sumazintas, 1);

%% 3) Pagalbine funkcija tinklo atsakymui i raides paversti
raides = {'L','M','N','O','P','R','S','T','U','V','Z'};

    function atsakymas = klasifikuoti_ir_paversti_i_raides(tinklas, P_test, raides)
        Y = sim(tinklas, P_test);
        [~, b] = max(Y);
        atsakymas = '';
        for k = 1:length(b)
            atsakymas = [atsakymas, raides{b(k)}]; %#ok<AGROW>
        end
    end

%% 4) Tinklo patikra su mokymo metu naudotomis raidemis (2-a eilute)
P2 = P(:, raidziu_sk_eiluteje+1 : 2*raidziu_sk_eiluteje);
atsakymas_orig_mok  = klasifikuoti_ir_paversti_i_raides(tinklas_orig,  P2, raides);
atsakymas_mazas_mok = klasifikuoti_ir_paversti_i_raides(tinklas_mazas, P2, raides);

fprintf('Mokymo pavyzdziu (2-a eilute) atpazinimas:\n');
fprintf('  RBFN %d neuronu:  %s\n', MN_originalus,  atsakymas_orig_mok);
fprintf('  RBFN %d neuronu:  %s\n', MN_sumazintas,  atsakymas_mazas_mok);

%% 5) Pirmojo testinio zodzio (MUST - 4 raides) atpazinimas
pavadinimas = 'Test_ntr_1.png';
pozymiai_patikrai_1 = pozymiai_atpazinti(pavadinimas, 1);
P_test1 = cell2mat(pozymiai_patikrai_1);

atsakymas_orig_1  = klasifikuoti_ir_paversti_i_raides(tinklas_orig,  P_test1, raides);
atsakymas_mazas_1 = klasifikuoti_ir_paversti_i_raides(tinklas_mazas, P_test1, raides);

fprintf('\nTest_ntr_1.png (4 raidziu zodis) atpazinimas:\n');
fprintf('  RBFN %d neuronu:  %s\n', MN_originalus,  atsakymas_orig_1);
fprintf('  RBFN %d neuronu:  %s\n', MN_sumazintas,  atsakymas_mazas_1);

%% 6) Antrojo testinio zodzio (OUTPOST - 7 raides) atpazinimas
pavadinimas = 'Test_ntr_2.png';
pozymiai_patikrai_2 = pozymiai_atpazinti(pavadinimas, 1);
P_test2 = cell2mat(pozymiai_patikrai_2);

atsakymas_orig_2  = klasifikuoti_ir_paversti_i_raides(tinklas_orig,  P_test2, raides);
atsakymas_mazas_2 = klasifikuoti_ir_paversti_i_raides(tinklas_mazas, P_test2, raides);

fprintf('\nTest_ntr_2.png (7 raidziu zodis) atpazinimas:\n');
fprintf('  RBFN %d neuronu:  %s\n', MN_originalus,  atsakymas_orig_2);
fprintf('  RBFN %d neuronu:  %s\n', MN_sumazintas,  atsakymas_mazas_2);

%% 7) Rezultatu atvaizdavimas
figure(7);
subplot(2,1,1);
text(0.1, 0.7, ['Zodis 1 (', num2str(MN_originalus), ' n.): ', atsakymas_orig_1], 'FontSize', 20);
text(0.1, 0.3, ['Zodis 1 (', num2str(MN_sumazintas), ' n.): ', atsakymas_mazas_1], 'FontSize', 20);
axis off;
subplot(2,1,2);
text(0.1, 0.7, ['Zodis 2 (', num2str(MN_originalus), ' n.): ', atsakymas_orig_2], 'FontSize', 20);
text(0.1, 0.3, ['Zodis 2 (', num2str(MN_sumazintas), ' n.): ', atsakymas_mazas_2], 'FontSize', 20);
axis off;
