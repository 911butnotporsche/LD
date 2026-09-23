function pozymiai = pozymiai_atpazinti(pavadinimas, eiluciu_sk)
% pozymiai = pozymiai_atpazinti(pavadinimas, eiluciu_sk)
% Taikymo pavyzdys:
% pozymiai = pozymiai_atpazinti('train_data.png', 8);
%
% PASTABA: nuotraukos fotografuotos telefonu (netolygus apsvietimas,
% seseliai, popieriaus tekstura), todel naudojamas ADAPTYVUS slenkstinimas
% (imbinarize su 'adaptive'), atsparesnis netolygiam apsvietimui nei
% paprastas globalus Otu slenkstis (graythresh + im2bw).

V = imread(pavadinimas);
V_pustonis = rgb2gray(V);

V_dvejetainis = imbinarize(V_pustonis, 'adaptive', ...
    'ForegroundPolarity', 'dark', 'Sensitivity', 0.45);

V_konturais = edge(uint8(V_dvejetainis));
se = strel('square', 7);
V_uzpildyti = imdilate(V_konturais, se);
V_vientisi = imfill(V_uzpildyti, 'holes');

min_plotas = 500;
V_vientisi = bwareaopen(V_vientisi, min_plotas);

[O_suzymeti, Skaicius] = bwlabel(V_vientisi);

if mod(Skaicius, eiluciu_sk) ~= 0
    error(['pozymiai_atpazinti: vaizde "%s" rasta %d objektu, bet %d ' ...
        'eiluciu turetu dalytis be liekanos. Koreguok min_plotas arba ' ...
        'Sensitivity reiksme.'], pavadinimas, Skaicius, eiluciu_sk);
end

O_pozymiai = regionprops(O_suzymeti);
O_ribos = [O_pozymiai.BoundingBox];
O_ribos = reshape(O_ribos, [4 Skaicius]);
O_centras = [O_pozymiai.Centroid];
O_centras = reshape(O_centras, [2 Skaicius]);
O_centras = O_centras';
O_centras(:,3) = 1:Skaicius;

O_centras = sortrows(O_centras, 2);
raidziu_sk = Skaicius / eiluciu_sk;

for k = 1:eiluciu_sk
    O_centras((k-1)*raidziu_sk+1:k*raidziu_sk, :) = ...
        sortrows(O_centras((k-1)*raidziu_sk+1:k*raidziu_sk, :), 3);
end

for k = 1:Skaicius
    objektai{k} = imcrop(V_dvejetainis, O_ribos(:, O_centras(k,3)));
end

for k = 1:Skaicius
    V_fragmentas = objektai{k};
    [aukstis, plotis] = size(V_fragmentas);
    stulpeliu_sumos = sum(V_fragmentas, 1);
    V_fragmentas(:, stulpeliu_sumos == aukstis) = [];
    [aukstis, plotis] = size(V_fragmentas);
    eiluciu_sumos = sum(V_fragmentas, 2);
    V_fragmentas(eiluciu_sumos == plotis, :) = [];
    objektai{k} = V_fragmentas;
end

for k = 1:Skaicius
    V_fragmentas = objektai{k};
    V_fragmentas_7050 = imresize(V_fragmentas, [70, 50]);
    Vid_sviesumas = zeros(1, 35);
    for m = 1:7
        for n = 1:5
            Vid_sviesumas_eilutese = ...
                sum(V_fragmentas_7050((m*10-9:m*10), (n*10-9:n*10)));
            Vid_sviesumas((m-1)*5+n) = sum(Vid_sviesumas_eilutese);
        end
    end
    Vid_sviesumas = ((100 - Vid_sviesumas) / 100);
    Vid_sviesumas = Vid_sviesumas(:);
    pozymiai{k} = Vid_sviesumas;
end
end
