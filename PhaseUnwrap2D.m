% PhaseUnwrap2D.m - Upravené pre paralelné spracovanie datasetu TIFF obrázkov
% Pôvodný autor: Aaron James Lemmer
% Úpravy pre dataset, Tiff objekt, názvy súborov a explicitnú paralelizáciu: Gemini

% --- INICIALIZÁCIA ---
clear all; % Odporúča sa na vyčistenie premenných z predchádzajúcich behov
close all; % Zatvorí všetky otvorené obrázky
clc;       % Vymaže Command Window

disp('------------------------------------------------------------');
disp('Skript na PARALELNÉ rozbalenie fázového datasetu (Goldstein)');
disp('------------------------------------------------------------');

% --- DEFINUJTE CESTY K ADRESÁROM ---
% !!! DÔLEŽITÉ: Upravte tieto cesty podľa vašej štruktúry adresárov !!!
vstupny_adresar = 'C:\Users\viera\Desktop\cchm_ttet\images'; 
vystupny_adresar = 'C:\Users\viera\Desktop\matlab_goldstain\2d-phase-unwrap-goldstein\goldstein_unwrapped2'; 

disp(['Vstupný adresár: ', vstupny_adresar]);
disp(['Výstupný adresár: ', vystupny_adresar]);

% Vytvorenie výstupného adresára, ak neexistuje
if ~exist(vystupny_adresar, 'dir')
   mkdir(vystupny_adresar);
   disp(['Vytvorený výstupný adresár: ', vystupny_adresar]);
end

% Získanie zoznamu všetkých TIFF súborov vo vstupnom adresári
tiff_files_pattern = '*.tiff'; 
subory = dir(fullfile(vstupny_adresar, tiff_files_pattern));

if isempty(subory)
    error(['Vo vstupnom adresári "', vstupny_adresar, '" neboli nájdené žiadne súbory typu "', tiff_files_pattern, '". Skontrolujte cestu a koncovku súborov.']);
end

num_total_subory = length(subory);
disp(['Nájdených súborov na spracovanie: ', num2str(num_total_subory)]);
disp(' ');

% --- ŠTARTOVANIE PARALELNÉHO POOLU S 10 PRACOVNÍKMI ---
% Najprv skúsime zatvoriť akýkoľvek existujúci pool
disp('Pokus o zatvorenie existujúceho paralelného poolu...');
delete(gcp('nocreate')); 

% Spustíme nový pool s 10 pracovníkmi (jadrami)
disp('Pokus o spustenie nového paralelného poolu s 10 pracovníkmi...');
try
    parpool(10); 
    disp('Paralelný pool úspešne spustený s 10 pracovníkmi.');
catch ME_parpool
    disp('CHYBA: Nepodarilo sa spustiť paralelný pool s 10 pracovníkmi.');
    disp(['Chybové hlásenie MATLABu: ', ME_parpool.message]);
    disp('Skontrolujte licenciu pre Parallel Computing Toolbox a dostupné zdroje.');
    disp('Ak problém pretrváva, skúste reštartovať MATLAB alebo použiť menší počet pracovníkov.');
    error('Paralelný pool sa nepodarilo spustiť. Ukončujem skript.');
end

disp('Začínam paralelné spracovanie...');
overall_tic = tic; % Meranie celkového času

% --- HLAVNÝ PARALELNÝ CYKLUS PRE SPRACOVANIE KAŽDÉHO SÚBORU ---
parfor idx_subor = 1:num_total_subory
    
    % Premenné špecifické pre iteráciu
    nazov_suboru_iter = subory(idx_subor).name;
    plna_cesta_vstup_iter = fullfile(vstupny_adresar, nazov_suboru_iter);
    
    worker_info_str = '';
    try
        worker = getCurrentTask();
        if ~isempty(worker)
            worker_info_str = sprintf('Worker %d: ', worker.ID);
        end
    catch
        % getCurrentTask() nemusí byť dostupné vo všetkých kontextoch alebo verziách
        % ponecháme worker_info_str prázdny
    end
    
    fprintf('%s--- Začína spracovanie súboru (%d/%d): %s ---\n', worker_info_str, idx_subor, num_total_subory, nazov_suboru_iter);
    iter_tic = tic; % Začiatok merania času pre túto iteráciu
    
    % NAČÍTANIE AKTUÁLNEHO TIFF OBRÁZKA
    phaseAng_single_iter = []; % Inicializácia pre parfor analyzátor
    try
        phaseAng_single_iter = imread(plna_cesta_vstup_iter);
    catch ME
        fprintf('%sCHYBA: Nepodarilo sa načítať súbor: %s\n', worker_info_str, plna_cesta_vstup_iter);
        fprintf('%sChybové hlásenie: %s\n', worker_info_str, ME.message);
        fprintf('%sPreskakujem tento súbor.\n', worker_info_str);
        continue; 
    end
    
    phaseAng_iter = []; % Inicializácia
    if ~isa(phaseAng_single_iter, 'double')
        phaseAng_iter = double(phaseAng_single_iter);
    else
        phaseAng_iter = phaseAng_single_iter; 
    end

    [num_row_iter, num_col_iter] = size(phaseAng_iter);
    mask_iter = ones(num_row_iter, num_col_iter);
    border_iter = ~mask_iter; 

    % --- GOLDSTEINOV ALGORITMUS NA ROZBALENIE FÁZY ---
    [residues_iter, num_residues_iter] = LocateResidues(phaseAng_iter, border_iter);
    
    branch_cuts_iter = zeros(num_row_iter, num_col_iter); 
    num_dipoles_iter = 0; 
    % Ak chcete použiť spracovanie dipólov, odkomentujte:
    % fprintf('%sKrok 2: Odstraňovanie dipólov (ak je aktívne)...\n', worker_info_str);
    % [residues_iter, branch_cuts_iter, num_dipoles_iter] = Dipoles(num_row_iter, num_col_iter, branch_cuts_iter, residues_iter);
    
    num_residues_for_branchcuts_iter = num_residues_iter - 2*num_dipoles_iter; 
    [branch_cuts_updated_iter] = BranchCuts(branch_cuts_iter, residues_iter, num_residues_for_branchcuts_iter, border_iter);
    branch_cuts_iter = branch_cuts_updated_iter; 
    
    phase_soln_iter = nan(size(branch_cuts_iter)); 
    [~, phase_soln_updated_iter, ~] = UnwrapAroundCuts(phaseAng_iter, phase_soln_iter, branch_cuts_iter, border_iter);
    phase_soln_iter = phase_soln_updated_iter; 

    % --- ULOŽENIE VÝSLEDKU AKO 32-BIT TIFF pomocou Tiff objektu ---
    [~, meno_bez_pripony_original_iter, ~] = fileparts(nazov_suboru_iter);
    prefix_na_odstranenie_iter = 'wrappedbg_'; 
    identifikator_obrazka_iter = ''; 

    if startsWith(meno_bez_pripony_original_iter, prefix_na_odstranenie_iter)
        identifikator_obrazka_iter = extractAfter(meno_bez_pripony_original_iter, prefix_na_odstranenie_iter);
    else
        identifikator_obrazka_iter = meno_bez_pripony_original_iter;
        fprintf('%sVAROVANIE: Súbor "%s" nemá očakávaný prefix "wrappedbg_". Použije sa celý názov bez prípony ako identifikátor.\n', worker_info_str, nazov_suboru_iter);
    end
    vystupny_nazov_suboru_iter = ['unwrapped_', identifikator_obrazka_iter, '.tiff'];
    plna_cesta_vystup_iter = fullfile(vystupny_adresar, vystupny_nazov_suboru_iter);
    
    % fprintf('%sUkladám rozbalený obrázok (pomocou Tiff objektu) do: %s\n', worker_info_str, plna_cesta_vystup_iter);
    
    if isempty(phase_soln_iter) || all(isnan(phase_soln_iter(:)))
        fprintf('%sCHYBA PRED ULOŽENÍM: Matica phase_soln pre súbor %s je prázdna alebo obsahuje iba NaN. Súbor sa neuloží.\n', worker_info_str, nazov_suboru_iter);
    else
        t_iter = []; 
        try
            data_na_ulozenie_iter = single(phase_soln_iter);
            
            t_iter = Tiff(plna_cesta_vystup_iter, 'w');
            
            % Vytvorenie tagstruct lokálne pre každú iteráciu
            tagstruct_iter = struct(); 
            tagstruct_iter.ImageLength = size(data_na_ulozenie_iter, 1);
            tagstruct_iter.ImageWidth = size(data_na_ulozenie_iter, 2);
            tagstruct_iter.Photometric = Tiff.Photometric.MinIsBlack;
            tagstruct_iter.BitsPerSample = 32;
            tagstruct_iter.SamplesPerPixel = 1;
            tagstruct_iter.SampleFormat = Tiff.SampleFormat.IEEEFP; 
            tagstruct_iter.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tagstruct_iter.Compression = Tiff.Compression.None;
            t_iter.setTag(tagstruct_iter);
            
            t_iter.write(data_na_ulozenie_iter);
            t_iter.close();
            
            % fprintf('%sUloženie súboru %s úspešné.\n', worker_info_str, vystupny_nazov_suboru_iter);
            
        catch ME_tiff_write
            fprintf('%sCHYBA PRI UKLADANÍ (pomocou Tiff objektu): Nepodarilo sa uložiť rozbalený obrázok: %s\n', worker_info_str, plna_cesta_vystup_iter);
            fprintf('%sChybové hlásenie MATLABu: %s\n', worker_info_str, ME_tiff_write.message);
            if isa(t_iter, 'Tiff') 
                try
                    t_iter.close(); % Skúsi zavrieť Tiff objekt aj v prípade chyby
                catch
                    % Ak aj zatvorenie zlyhá, už sa nedá veľa robiť
                end
            end
        end
    end
    
    iter_toc_val = toc(iter_tic); % Získanie hodnoty času iterácie
    fprintf('%s--- Dokončené spracovanie súboru %s za %.2f sekúnd ---\n', worker_info_str, nazov_suboru_iter, iter_toc_val);
    
% --- KONIEC HLAVNÉHO PARALELNÉHO CYKLU ---
end

overall_toc_val = toc(overall_tic); % Získanie hodnoty celkového času
disp('----------------------------------------------------');
fprintf('Spracovanie všetkých súborov v datasete dokončené za %.2f sekúnd.\n', overall_toc_val);
disp('----------------------------------------------------');

% --- (VOLITEĽNÉ) ZATVORENIE PARALELNÉHO POOLU ---
disp('Zatváram paralelný pool...');
delete(gcp('nocreate')); 
disp('Paralelný pool zatvorený.');