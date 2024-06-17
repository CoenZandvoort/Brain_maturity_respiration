% Statistically compare the correlation coefficient
%
% This script statistically evaluates whether the correlation coefficients
% of brain maturity and post-menstrual age are different for apnoea rate. 
%
% CZ, Apr-2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1) % fix random number generator
par_cor = @(t, N, M)(sign(t) .* sqrt(t .^ 2 ./ (N - rank(M) + t .^ 2)));

data_folder = '~/Documents/Papers/Brain_age_variability_respiration/Codes/';


% load sheet
sheet = importdata('data_overview.xlsx');


% load session-IDs, infant-IDs, respiration support, infection, and PMA.
sheet.textdata(:, 1) = strrep(sheet.textdata(:, 1), 'X', 'x');
ses_labels = strcat(sheet.textdata(2 : end, 1), sheet.textdata(2 : end, 2), sheet.textdata(2 : end, 3));
infant_labels = strcat(sheet.textdata(2 : end, 1), sheet.textdata(2 : end, 2));
infant_labels_unique = unique(infant_labels);

idx_resp_support = find(strcmp('Ventilation', sheet.textdata(1, :)));
resp_support_labels = sheet.textdata(2 : end, idx_resp_support);

idx_infection = find(strcmp('Infection', sheet.textdata(1, :)));
infection_labels = sheet.textdata(2 : end, idx_infection);

pma_labels = (sheet.data(:, 1) + sheet.data(:, 2) / 7);


% load brain age models outputs
Y_predict_sens = load(fullfile(data_folder, 'output/brain_age_sensory/brain_age_sensory.mat'));
Y_predict_rest = load(fullfile(data_folder, 'output/brain_age_rest/brain_age_rest.mat'));

Y_predict = struct;
for s = 1 : numel(ses_labels)
    Y_predict.Y(s) = pma_labels(s);
    Y_predict.ses_labels(s) = ses_labels(s);


    % get the mean of both models
    idx_mdl_sens = find(strcmp(ses_labels{s}, Y_predict_sens.ses_labels));
    idx_mdl_rest = find(strcmp(ses_labels{s}, Y_predict_rest.ses_labels));
    Y_predict.Y_predict(s) = mean([Y_predict_sens.Y_predict(idx_mdl_sens), Y_predict_rest.Y_predict(idx_mdl_rest)], 'omitnan');
end


% remove bias from combined brain age estimates
mdl = fitglm(Y_predict.Y, Y_predict.Y_predict);
mdl_ypred = predict(mdl, Y_predict.Y(:));
bias = mdl_ypred - Y_predict.Y(:);
Y_predict.Y_predict = Y_predict.Y_predict - bias';


% pre-allocate some vectors for the predictors and responses
pred = struct;
response = struct;
sub_counter = 1;
for s = 1 : numel(ses_labels)

    % check if the baby has a brain age prediction for this model
    if sum(strcmp(ses_labels{s}, Y_predict.ses_labels)) == 0
        continue
    end


    % find corresponding respiration outcomes
    try
        ibi_data = load(fullfile(pwd, 'output/ibi_outcomes', sprintf('ibi_stat_%s.mat', ses_labels{s})));
    catch
        warning('no respiration outcomes for %s', ses_labels{s})
        continue
    end
    
    % only get real data (some data of the visual+tactile model
    % may be NaNs, and this will problematic for the linear
    % regression models below).
    idx_sub = find(strcmp(ses_labels{s}, Y_predict.ses_labels));
    if isnan(Y_predict.Y_predict(idx_sub))
        continue
    end
    

    % get predictors and responses
    pred.brain_maturity(sub_counter, 1) = Y_predict.Y_predict(idx_sub) - Y_predict.Y(idx_sub);
    pred.pma(sub_counter, 1) = Y_predict.Y(idx_sub);
    pred.data_length(sub_counter, 1) = ibi_data.data_length_sec;
    pred.infection(sub_counter, 1) = infection_labels(s);
    pred.resp_support(sub_counter, 1) = resp_support_labels(s);
    pred.infant_num(sub_counter, 1) = find(strcmp(infant_labels(s), infant_labels_unique));
    
    response.ibi_rate_15_0_sec(sub_counter, 1) = ibi_data.ibi_rate_15_0_sec;
    
    fprintf('%s: %d\n', ses_labels{s}, sub_counter)
    sub_counter = sub_counter + 1;

end


% get true rho for brain age gap
tbl = table(response.ibi_rate_15_0_sec, pred.brain_maturity, pred.pma, pred.data_length, pred.infection, pred.resp_support, pred.infant_num, ...
    'VariableNames', {'apnoea_rate', 'brain_maturity', 'pma', 'data_length', 'infection', 'resp_support', 'infant'});
equation_label = 'apnoea_rate ~ brain_maturity + data_length + infection + (1 + brain_maturity | infant)';
mdl = fitlme(tbl, equation_label);
rho_true = par_cor(mdl.Coefficients.tStat(2), numel(tbl(:, 2)), pred.brain_maturity);


% get bootstrap distribution
tbl = table(response.ibi_rate_15_0_sec, pred.pma, pred.data_length, pred.infection, pred.resp_support, pred.infant_num, ...
    'VariableNames', {'apnoea_rate', 'age', 'data_length', 'infection', 'resp_support', 'infant'});
rho_bootstrap = bootstrp(10000, @bootstrap_fitlme, tbl, pred.pma);

p_val = sum(rho_true >= rho_bootstrap) ./ numel(rho_bootstrap);


% function to perform bootstrapping
function myout = bootstrap_fitlme(tbl, age)
par_cor = @(t, N, M)(sign(t) .* sqrt(t .^ 2 ./ (N - rank(M) + t .^ 2)));
mdl = fitlme(tbl, 'apnoea_rate ~ age + data_length + infection + (1 + age | infant)');
myout = par_cor(mdl.Coefficients.tStat(2), numel(tbl(:, 2)), age);
end