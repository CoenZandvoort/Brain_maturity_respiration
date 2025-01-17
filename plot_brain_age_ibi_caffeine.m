% Plot brain age against post-menstrual age at caffeine discontinuation
%
% This script visualises and statistically evaluates brain age and
% post-menstrual age when caffeine is discontinued. At this point it is 
% assumed that brain age is computed from the resting state and sensory 
% models.
%
% Data on caffeine continuation can be found in data_overview.xlsx.
%
% CZ, Jun-2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_folder = '~/Documents/Papers/Brain_age_variability_respiration/Codes/';


% load spreadsheet data
sheet = importdata('data_overview.xlsx');


% get subj-IDs and caffeine details
sheet.textdata(:, 1) = strrep(sheet.textdata(:, 1), 'X', 'x');
subj_labels = strcat(sheet.textdata(2 : end, 1), sheet.textdata(2 : end, 2), sheet.textdata(2 : end, 3));
infant_labels = strcat(sheet.textdata(2 : end, 1), sheet.textdata(2 : end, 2));
infection_labels = sheet.textdata(2 : end, strcmp('Infection', sheet.textdata(1, :)));

pma_sheet = (sheet.data(:, 1) + sheet.data(:, 2) / 7);
caf_on = sheet.data(:, 5) + sheet.data(:, 6) / 7;
caf_off = sheet.data(:, 7) + sheet.data(:, 8) / 7;
caf_on_off = sheet.data(:, 9);
unsure_caf_stop = sheet.textdata(2 : end, 13);


% load brain age models outputs
Y_predict_sens = load(fullfile(data_folder, 'output/brain_age_sensory/brain_age_sensory.mat'));
Y_predict_rest = load(fullfile(data_folder, 'output/brain_age_rest/brain_age_rest.mat'));


% load brain age models
data = struct;
infant_labels_unique = unique(infant_labels);
ses_counter = 1;
for s = 1 : numel(subj_labels)
    
    % caffeine information
    data.caf_on(ses_counter) = caf_on(s);
    data.caf_off(ses_counter) = caf_off(s);
    data.caf_on_off(ses_counter) = caf_on_off(s);
    data.unsure(ses_counter) = unsure_caf_stop(s);
    

    % store (brain) ages
    data.pma(ses_counter) = pma_sheet(s);
    data.ses_labels(ses_counter) = subj_labels(s);

    idx_mdl_sens = find(strcmp(subj_labels{s}, Y_predict_sens.ses_labels));
    idx_mdl_rest = find(strcmp(subj_labels{s}, Y_predict_rest.ses_labels));
    data.brain_age(ses_counter) = mean([Y_predict_sens.Y_predict(idx_mdl_sens), Y_predict_rest.Y_predict(idx_mdl_rest)], 'omitnan');


    % get infant number and infection details
    data.infant_num(ses_counter) = find(strcmp(subj_labels{s}(1 : end - 2), infant_labels_unique));
    data.infection{ses_counter} = infection_labels{s};
    ses_counter = ses_counter + 1;

end


% remove bias from combined brain age estimates
mdl = fitglm(data.pma, data.brain_age);
mdl_ypred = predict(mdl, data.pma(:));
bias = mdl_ypred - data.pma(:);
data.brain_age = data.brain_age - bias';


% find the session that closest to the PMA of caffeine discontinuation
data.last_with_caf = zeros(size(data.pma));
for i = 1 : numel(infant_labels_unique)
    idx_infant = find(contains(data.ses_labels, infant_labels_unique{i}));
    idx_last_with_caf = find(data.pma(idx_infant) - data.caf_off(idx_infant) <= 0, 1, 'last');
    data.last_with_caf(idx_infant(idx_last_with_caf)) = 1;
end


% check if any data entries should be removed because we are unsure about
% the discontinuation date.
idx_include = find(data.last_with_caf == 1 & (data.caf_off - data.pma) < 2);
idx_include = idx_include(~strcmp(data.unsure(idx_include), 'medical notes unavailable'));


% remove bias from combined brain age estimates
brain_maturity = data.brain_age(idx_include)' - data.pma(idx_include)';
infection = data.infection(idx_include);
infant_num = data.infant_num(idx_include);


% linear models
stat_model_label = 'caf_stop ~ brain_maturity + Infection';
tbl = table(data.caf_off(idx_include)', brain_maturity, categorical(infection)', ...
    'VariableNames', {'caf_stop', 'brain_maturity', 'Infection'});

lm = fitlm(tbl, stat_model_label);

figure;
adj = plotAdjustedResponse(lm, 'brain_maturity');
brain_maturity_adj = adj(1).XData;
caf_off_adj = adj(1).YData;
close


% regression of adjusted responses
stat_model_label = 'caf_stop ~ brain_maturity';
tbl = table(caf_off_adj', brain_maturity_adj', ...
    'VariableNames', {'caf_stop', 'brain_maturity'});
lme = fitlme(tbl, stat_model_label);


% brain age gap versus time of caffeine discontinuation
figure; 
po = get(gcf, 'Position');
set(gcf, 'Position', [po(1 : 2), 400, 400], 'name', 'brain_maturity_vs_caf_stop')

scatter(brain_maturity_adj, caf_off_adj, 'Filled', 'SizeData', 64, 'MarkerFaceColor', [0.9, 0.1, 0.1])
hold on
ci_mean = glmval(lme.Coefficients.Estimate, brain_maturity_adj', 'identity', lme.Coefficients);
plot(brain_maturity_adj', ci_mean, 'k', 'linewidth', 2)

title(sprintf('%.04f - %.06f', lm.Coefficients.Estimate(2), lm.Coefficients.pValue(2)))
xlabel('Brain maturity [weeks]')
ylabel('PMA at caffeine stop [weeks]')

set(findall(0, 'type', 'axes'), 'box', 'off', 'linewidth', 2, 'Fontsize', 20, 'Fontname', 'Times', 'TickDir', 'out', 'ticklength', [0.015, 0.015])
