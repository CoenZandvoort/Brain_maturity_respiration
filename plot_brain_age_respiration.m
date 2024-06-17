% Plot respiration and brain age outcomes
%
% This script visualises and statistically evaluates the respiration-EEG
% associations. At this point it is assumed that brain age is computed from
% the resting state and sensory models, and apnoea/respiration rate from
% the inter-breath interval data.
%
% CZ, Jun-2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par_cor = @(t, N, M)(sign(t) .* sqrt(t .^ 2 ./ (N - rank(M) + t .^ 2)));

data_folder = '~/Documents/Papers/Brain_age_variability_respiration/Codes/';
pred_labels = {'brain_maturity', 'pma'};
ibi_labels = {'ibi_resp_rate', 'ibi_rate_15_0_sec'};


% load sheet
sheet = importdata('data_overview.xlsx');
study_label = 'PMA_31_to_37_weeks';


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
        ibi_data = load(fullfile(data_folder, 'output/ibi_outcomes', sprintf('ibi_stat_%s.mat', ses_labels{s})));
    catch
        continue
    end


    % only get real data (some data of the visual+tactile model
    % may be NaNs, and this will problematic for the linear
    % regression models below).
    idx_sub = find(strcmp(ses_labels{s}, Y_predict.ses_labels));
    
    
    % get predictors and responses
    pred.brain_maturity(sub_counter, 1) = Y_predict.Y_predict(idx_sub) - Y_predict.Y(idx_sub);
    pred.pma(sub_counter, 1) = Y_predict.Y(idx_sub);
    pred.data_length(sub_counter, 1) = ibi_data.data_length_sec;
    pred.infection(sub_counter, 1) = infection_labels(s);
    pred.resp_support(sub_counter, 1) = resp_support_labels(s);
    pred.infant_num(sub_counter, 1) = find(strcmp(infant_labels(s), infant_labels_unique));

    for res = 1 : numel(ibi_labels)
        response.(ibi_labels{res})(sub_counter, 1) = ibi_data.(ibi_labels{res});
    end

    fprintf('%s: %d\n', ses_labels{s}, sub_counter)
    sub_counter = sub_counter + 1;

end


% make the statistical models and plot the data (predictor vs response)
for res = 1 : numel(ibi_labels)

    for p = 1 : numel(pred_labels)

        % linear model with all confounding factors
        if strcmp(ibi_labels{res}, 'ibi_resp_rate')
            stat_model_label = ['IBI_outcome ~ ', pred_labels{p}, ' + infection'];
        elseif strcmp(ibi_labels{res}, 'ibi_rate_15_0_sec')
            stat_model_label = ['IBI_outcome ~ ', pred_labels{p}, ' + infection + data_length'];
        end
        tbl = table(response.(ibi_labels{res}), pred.(pred_labels{p}), pred.data_length, categorical(pred.infection), categorical(pred.resp_support), ...
            'VariableNames', {'IBI_outcome', pred_labels{p}, 'data_length', 'infection', 'resp_support'});
        
        if strcmp(pred_labels{p}, 'brain_age')
            stat_model_label = [stat_model_label, ' + pma'];
            tbl = [tbl, table(pred.pma, 'VariableNames', {'pma'})];
        end

        lm = fitlm(tbl, stat_model_label);


        % get p-values for model (here, we make a full model including 
        % predictors (fixed+random incl. confounding factors) and response
        stat_model_label = [stat_model_label, ' + (1 + ', pred_labels{p},' | infant)'];
        tbl = [tbl, table(pred.infant_num, 'VariableNames', {'infant'})];
        lme_p_values = fitlme(tbl, stat_model_label);

        pr = par_cor(lme_p_values.Coefficients.tStat(2), size(tbl, 1), pred.(pred_labels{p}));

        
        % create the linear regression where we average out the effects of
        % the confounding factors
        figure;
        adj = plotAdjustedResponse(lm, pred_labels{p});
        pred.(strcat(pred_labels{p}, '_adjusted')) = adj(1).XData;
        response.(strcat(ibi_labels{res}, '_adjusted')) = adj(1).YData;
        close


        % fit a linear mixed effects model to the data
        stat_model_label = ['IBI_outcome ~ 1 + ', pred_labels{p}, ' + (1 + ', pred_labels{p},' | infant)'];
        tbl = table(response.(strcat(ibi_labels{res}, '_adjusted'))(:), pred.(strcat(pred_labels{p}, '_adjusted'))(:), pred.infant_num, ...
            'VariableNames', {'IBI_outcome', pred_labels{p}, 'infant'});
        lme = fitlme(tbl, stat_model_label);

        
        % plot regression model (mean) with superimposed data
        col = [0.0, 0.5, 1.0; 1.0, 0.25, 0.0; 0.2, 0.7, 0.3];
        figure(p * 10 + res)
        po = get(gcf, 'position');
        set(gcf, 'position', [po(1:2), 400, 400], 'name', sprintf('resp_%s_%s_%s', ibi_labels{res}, pred_labels{p}, study_label))

        scatter(pred.(strcat(pred_labels{p}, '_adjusted')), response.(strcat(ibi_labels{res}, '_adjusted')), 'filled', 'MarkerFaceAlpha', 0.5, ...
            'MarkerFaceColor', col(p, :), 'MarkerEdgeColor', 'k', 'sizedata', 192) % data
        hold on
        ci_mean = glmval(lme.Coefficients.Estimate, pred.(strcat(pred_labels{p}, '_adjusted')), 'identity', lme.Coefficients);
        plot(pred.(strcat(pred_labels{p}, '_adjusted')), ci_mean, 'k', 'linewidth', 1.5)


        title(sprintf('\\rho:%.04f; p:%.04f; Beta:%.02f', pr, lme_p_values.Coefficients.pValue(2), lme_p_values.Coefficients.Estimate(2)))
        xlabel(strcat(pred_labels{p}, ' [weeks]'), 'Interpreter', 'none');
        ylabel(ibi_labels{res}, 'Interpreter', 'none');
        g = gca;
        g.XLim = round(g.XLim + [-1, 1]);
        g.YLim = round(g.YLim * 10) / 10;

    end

end

set(findall(0, 'type', 'axes'), 'FontName', 'Times', 'Fontsize', 20, 'TickDir', 'out', 'box', 'off', 'linewidth', 2, 'ticklength', [0.015, 0.015])
