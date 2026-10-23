function E = eb_load(cfg)
%EB_LOAD  The extended benefits panel (applications/data/eb/eb_euc_panel.csv): unemployment rate,
%   weeks of EB and of EUC (in tens of weeks) by state and month, the never-triggered controls with
%   their neighbours, and the states with at least twelve months of EB (the treated states).
P = readtable(fullfile(cfg.data, 'eb', 'eb_euc_panel.csv'), 'TextType', 'string');
P.date = dateshift(datetime(P.date), 'start', 'month');
dates = unique(P.date);
st = unique(P.st)';
T = numel(dates); S = numel(st);
[~, it] = ismember(P.date, dates); [~, is] = ismember(P.st, st);
mk = @(v) accumarray([it, is], v, [T, S], @(x) x(end), NaN);
E.dates = dates; E.states = cellstr(st);
E.UR = mk(P.ur); E.EB = mk(P.eb_weeks) / 10; E.EUC = mk(P.euc_weeks) / 10;
E.ctrl = {'SD', 'ND', 'IA', 'NE', 'WY', 'UT', 'OK'};          % never triggered EB, 2002-2019
E.nb = containers.Map(E.ctrl, {{'ND', 'MN', 'IA', 'NE', 'WY', 'MT'}, {'MN', 'SD', 'MT'}, ...
    {'MN', 'WI', 'IL', 'MO', 'NE', 'SD'}, {'SD', 'IA', 'MO', 'KS', 'CO', 'WY'}, ...
    {'MT', 'SD', 'NE', 'CO', 'UT', 'ID'}, {'ID', 'WY', 'CO', 'AZ', 'NV'}, {'KS', 'MO', 'AR', 'TX', 'NM', 'CO'}});
months = sum(E.EB > 0, 1);
E.elig = E.states(months >= 12 & ~ismember(E.states, E.ctrl));
end
