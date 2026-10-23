function d = eb_out(cfg)
%EB_OUT  Folder of the EB results.
d = fullfile(cfg.results, 'eb'); if ~exist(d, 'dir'), mkdir(d); end
end
