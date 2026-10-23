function t = sim_tag(d, T)
%SIM_TAG  Name of a design.
if d.constrained, c = 'nn'; else, c = 'uc'; end
if d.boundary, bd = '_bd'; else, bd = ''; end
lr = d.lagrule; if isnumeric(lr), lr = num2str(lr); end
t = sprintf('T%d_K%d_r%d_fb%s_%s%s_lag-%s', T, d.K, d.r, num2str(d.feedback), c, bd, lr);
end
