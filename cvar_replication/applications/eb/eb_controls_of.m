function cc = eb_controls_of(E, s)
%EB_CONTROLS_OF  Never-triggered controls that do not border state s.
cc = E.ctrl(cellfun(@(c) ~ismember(s, E.nb(c)), E.ctrl));
end
