function id = ak_ids(st, kind, code, K)
%AK_IDS  BLS series identifiers: LAUS (kind 'laus') and CES (kind 'ces') for state st.
switch kind
    case 'laus', id = sprintf('LASST%s00000000000%s', K.fips(st), code);
    case 'ces',  id = sprintf('SMS%s00000%s01', K.fips(st), code);
end
end
