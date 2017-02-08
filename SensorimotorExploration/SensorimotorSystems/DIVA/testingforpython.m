art = 0.7 * ones(13,1);
[Aud,Som,Outline,af] = diva_synth(art);


art = [0.1 * ones(13,1), 0.7 * ones(13,1)];
art(11:13,1) = 1;
art(11:13,2) = 1;

art_ = repmat(art', [40,1]);

Aud = diva_synth(art_','sound');

sound(Aud, 11025)