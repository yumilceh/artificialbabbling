art = 0.7 * ones(13,1);
[Aud,Som,Outline,af] = diva_synth(art);

art(11:13) = 0.8;

art_ = repmat(art', [800,1]);

Aud = diva_synth(art_','sound');

sound(Aud, 11025)