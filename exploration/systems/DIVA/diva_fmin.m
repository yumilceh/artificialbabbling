function e_sq = diva_fmin(x, vowel)
%    scale =[100,500,1500,3000]';
scale=[1,1,1,1]';
   [Aud,~,~,af] = diva_synth(x);
   Aud = Aud./scale;
   
   if min(af) < 0 
      Aud = 0*Aud;
   end
 
   s = Aud(2:3) ;
   e_sq = norm(s-vowel);
   
end
