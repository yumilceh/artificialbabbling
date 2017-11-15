%Created on: 15th February 2016
%by: Juan Manuel Aceveo Valle
%
%art_states and outputScale exit

samples=size(art_states,1);
auditory_states=zeros(4,samples);
minaf=zeros(1,samples);
for k=1:samples
    [auditory_states(:,k), som, ~, af]=diva_synth(art_states(k,1:13)');
    auditory_states(:,k)=auditory_states(:,k)./outputScale;
    minaf(k)=min(af);
    clear af;
end
auditory_states=auditory_states';
minaf=minaf';
%save test1.mat samples auditory_states minaf art_states outputScale

