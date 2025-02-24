(* ::Package:: *)

(* ::Input:: *)
(*hUAV=120;*)


(* ::Input:: *)
(*x0x = -162; (*CHANGE THIS WHEN NEEDED*)*)


(* ::Input:: *)
(*x0y=362;     (*CHANGE THIS WHEN NEEDED*)*)


(* ::Input:: *)
(*x0 = Sqrt[x0x^2+x0y^2];*)


(* ::Input:: *)
(*u0x = 30;    (*CHANGE THIS WHEN NEEDED*)*)


(* ::Input:: *)
(*u0y=282;    (*CHANGE THIS WHEN NEEDED*)*)


(* ::Input:: *)
(*u0 = Sqrt[u0x^2+u0y^2];*)


(* ::Input:: *)
(*q = EuclideanDistance[{x0x,x0y,0},{u0x,u0y,hUAV}]*)


(* ::Input:: *)
(*nDev=50;*)


(* ::Input:: *)
(*nUAV = 10;*)


(* ::Input:: *)
(*R=500(*Radius of a cluster*);*)


(* ::Input:: *)
(*alphaL=2.0(*path loss exponent*);*)


(* ::Input:: *)
(*alphaN=3.5(*path loss exponent*);*)


(* ::Input:: *)
(*Pd = 0.75(*power of a device*);*)


(* ::Input:: *)
(*Pu = 1.5(*power of a UAV*);*)


(* ::Input:: *)
(*mL = 4; (*Nakagami channel parameter, mL = 1 for Rayleigh channel*)*)


(* ::Input:: *)
(*mN = 1;*)


(* ::Input:: *)
(*etaL=mL*(mL!)^(-1/mL);*)


(* ::Input:: *)
(*etaN=mN*(mN!)^(-1/mN);*)


(* ::Input:: *)
(*n0Squared =4.14*10^(-6);*)


(* ::Input:: *)
(*a=9.61;*)


(* ::Input:: *)
(*b=0.16;*)


(* ::Input:: *)
(*RBsUAV = 5;*)


(* ::Input:: *)
(*RBsDevs = 15;*)


(* ::Input:: *)
(*activeProbUAV =nUAV/(RBsUAV (nUAV))*)


(* ::Input:: *)
(*activeProbDevs=nDev/(RBsDevs( nDev));*)


(* ::Chapter:: *)
(*UAV 1 cell,  5 devs*)


(* ::Input:: *)
(*(*u0x=-28.8580891749403; *)*)


(* ::Input:: *)
(*(*u0y=-217.583212004082;*)*)


(* ::Input:: *)
(*(*x0x = {45.0644,-45.8937,-194.7375,-39.1584,-9.8550};*)*)


(* ::Input:: *)
(*(*x0y = {-298.7019,-294.0053,-382.6607,-117.3856,-212.8473};*)*)


(* ::Input:: *)
(*(**)*)


(* ::Subsection:: *)
(*UAV 2 cell,  6 devs*)


(* ::Input:: *)
(*(*u0x=-336.623047567017; *)*)


(* ::Input:: *)
(*(*u0y=-184.916650881321;*)*)


(* ::Input:: *)
(*(*x0x = {-373.0877,-258.9392,-315.2731,-237.5997,-220.1079,-312.8838};*)*)


(* ::Input:: *)
(*(*x0y = {-154.245790831349,-386.336296245402,-324.009097041169,-197.520957421378,-423.402937680860,-169.415474770594};*)*)


(* ::Subsection:: *)
(*UAV 3 cell, 6  devs*)


(* ::Input:: *)
(*(*u0x=171.356884833481; *)*)


(* ::Input:: *)
(*(*u0y=171.286037581952;*)*)


(* ::Input:: *)
(*(*x0x = {364.769221908557,229.277166843565,168.357141752163,312.965873119628,229.224741845818,301.989204258265};*)*)


(* ::Input:: *)
(*(*x0y = {234.743799599321,113.387934748891,218.571064929920,340.500315720564,89.3852233204108,239.796864017628};*)*)


(* ::Subsection:: *)
(*UAV 4 cell,  4 devs*)


(* ::Input:: *)
(*(*u0x=-64.9381515640485; *)*)


(* ::Input:: *)
(*(*u0y=48.6980348567818;*)*)


(* ::Input:: *)
(*(*x0x = {-99.9012409060819,-142.564124041723,44.3646658045471,89.8354502371193};*)*)


(* ::Input:: *)
(*(*x0y = {47.3409492016910,249.654204853808,66.4253963439721,-1.62148713514132};*)*)


(* ::Subsection:: *)
(*UAV 5 cell, 6 devs*)


(* ::Input:: *)
(*(*u0x=65.0523550988083; *)*)


(* ::Input:: *)
(*(*u0y=-406.948776667274;*)*)


(* ::Input:: *)
(*(*x0x = {201.770496950756,213.841984170876,42.1843002995864,93.8697366449182,27.7172615062848,-66.8838250000861};*)*)


(* ::Input:: *)
(*(*x0y = {-366.640347042608,-331.785956591814,-438.280873636693,-463.763375338924,-331.767546726852,-444.848799707344};*)*)


(* ::Subsection:: *)
(*UAV 6 cell,  4 devs*)


(* ::Input:: *)
(*(*u0x=-357.888789929175; *)*)


(* ::Input:: *)
(*(*u0y=-33.1299398908958;*)*)


(* ::Input:: *)
(*(*x0x = {-436.252117532034,-315.234809384303,-441.333487850414,-359.741394536313};*)*)


(* ::Input:: *)
(*(*x0y = {150.396015036176,-67.1081446232431,6.07176771912633,20.2051008661792};*)*)


(* ::Subsection:: *)
(*UAV 7 cell,  9 devs*)


(* ::Input:: *)
(*(*u0x=95.2603196371070; *)*)


(* ::Input:: *)
(*(*u0y=282.146501064287;*)*)


(* ::Input:: *)
(*(*x0x = {-175.740030077146,215.252142866635,-2.29664682434483,259.993570561251,-162.162918799562,40.4106311332884,-37.8359386171428,-53.3129495578848,178.074443690974};*)*)


(* ::Input:: *)
(*(*x0y = {460.376711763411,293.869472037462,309.074634924192,420.848504227632,362.271457304406,319.389664763052,235.946491921929,240.409617092306,289.616832951514};*)*)


(* ::Subsection:: *)
(*UAV 8 cell,  4 devs*)


(* ::Input:: *)
(*(*u0x=123.450823839841; *)*)


(* ::Input:: *)
(*(*u0y=-162.260037110253;*)*)


(* ::Input:: *)
(*(*x0x = {52.4457909989394,102.956838753507,235.402550000830,205.033011225291};*)*)


(* ::Input:: *)
(*(*x0y = {-94.1499418112316,-178.583866218225,-232.903284268120,-223.865683808523};*)*)


(* ::Subsection:: *)
(*UAV 9 cell,   4 devs*)


(* ::Input:: *)
(*(*u0x=302.133231215112; *)*)


(* ::Input:: *)
(*(*u0y=-87.0127608665662;*)*)


(* ::Input:: *)
(*(*x0x = {222.051480410533,344.553057722280,245.628841406264,218.408510734865};*)*)


(* ::Input:: *)
(*(*x0y = {-4.28475589429730,-99.7582364522511,-188.612032542477,-102.963749775095};*)*)


(* ::Subsection:: *)
(*UAV 10 cell,  2 devs*)


(* ::Input:: *)
(*(*u0x=355.549545238428; *)*)


(* ::Input:: *)
(*(*u0y=-15.3829700495814;*)*)


(* ::Input:: *)
(*(*x0x = {439.0670, 389.9279};*)*)


(* ::Input:: *)
(*(*x0y = {110.6865,-28.4250};*)*)


(* ::Title:: *)
(*Joint UAV to Device*)


(* ::Input:: *)
(*(*x0 = Sqrt[x0x^2+x0y^2]*)*)


(* ::Input:: *)
(*u0 = Sqrt[u0x^2+u0y^2]*)


(* ::Input:: *)
(*(*q=Sqrt[(x0x-u0x)^2+(x0y-u0y)^2+hUAV^2]*)*)


(* ::Input:: *)
(*d=Sqrt[R^2+hUAV^2];*)


(* ::Input:: *)
(*Plos[u_]:=1/(1+a Exp[-b ((180 ArcTan[hUAV/Sqrt[u^2-hUAV^2]])/\[Pi]-a)])*)


(* ::Input:: *)
(*Pnlos[u_]:=1-Plos[u]*)


(* ::Section:: *)
(*Device to UAV*)


(* ::Input:: *)
(*wmDU=Sqrt[(R-u0)^2+hUAV^2];*)


(* ::Input:: *)
(*wpDU=Sqrt[(R+u0)^2+hUAV^2];*)


(* ::Input:: *)
(*fw1DU[u_]:=(2 u)/R^2*)


(* ::Input:: *)
(*fw2DU[u_]:=(2 u ArcCos[(u^2+u0^2-d^2)/(2 u0 Sqrt[u^2-hUAV^2])])/(\[Pi] R^2)*)


(* ::Input:: *)
(*fw1DUlos[u_]:=(fw1DU[u] Plos[u])/(NIntegrate[fw1DU[w] Plos[w],{w,hUAV,wmDU}]+NIntegrate[fw2DU[w] Plos[w],{w,wmDU,wpDU}])*)


(* ::Input:: *)
(*fw2DUlos[u_]:=(fw2DU[u] Plos[u])/(NIntegrate[fw1DU[w] Plos[w],{w,hUAV,wmDU}]+NIntegrate[fw2DU[w] Plos[w],{w,wmDU,wpDU}])*)


(* ::Input:: *)
(*fw1DUnlos[u_]:=(fw1DU[u] Pnlos[u])/(NIntegrate[fw1DU[w] Pnlos[w],{w,hUAV,wmDU}]+NIntegrate[fw2DU[w] Pnlos[w],{w,wmDU,wpDU}])*)


(* ::Input:: *)
(*fw2DUnlos[u_]:=(fw2DU[u] Pnlos[u])/(NIntegrate[fw1DU[w] Pnlos[w],{w,hUAV,wmDU}]+NIntegrate[fw2DU[w] Pnlos[w],{w,wmDU,wpDU}])*)


(* ::Input:: *)
(*QduLOS[s_]:=NIntegrate[(1+(s Pd)/(mL (u)^alphaL))^-mL fw1DUlos[u],{u,hUAV,wmDU}]+NIntegrate[(1+(s Pd)/(mL (u)^alphaL))^-mL fw2DUlos[u],{u,wmDU,wpDU}]*)


(* ::Input:: *)
(*QduNLOS[s_]:=NIntegrate[(1+(s Pd)/(mN (u)^alphaN))^-mN fw1DUnlos[u],{u,hUAV,wmDU}]+NIntegrate[(1+(s Pd)/(mN (u)^alphaN))^-mN fw2DUnlos[u],{u,wmDU,wpDU}]*)


(* ::Input:: *)
(*QQduLOS[s_]:=(NIntegrate[Plos[u] fw1DU[u],{u,hUAV,wmDU}]+NIntegrate[Plos[u] fw2DU[u],{u,wmDU,wpDU}])/((NIntegrate[Plos[u] fw1DU[u],{u,hUAV,wmDU}]+NIntegrate[Plos[u] fw2DU[u],{u,wmDU,wpDU}])+(NIntegrate[Pnlos[u] fw1DU[u],{u,hUAV,wmDU}]+NIntegrate[Pnlos[u] fw2DU[u],{u,wmDU,wpDU}]))*QduLOS[s]*)


(* ::Input:: *)
(*QQduNLOS[s_]:=(NIntegrate[Pnlos[u] fw1DU[u],{u,hUAV,wmDU}]+NIntegrate[Pnlos[u] fw2DU[u],{u,wmDU,wpDU}])/((NIntegrate[Plos[u] fw1DU[u],{u,hUAV,wmDU}]+NIntegrate[Plos[u] fw2DU[u],{u,wmDU,wpDU}])+(NIntegrate[Pnlos[u] fw1DU[u],{u,hUAV,wmDU}]+NIntegrate[Pnlos[u] fw2DU[u],{u,wmDU,wpDU}]))*QduNLOS[s]*)


(* ::Input:: *)
(*QQdu[s_]:=QQduLOS[s]+QQduNLOS[s]*)


(* ::Input:: *)
(*Ldu[s_]:=QQdu[s]^(nDev*activeProbDevs-1)*)


(* ::Input:: *)
(*PcondDULOS[Thr_]:= \!\( *)
(*\*UnderoverscriptBox[\(\[Sum]\), \(j = 1\), \(mL\)]\(Binomial[mL, j]\ *)
(*\*SuperscriptBox[\((\(-1\))\), \(j + 1\)]\ Exp[\(-*)
(*\*FractionBox[\(j\ etaL\ Thr\ n0Squared\), *)
(*FractionBox[\(Pd\), *)
(*SuperscriptBox[\((q)\), \(alphaL\)]]]\)]\ Ldu[*)
(*\*FractionBox[\(j\ etaL\ Thr\), *)
(*FractionBox[\(Pd\), *)
(*SuperscriptBox[\((q)\), \(alphaL\)]]]]\)\)*)


(* ::Input:: *)
(*PcondDUNLOS[Thr_]:= \!\( *)
(*\*UnderoverscriptBox[\(\[Sum]\), \(j = 1\), \(mN\)]\(Binomial[mN, j]\ *)
(*\*SuperscriptBox[\((\(-1\))\), \(j + 1\)]\ Exp[\(-*)
(*\*FractionBox[\(j\ etaN\ Thr\ n0Squared\), *)
(*FractionBox[\(Pd\), *)
(*SuperscriptBox[\((q)\), \(alphaN\)]]]\)]\ Ldu[*)
(*\*FractionBox[\(j\ etaN\ Thr\), *)
(*FractionBox[\(Pd\), *)
(*SuperscriptBox[\((q)\), \(alphaN\)]]]]\)\)*)


(* ::Input:: *)
(*PcondDU[Thr_]:=Plos[q]*PcondDULOS[Thr]+Pnlos[q]*PcondDUNLOS[Thr]*)


(* ::Section:: *)
(*UAV to Device*)


(* ::Input:: *)
(*thetaStar[u_]:=ArcCos[Clip[(u^2+x0^2-d^2)/(2 x0 Sqrt[u^2-hUAV^2]),{-1,1}]]*)


(* ::Input:: *)
(*PhiStar[u_]:=ArcCos[Clip[(x0^2+d^2-u^2)/(2 x0 R),{-1,1}]]*)


(* ::Input:: *)
(*Fw1[u_]:=(u^2-hUAV^2)/R^2*)


(* ::Input:: *)
(*Fw2[u_]:=((u^2-hUAV^2) (thetaStar[u]-1/2 Sin[2 thetaStar[u]]))/(\[Pi] R^2)+(PhiStar[u]-1/2 Sin[2 PhiStar[u]])/\[Pi]*)


(* ::Input:: *)
(*fw1[u_]:=(2 u)/R^2*)


(* ::Input:: *)
(*fw2[u_]:=(2 u ArcCos[Clip[(u^2+x0^2-d^2)/(2 x0 Sqrt[u^2-hUAV^2]),{-1,1}]])/(\[Pi] R^2)*)


(* ::Input:: *)
(*wm=N[Sqrt[(R-x0)^2+hUAV^2]]*)


(* ::Input:: *)
(*wp=N[Sqrt[(R+x0)^2+hUAV^2]]*)


(* ::Input:: *)
(*Eln[q_]:=Clip[q^(alphaL/alphaN),{hUAV,wp}]*)


(* ::Input:: *)
(*Enl[q_]:=Clip[q^(alphaN/alphaL),{hUAV,wp}]*)


(* ::Input:: *)
(*Eln[q]*)


(* ::Input:: *)
(*Enl[q]*)


(* ::Subtitle:: *)
(*LOS Part;*)


(* ::Input:: *)
(*fw1UDlos[u_,q_]:=(fw1[u] Plos[u])/(NIntegrate[fw1[w] Plos[w],{w,q,wm}]+NIntegrate[fw2[w] Plos[w],{w,wm,wp}])*)


(* ::Input:: *)
(*fw2UDlos[u_,q_]:=(fw2[u] Plos[u])/(NIntegrate[fw1[w] Plos[w],{w,q,wm}]+NIntegrate[fw2[w] Plos[w],{w,wm,wp}])*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*fw1UDlosToNlos[u_,q_]:=(fw1[u] Pnlos[u])/(NIntegrate[fw1[w] Pnlos[w],{w,Eln[q],wm}]+NIntegrate[fw2[w] Pnlos[w],{w,wm,wp}])*)


(* ::Input:: *)
(*fw2UDlosToNlos[u_,q_]:=(fw2[u] Pnlos[u])/(NIntegrate[fw1[w] Pnlos[w],{w,Eln[q],wm}]+NIntegrate[fw2[w] Pnlos[w],{w,wm,wp}])*)


(* ::Input:: *)
(**)
(**)
(**)


(* ::Input:: *)
(*Laplace1LOSLOS[s_,q_]:=NIntegrate[(1+(s Pu)/(mL (u)^alphaL))^-mL fw1UDlos[u,q],{u,q,wm}]+NIntegrate[(1+(s Pu)/(mL (u)^alphaL))^-mL fw2UDlos[u,q],{u,wm,wp}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*Laplace1LOSNLOS[s_,q_]:=NIntegrate[(1+(s Pu)/(mN (u)^alphaN))^-mN fw1UDlosToNlos[u,q],{u,Eln[q],wm}]+NIntegrate[(1+(s Pu)/(mN (u)^alphaN))^-mN fw2UDlosToNlos[u,q],{u,wm,wp}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*La1forLOS[s_,q_]:=(((NIntegrate[Plos[u] fw1[u],{u,q,wm}]+NIntegrate[Plos[u] fw2[u],{u,wm,wp}]) Laplace1LOSLOS[s,q])/((NIntegrate[Plos[u] fw1[u],{u,q,wm}]+NIntegrate[Plos[u] fw2[u],{u,wm,wp}])+( NIntegrate[Pnlos[u] fw1[u],{u,Eln[q],wm}]+NIntegrate[Pnlos[u] fw2[u],{u,wm,wp}]))+(( NIntegrate[Pnlos[u] fw1[u],{u,Eln[q],wm}]+NIntegrate[Pnlos[u] fw2[u],{u,wm,wp}])Laplace1LOSNLOS[s,q])/((NIntegrate[Plos[u] fw1[u],{u,q,wm}]+NIntegrate[Plos[u] fw2[u],{u,wm,wp}])+( NIntegrate[Pnlos[u] fw1[u],{u,Eln[q],wm}]+NIntegrate[Pnlos[u] fw2[u],{u,wm,wp}])))^(nUAV*activeProbUAV-1)*)


(* ::Subtitle:: *)
(*NLOS Part;*)
(**)


(* ::Input:: *)
(*fw1UDnlos[u_,q_]:=(fw1[u] Pnlos[u])/(NIntegrate[fw1[w] Pnlos[w],{w,q,wm}]+NIntegrate[fw2[w] Pnlos[w],{w,wm,wp}])*)


(* ::Input:: *)
(*fw2UDnlos[u_,q_]:=(fw2[u] Pnlos[u])/(NIntegrate[fw1[w] Pnlos[w],{w,q,wm}]+NIntegrate[fw2[w] Pnlos[w],{w,wm,wp}])*)


(* ::Input:: *)
(*fw1UDnlosToLos[u_,q_]:=(fw1[u] Plos[u])/(NIntegrate[fw1[w] Plos[w],{w,First[Boole[{Enl[q]>wm}] wm+Boole[{Enl[q]<wm}] Enl[q]],wm}]+NIntegrate[fw2[w] Plos[w],{w,First[Boole[{Enl[q]<wm}] wm+Boole[{Enl[q]>wm}] Enl[q]],wp}])*)


(* ::Input:: *)
(*fw2UDnlosToLos[u_,q_]:=(fw2[u] Plos[u])/(NIntegrate[fw1[w] Plos[w],{w,First[Boole[{Enl[q]>wm}] wm+Boole[{Enl[q]<wm}] Enl[q]],wm}]+NIntegrate[fw2[w] Plos[w],{w,First[Boole[{Enl[q]<wm}] wm+Boole[{Enl[q]>wm}] Enl[q]],wp}])*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*Laplace1NLOSNLOS[s_,q_]:=NIntegrate[(1+(s Pu)/(mN (u)^alphaN))^-mN fw1UDnlos[u,q],{u,q,wm}]+NIntegrate[(1+(s Pu)/(mN (u)^alphaN))^-mN fw2UDnlos[u,q],{u,wm,wp}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*Laplace1NLOSLOS[s_,q_]:=NIntegrate[(1+(s Pu)/(mL (u)^alphaL))^-mL fw1UDnlosToLos[u,q],{u,First[Boole[{Enl[q]>wm}]*wm+Boole[{Enl[q]<wm}]*Enl[q]],wm}]+NIntegrate[(1+(s Pu)/(mL (u)^alphaL))^-mL fw2UDnlosToLos[u,q],{u,First[Boole[{Enl[q]<wm}]*wm + Boole[{Enl[q]>wm}]*Enl[q]],wp}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*denom1 = (NIntegrate[Plos[u] fw1[u],{u,Enl[q],wm}]+NIntegrate[Plos[u] fw2[u],{u,wm,wp}])*)


(* ::Input:: *)
(*denom2 = (NIntegrate[Plos[u] fw2[u],{u,Enl[q],wp}])*)


(* ::Input:: *)
(*La1forNLOS[s_,q_]:=(((NIntegrate[Pnlos[u] fw1[u],{u,q,wm}]+NIntegrate[Pnlos[u] fw2[u],{u,wm,wp}]) Laplace1NLOSNLOS[s,q])/((NIntegrate[Pnlos[u] fw1[u],{u,q,wm}]+NIntegrate[Pnlos[u] fw2[u],{u,wm,wp}])+(denom1*Boole[{Enl[q]<wm}]+denom2*Boole[{Enl[q]>wm}]))+((denom1*Boole[{Enl[q]<wm}]+denom2*Boole[{Enl[q]>wm}]) Laplace1NLOSLOS[s,q])/((NIntegrate[Pnlos[u] fw1[u],{u,q,wm}]+NIntegrate[Pnlos[u] fw2[u],{u,wm,wp}])+(denom1*Boole[{Enl[q]<wm}]+denom2*Boole[{Enl[q]>wm}])))^(nUAV*activeProbUAV-1)*)


(* ::Input:: *)
(**)
(**)
(**)


(* ::Subtitle:: *)
(*Conditional Prob Calculation;*)


(* ::Input:: *)
(*PcondUDLOS[Thr_]:=(\!\( *)
(*\*UnderoverscriptBox[\(\[Sum]\), \(j = 1\), \(mL\)]\(Binomial[mL, j]\ *)
(*\*SuperscriptBox[\((\(-1\))\), \(j + 1\)]\ Exp[\(-*)
(*\*FractionBox[\(j\ etaL\ Thr\ n0Squared\), *)
(*FractionBox[\(Pu\), *)
(*SuperscriptBox[\((q)\), \(alphaL\)]]]\)]\ La1forLOS[*)
(*\*FractionBox[\(j\ etaL\ Thr\), *)
(*FractionBox[\(Pu\), *)
(*SuperscriptBox[\((q)\), \(alphaL\)]]], q]\)\))*)


(* ::Input:: *)
(*PcondUDNLOS[Thr_]:=(\!\( *)
(*\*UnderoverscriptBox[\(\[Sum]\), \(j = 1\), \(mN\)]\(Binomial[mN, j]\ *)
(*\*SuperscriptBox[\((\(-1\))\), \(j + 1\)]\ Exp[\(-*)
(*\*FractionBox[\(j\ etaN\ Thr\ n0Squared\), *)
(*FractionBox[\(Pu\), *)
(*SuperscriptBox[\((q)\), \(alphaN\)]]]\)]\ La1forNLOS[*)
(*\*FractionBox[\(j\ etaN\ Thr\), *)
(*FractionBox[\(Pu\), *)
(*SuperscriptBox[\((q)\), \(alphaN\)]]], q]\)\))*)


(* ::Input:: *)
(*PcondUD[Thr_]:=Plos[q]*PcondUDLOS[Thr]+Pnlos[q]*PcondUDNLOS[Thr]*)


(* ::Subtitle:: *)
(*Joint calculation*)


(* ::Input:: *)
(*P[Thr_]:=Plos[q]*(PcondDULOS[Thr]*PcondUDLOS[Thr])+Pnlos[q]*(PcondDUNLOS[Thr]*PcondUDNLOS[Thr])*)


(* ::Input:: *)
(*tauu = Range[-20,10,2];*)
(*Tau = Transpose[{10^(tauu/10)}];*)


(* ::Input:: *)
(*output=Flatten@Table[P@@Tau[[i]],{i,Length[Tau]}]*)


(* ::Input:: *)
(*(*Quiet[For[i=1,i<=Length[x0x],i++,Q=Sqrt[(x0x-u0x)^2+(x0y-u0y)^2+hUAV^2];q=Q[[i]];X0 = Sqrt[x0x^2+x0y^2];x0=X0[[i]];output=Flatten@Table[P@@Tau[[i]],{i,Length[Tau]}];Print[StringJoin[ToString/@output]]]]*)*)
