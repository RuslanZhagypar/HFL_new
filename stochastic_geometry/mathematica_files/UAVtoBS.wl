(* ::Package:: *)

(* ::Input:: *)
(*hUAV=120 ;*)


(* ::Input:: *)
(*x0x = 0;*)


(* ::Input:: *)
(*x0y=0;*)


(* ::Input:: *)
(*x0 = Sqrt[x0x^2+x0y^2];*)


(* ::Input:: *)
(*u0x = 30;*)


(* ::Input:: *)
(*u0y=282;*)


(* ::Input:: *)
(*u0 = Sqrt[u0x^2+u0y^2];*)


(* ::Input:: *)
(*q = EuclideanDistance[{x0x,x0y,0},{u0x,u0y=282,hUAV}]*)


(* ::Input:: *)
(*(*q=Sqrt[x^2+y^2+hUAV^2]*)*)


(* ::Input:: *)
(*nUAV = 10;*)


(* ::Input:: *)
(*RBs=5;*)


(* ::Input:: *)
(*activeProb = ((nUAV)/RBs)/nUAV*)


(* ::Input:: *)
(*R=500(*Radius of a cluster*);*)


(* ::Input:: *)
(*alphaL=2.0(*path loss exponent*);*)


(* ::Input:: *)
(*alphaN=3.5(*path loss exponent*);*)


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


(* ::Section:: *)
(*UAV to BS*)


(* ::Input:: *)
(*d=Sqrt[R^2+hUAV^2];*)


(* ::Input:: *)
(*Plos[u_]:=1/(1+a Exp[-b ((180 ArcTan[hUAV/Sqrt[u^2-hUAV^2]])/\[Pi]-a)])*)


(* ::Input:: *)
(*Pnlos[u_]:=1-Plos[u]*)


(* ::Input:: *)
(*fw1[u_]:=((2 u)/(R^2))        (*h<u<d*)*)


(* ::Input:: *)
(*Eln[q_]:=Clip[q^(alphaL/alphaN),{hUAV,d}]*)


(* ::Input:: *)
(*Enl[q_]:=Clip[q^(alphaN/alphaL),{hUAV,d}]*)


(* ::Input:: *)
(*Eln[q]*)


(* ::Input:: *)
(*Enl[q]*)


(* ::Subtitle:: *)
(*LOS Part;*)


(* ::Input:: *)
(*fw1UDlos[u_,q_]:=(fw1[u] Plos[u])/NIntegrate[fw1[w] Plos[w],{w,hUAV,d}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*fw1UDlosToNlos[u_,q_]:=(fw1[u] Pnlos[u])/NIntegrate[fw1[w] Pnlos[w],{w,hUAV,d}]*)


(* ::Input:: *)
(**)
(**)
(**)


(* ::Input:: *)
(*Laplace1LOSLOS[s_,q_]:=NIntegrate[(1+(s Pu)/(mL (u)^alphaL))^-mL fw1UDlos[u,q],{u,hUAV,d}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*Laplace1LOSNLOS[s_,q_]:=NIntegrate[(1+(s Pu)/(mN (u)^alphaN))^-mN fw1UDlosToNlos[u,q],{u,hUAV,d}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*La1forLOS[s_,q_]:=(((NIntegrate[Plos[u] fw1[u],{u,hUAV,d}]) Laplace1LOSLOS[s,q])/((NIntegrate[Plos[u] fw1[u],{u,hUAV,d}])+( NIntegrate[Pnlos[u] fw1[u],{u,hUAV,d}]))+(( NIntegrate[Pnlos[u] fw1[u],{u,hUAV,d}])Laplace1LOSNLOS[s,q])/((NIntegrate[Plos[u] fw1[u],{u,hUAV,d}])+( NIntegrate[Pnlos[u] fw1[u],{u,hUAV,d}])))^(nUAV*activeProb-1)*)


(* ::Subtitle:: *)
(*NLOS Part;*)
(**)


(* ::Input:: *)
(*fw1UDnlos[u_,q_]:=(fw1[u] Pnlos[u])/NIntegrate[fw1[w] Pnlos[w],{w,hUAV,d}]*)


(* ::Input:: *)
(*fw1UDnlosToLos[u_,q_]:=(fw1[u] Plos[u])/NIntegrate[fw1[w] Plos[w],{w,hUAV,d}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*Laplace1NLOSNLOS[s_,q_]:=NIntegrate[(1+(s Pu)/(mN (u)^alphaN))^-mN fw1UDnlos[u,q],{u,hUAV,d}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*Laplace1NLOSLOS[s_,q_]:=NIntegrate[(1+(s Pu)/(mL (u)^alphaL))^-mL fw1UDnlosToLos[u,q],{u,hUAV,d}]*)


(* ::Input:: *)
(**)
(**)


(* ::Input:: *)
(*La1forNLOS[s_,q_]:=(((NIntegrate[Pnlos[u] fw1[u],{u,hUAV,d}]) Laplace1NLOSNLOS[s,q])/((NIntegrate[Pnlos[u] fw1[u],{u,hUAV,d}])+(NIntegrate[Plos[u] fw1[u],{u,hUAV,d}]))+(NIntegrate[Plos[u] fw1[u],{u,hUAV,d}] Laplace1NLOSLOS[s,q])/((NIntegrate[Pnlos[u] fw1[u],{u,hUAV,d}])+(NIntegrate[Plos[u] fw1[u],{u,hUAV,d}])))^(nUAV*activeProb-1)*)


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
(*P[Thr_]:=Plos[q]*(PcondUDLOS[Thr])+Pnlos[q]*(PcondUDNLOS[Thr])*)


(* ::Input:: *)
(*tauu = Range[-20,10,2];*)
(*Tau = Transpose[{10^(tauu/10)}];*)


(* ::Input:: *)
(*output=Flatten@Table[P@@Tau[[i]],{i,Length[Tau]}]*)



