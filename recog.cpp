#include<algorithm>
#include<iostream>
#include<cstring>
#include<cstdio>
#include<ctime>
#include<cmath>
#define sqr(x) ((x)*(x))
using namespace std;
inline int read(){
	int x=0,f=1;char ch;
	do{ch=getchar();if(ch=='-')f=-1;}while(ch<'0'||ch>'9');
	do{x=x*10+ch-'0';ch=getchar();}while(ch>='0'&&ch<='9');
	return x*f;
}
const int N=784,M=10,D=30,T=50000,LT=50,AT=(1<<7)-1;
/*	N: Input nodes num
	M: Output nodes num
	D: Coverd nodes num
	T: Max Input num
	LT: Learn time
	AT: Auto save time
*/
char Dinput[]="recog.data",Linput[]="train.in",Loutput[]="train.out",Tinput[]="test.in",Toutput[]="test.out";
/*	DInput: Data file
	Linput: Learning data input
	Loutput: Learning data output
	Tinput: Test data input
	Toutput: Test data output
*/
const double EPS=1e-1,AEPS=5e-2,A=1,B=1,SETA=0.05,SMC=0.1;
double Eta=SETA,MC=SMC;
/*	Eta/SETA: Learning speed
	MC/SMC: 
	A: Sigmoid-A
	B: Sigmoid-B
	AEPS: All EPS
*/
inline double getrand(){
	double ans=(double)rand()/(double)RAND_MAX*2;
	ans-=1;
	ans*=sqrt(6)/sqrt(N+M);
	return ans;
}
int idn[N],idd[D],idm[M];
double G1[N][D],G2[D][M],DG1[N][D],DG2[D][M];
double In[T][N],Out[T][M];
struct Node{
	double val,b,d,Db;
}node[N+D+M];
void Getdata(){
	FILE *fin=fopen(Dinput,"r");
	for (int i=0;i<N;++i){
		for (int j=0;j<D;++j) fscanf(fin,"%lf",&G1[i][j]),DG1[i][j]=0;
	}
	for (int i=0;i<D;++i){
		for (int j=0;j<M;++j) fscanf(fin,"%lf",&G2[i][j]),DG2[i][j]=0;
	}
	for (int i=0;i<D;++i) fscanf(fin,"%lf",&node[idd[i]].b),node[idd[i]].Db=0;
	for (int i=0;i<M;++i) fscanf(fin,"%lf",&node[idm[i]].b),node[idm[i]].Db=0;
	fscanf(fin,"%lf",&Eta);
	fscanf(fin,"%lf",&MC);
	fclose(fin);
}
void Outdata(){
	FILE *fout=fopen(Dinput,"w");
	for (int i=0;i<N;++i){
		for (int j=0;j<D;++j) fprintf(fout,"%.15lf ",G1[i][j]);
		fprintf(fout,"\n");
	}
	for (int i=0;i<D;++i){
		for (int j=0;j<M;++j) fprintf(fout,"%.15lf ",G2[i][j]);
		fprintf(fout,"\n");
	}
	for (int i=0;i<D;++i) fprintf(fout,"%.15lf ",node[idd[i]].b);
	fprintf(fout,"\n");
	for (int i=0;i<M;++i) fprintf(fout,"%.15lf ",node[idm[i]].b);
	fprintf(fout,"\n");
	fprintf(fout,"%.6lf\n",Eta);
	fprintf(fout,"%.6lf\n",MC);
	fclose(fout);
}
inline double Sigmoid(double x){
	return A/(1.+exp(-B*x));
}
void Forward(int wh){
	for (int i=0;i<N;++i) node[i].val=In[wh][i];
	for (int i=0;i<D;++i) node[idd[i]].val=0;
	for (int i=0;i<M;++i) node[idm[i]].val=0;
	for (int j=0;j<D;++j){
		for (int i=0;i<N;++i) node[idd[j]].val+=G1[i][j]*node[idn[i]].val;
		node[idd[j]].val+=node[idd[j]].b;
		node[idd[j]].val=Sigmoid(node[idd[j]].val);
	}
	for (int j=0;j<M;++j){
		for (int i=0;i<D;++i) node[idm[j]].val+=G2[i][j]*node[idd[i]].val;
		node[idm[j]].val+=node[idm[j]].b;
		node[idm[j]].val=Sigmoid(node[idm[j]].val);
	}
}
double getdelta(int wh){
	double ans=0;
	for (int i=0;i<M;++i) ans+=sqr(node[idm[i]].val-Out[wh][i]);
	ans/=2.;
	return ans;
}
void Backward(int wh){
	for (int i=0;i<M;++i){
		node[idm[i]].d=node[idm[i]].val-Out[wh][i];
		node[idm[i]].d*=node[idm[i]].val*(A-node[idm[i]].val)*B/A;
		for (int j=0;j<D;++j) DG2[j][i]=(1.-MC)*node[idd[j]].val*node[idm[i]].d*Eta+MC*DG2[j][i];
		node[idm[i]].Db=(1.-MC)*node[idm[i]].d*Eta+MC*node[idm[i]].Db;
	}
	for (int i=0;i<D;++i){
		node[idd[i]].d=0;
		for (int j=0;j<M;++j) node[idd[i]].d+=node[idm[j]].d*G2[i][j];
		node[idd[i]].d*=node[idd[i]].val*(A-node[idd[i]].val)*B/A;
		for (int j=0;j<N;++j) DG1[j][i]=(1.-MC)*node[idn[j]].val*node[idd[i]].d*Eta+MC*DG1[j][i];
		node[idd[i]].Db=(1.-MC)*node[idd[i]].d*Eta+MC*node[idd[i]].Db;
	}
	for (int i=0;i<N;++i){
		for (int j=0;j<D;++j) G1[i][j]-=DG1[i][j];
	}
	for (int i=0;i<D;++i){
		for (int j=0;j<M;++j) G2[i][j]-=DG2[i][j];
	}
	for (int i=0;i<D;++i) node[idd[i]].b-=node[idd[i]].Db;
	for (int i=0;i<M;++i) node[idm[i]].b-=node[idm[i]].Db;
}
void Learn(){
	Getdata();
	printf("Start get input");
	FILE *Inx=fopen(Linput,"r"),*Iny=fopen(Loutput,"r");
	int Data_num,now=1;
	fscanf(Inx,"%d",&Data_num);
	for (int i=0;i<Data_num;++i){
		if (i>Data_num/10*now) putchar('.'),++now;
		for (int j=0;j<N;++j) fscanf(Inx,"%lf",&In[i][j]),In[i][j]*=2.,In[i][j]-=1;
//		for (int j=0;j<M;++j) fscanf(Iny,"%lf",&Out[i][j]);
		for (int j=0;j<M;++j) Out[i][j]=0;
		int wh;fscanf(Iny,"%d",&wh);
		Out[i][wh]=A;
	}
	fclose(Inx),fclose(Iny);
	printf(".\nGet input complete.\n");
	double lastsum=-1;
//	for (int NowTime=0;NowTime<LT;++NowTime){
	for (int NowTime=0;;++NowTime){
		for (int wh=0;wh<Data_num;++wh){
			double delta=0;
			if (wh%1000==0) cerr<<wh<<endl;
			while (1){
				Forward(wh);
				delta=getdelta(wh);
				if (delta<EPS) break;
				Backward(wh);
			}
		}
		double deltasum=0;
		for (int wh=0;wh<Data_num;++wh) Forward(wh),deltasum+=getdelta(wh);
		deltasum/=(double)Data_num;
		printf("%.6lf\n",deltasum);
		if (deltasum<AEPS) break;
		if (lastsum>=0){
			if (deltasum>1.04*lastsum){
				printf("MC  changed.  %.3lf",MC);
				MC=0;
				printf("-->%.3lf\n",MC);
				printf("Eta changed.  %.3lf",Eta);
				Eta*=.7;
				printf("-->%.3lf\n",Eta);
			}
			if (deltasum<0.98*lastsum){
				printf("MC  changed.  %.3lf",MC);
				MC=SMC;
				printf("-->%.3lf\n",MC);
				printf("Eta changed.  %.3lf",Eta);
				Eta*=1.05;
				printf("-->%.3lf\n",Eta);
			}
		}
		lastsum=deltasum;
//		if ((NowTime&AT)==AT){
			Outdata();
			printf("Auto save complete.\n");
//		}
	}
	Outdata();
}
void Test(){
	Getdata();
	printf("Start test");
	FILE *Inx=fopen(Tinput,"r"),*Outy=fopen(Toutput,"w");
	int Data_num,now=1;
	fscanf(Inx,"%d",&Data_num);
	for (int i=0;i<Data_num;++i){
		if (i>Data_num/10*now) putchar('.'),++now;
		for (int j=0;j<N;++j) fscanf(Inx,"%lf",&In[0][j]),In[0][j]*=2.,In[0][j]-=1;;
		Forward(0);
//		for (int j=0;j<M;++j) fprintf(Outy,"%.5lf ",node[idm[j]].val);
		double mx=-1;int tans=0;
		for (int j=0;j<M;++j) if(node[idm[j]].val>mx) mx=node[idm[j]].val,tans=j;
		fprintf(Outy,"%d",tans);
		fprintf(Outy,"\n");
	}
	fclose(Inx),fclose(Outy);
	printf(".\nTest complete.\n");
}
void Clear(){
	printf("Data will be deleted.\nAre you sure?(Y/N):");
	char ch=getchar();
	if (ch=='Y'||ch=='y'){
		for (int i=0;i<N;++i){
			for (int j=0;j<D;++j) G1[i][j]=getrand();
		}
		for (int i=0;i<D;++i){
			for (int j=0;j<M;++j) G2[i][j]=getrand();
		}
		for (int i=0;i<D;++i) node[idd[i]].b=getrand();
		for (int i=0;i<M;++i) node[idm[i]].b=getrand();
		Eta=SETA;
		Outdata();
		printf("Data clear complete.\n");
	}
}
int main(){
	srand((unsigned)time(0)),srand(rand());
	for (int i=0;i<N;++i) idn[i]=i;
	for (int i=0;i<D;++i) idd[i]=N+i;
	for (int i=0;i<M;++i) idm[i]=N+D+i;
	while (1){
		system("cls");
		printf("Operation type:");
		int opt=read();
		//0: Data clear
		//1: Learn
		//2: test
		//233: Quit
		if (opt==0) Clear();
		else if (opt==1) printf("Start learn.\n"),Learn();
		else if (opt==2) Test();
		else if (opt==233) break;
		else printf("Error operation type!\n");
	}
	printf("Goodbye.");
	return 0;
}
