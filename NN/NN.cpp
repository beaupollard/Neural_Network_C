// cpp_compiler_options_openmp.cpp
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>  
#include <omp.h>

#define Yin	128
//#define S_cols	8112
//#define S_cols_val	1024
#define S_cols	5424
#define S_cols_val	688
#define Epochs 250
//#define ofstream myfile

using namespace std;
using std::cout;
using std::generate;
using std::vector;

void File_Save(float *in, int row, int col, string name) {
	ofstream myfile;
	myfile.open(name);
	if (myfile.is_open())
	{
		myfile << row;
		myfile << "\n";
		myfile << col;
		myfile << "\n";
		for (int i = 0; i < row*col; ++i) {
			myfile << in[i];
			myfile << "\n";
		}
		myfile.close();

	}
}

void ReLUC(float *W1, float *b1, float *S, float *a1, float *da1, int row, int col) {
	float crand = 0;
	for (int i = 0; i < row; ++i) {
		crand = 0;
		for (int k = 0; k < col; k++) {

			crand += W1[i * col + k] * S[k];

		}

		crand += b1[i];

		if (crand < 0) {
			a1[i] = 0.;
		}
		else if (crand > 12) {
			a1[i] = 12;
		}
		else {
			a1[i] = crand;
		}
		if (crand < 0) {
			da1[i] = 0.;
		}
		else if (crand > 12) {
			da1[i] = 0.;
		}
		else {
			da1[i] = 1;
		}
	}
}

void SoftMax(float *W2, float *b2, float *S, float *a2, float *da2, float *a1, int row, int col) {
	float crand = 0;
	float suma = 0;
	//if (row < N)
	for (int i = 0; i < row; ++i) {
		crand = 0;
		for (int k = 0; k < col; k++) {
			crand += W2[i * col + k] * a1[k];
		}
		crand += b2[i];
		a2[i] = exp(crand);
		suma += a2[i];
	}
	for (int i = 0; i < row; ++i) {
		da2[i] = a2[i] / suma - a2[i] * a2[i] / (suma*suma);
		a2[i] = a2[i] / suma;
	}
}

void NN_Create(int randvar, float *Strain, float *Stest, float *Ytrain, float *Ytest) {
	string str = to_string(randvar);
	srand(randvar);
	int N1 = 1 << 9;
	int N2 = 16;
	int rancol = 0;
	// Host vectors
	vector<int> srandInt(S_cols);
	vector<float> C(Epochs);
	vector<float> C2(Epochs);
	vector<float> W1(N1 * Yin);
	vector<float> b1(N1);
	vector<float> da1(N1);
	vector<float> a1(N1);
	vector<float> W2(N2 * N1);
	vector<float> b2(N2);
	vector <float> a2(N2);
	vector<float> da2(N2);
	vector<float> dCdw1(N1 * Yin);
	vector<float> dCdb1(N1);
	vector<float> dCda1(N1);
	vector<float> dCda2(N2);
	vector<float> dCdw2(N2 * N1);
	vector<float> dCdb2(N2);
	vector<float> S(Yin);
	vector<float> Y(16);

	float * pW1 = &W1[0];
	float * pb1 = &b1[0];
	float * pW2 = &W2[0];
	float * pb2 = &b2[0];
	float * pS = &S[0];
	float * pa1 = &a1[0];
	float * pda1 = &da1[0];
	float * pa2 = &a2[0];
	float * pda2 = &da2[0];


	float suma2 = 0;
	float sumW2 = 0;
	float sumW1 = 0;
	float sumb2 = 0;
	float sumb1 = 0;
	float Cin = 0;
	int c_out = 0;
	float maxa2;
	int indexa2;
	float min_err = 1;
	float eta = 0.0015;
	float nu = 0.00000001;
	float crand;

	// Initialize matrices
	generate(W1.begin(), W1.end(), []() { return rand() % 10000; });
	generate(b1.begin(), b1.end(), []() { return rand() % 10000; });
	generate(a1.begin(), a1.end(), []() { return rand() % 10000; });
	generate(da1.begin(), da1.end(), []() { return rand() % 10000; });
	generate(W2.begin(), W2.end(), []() { return rand() % 10000; });
	generate(b2.begin(), b2.end(), []() { return rand() % 10000; });
	generate(a2.begin(), a2.end(), []() { return rand() % 10000; });
	generate(da2.begin(), da2.end(), []() { return rand() % 10000; });

	for (int i = 0; i < N1 * Yin; ++i) {
		W1[i] = (W1[i] - 10000 / 2) / 20000;
	}
	for (int i = 0; i < N1; ++i) {
		b1[i] = (b1[i] - 10000 / 2) / 20000;
	}
	for (int i = 0; i < N1; ++i) {
		//b1[i] = (b1[i] - 50) / 200;
		a1[i] = 0.0;
		da1[i] = 0.0;
	}
	for (int i = 0; i < N2*N1; ++i) {
		W2[i] = (W2[i] - 10000 / 2) / 20000;
	}
	for (int i = 0; i < N2; ++i) {
		b2[i] = (b2[i] - 10000 / 2) / 20000;
		a2[i] = 0.0;
		da2[i] = 0.0;
	}

	for (int counter = 0; counter < Epochs; ++counter)
	{
		if (counter % 25 == 0 && counter > 0) {
			eta = eta * .75;
			printf("New Eta \n");
		}
		for (int h1 = 0; h1 < S_cols; ++h1) {
			srandInt[h1] = 0;
		}
		C[counter] = 0;

		for (int h1 = 0; h1 < S_cols; ++h1) {

			//for (int h1 = 0; h1 < 50; ++h1) {
			rancol = rand() % S_cols;
			if (srandInt[rancol] == 1) {
				c_out = 0;
				int i = 0;
				while (c_out == 0) {
					if (srandInt[i] == 0) {
						rancol = i;
						c_out = 1;
					}
					i = i + 1;
				}
				srandInt[rancol] = 1;
			}
			else {
				srandInt[rancol] = 1;
			}

			// Allocate device memory
			for (int i = 0; i < Yin; ++i) {

				S[i] = Strain[Yin * rancol + i];

			}
			for (int i = 0; i < 16; ++i) {
				Y[i] = Ytrain[16 * rancol + i];
			}
			ReLUC(pW1, pb1, pS, pa1, pda1, N1, Yin);

			//----------- Layer 2-----------------
			float crand = 0;

			SoftMax(pW2, pb2, pS, pa2, pda2, pa1, N2, N1);
			Cin = 0;

			for (int i = 0; i < 16; ++i) {
				dCda2[i] = -Y[i] / a2[i] + (1 - Y[i]) / (1 - a2[i]);
			}
			for (int i = 0; i < 16; ++i) {
				Cin += -Y[i] * log(a2[i]) - (1 - Y[i])*log(1 - a2[i]);
			}

			for (int i = 0; i < N2; ++i) {
				for (int j = 0; j < N1; ++j) {
					dCdw2[i*N1 + j] = dCda2[i] * da2[i] * a1[j];
					W2[i*N1 + j] -= eta * dCdw2[i*N1 + j] + 2 * nu * W2[i*N1 + j];
				}
				dCdb2[i] = dCda2[i] * da2[i];
				b2[i] -= eta * dCdb2[i] + 2 * nu * b2[i];
			}
			for (int i = 0; i < N1; ++i) {
				suma2 = 0;
				for (int j = 0; j < N2; ++j) {
					suma2 += W2[i + j * N1] * dCda2[j] * da2[j];
				}
				dCda1[i] = suma2;
			}
			for (int i = 0; i < N1; ++i) {
				for (int j = 0; j < Yin; ++j) {
					dCdw1[i*Yin + j] = dCda1[i] * da1[i] * S[j];
					W1[i*Yin + j] -= eta * dCdw1[i*Yin + j] + 2 * nu * W1[i*Yin + j];
				}
				dCdb1[i] = dCda1[i] * da1[i];
				b1[i] -= eta * dCdb1[i] + 2 * nu * b1[i];
			}
			C2[counter] += Cin;

		}

		//printf("C: \t %6f \n", C[counter]);

		// -------- Testing -----------
		Cin = 0;
		for (int h1 = 0; h1 < S_cols_val; ++h1) {

			for (int i = 0; i < Yin; ++i) {

				S[i] = Stest[Yin * h1 + i];

			}
			for (int i = 0; i < 16; ++i) {
				Y[i] = Ytest[16 * h1 + i];
			}
			ReLUC(pW1, pb1, pS, pa1, pda1, N1, Yin);


			//----------- Layer 2-----------------
			SoftMax(pW2, pb2, pS, pa2, pda2, pa1, N2, N1);

			maxa2 = 0;
			indexa2 = 0;
			for (int i = 0; i < 16; ++i) {

				if (a2[i] > maxa2) {
					maxa2 = a2[i];
					indexa2 = i;
				}
			}

			if (Y[indexa2] == 1) {
				Cin += 0;
			}
			else {
				Cin += 1;
			}
		}
		C[counter] = Cin / S_cols_val;
		if (C[counter] < min_err) {
			min_err = C[counter];
			string str2 = "RW1_" + str + ".dat";
			File_Save(pW1, N1, Yin, str2);
			str2 = "Rb1_" + str + ".dat";
			File_Save(pb1, N1, 1, str2);
			str2 = "RW2_" + str + ".dat";
			File_Save(pW2, N2, N1, str2);
			str2 = "Rb2_" + str + ".dat";
			File_Save(pb2, N2, 1, str2);

		}
		printf("C: %6f \t counter: %d\n", C[counter], counter);
		//printf("C2: %6f \n", C2[counter]);
		//myfile.open("C.dat");
		string str2 = "RC_" + str + ".dat";
		File_Save(&C[0], counter, 1, str2);
		//if (myfile.is_open())
		//{
		//	for (int i = 0; i < counter; ++i) {
		//		myfile << C[i];
		//		myfile << "\n";
		//	}
		//	myfile.close();

		//}

	}
	//if (isnan(Cin)) {
	//	printf("Cin became infinite \n");
	//}
	printf("C: %6f \t counter: %d\n", min_err, randvar);
}

int main() {
	// Matrix size of 1024 x 1024;

	vector<float> Stest(S_cols_val * Yin);
	vector<float> Sval(S_cols_val * Yin);
	vector<float> Strain(S_cols * Yin);
	vector<float> Ytest(S_cols_val * 16);
	vector<float> Yval(S_cols_val * 16);
	vector<float> Ytrain(S_cols * 16);

	// Bring in the training data
	ofstream myfile;
	ifstream infile;
	infile.open("D:/bpoll/Experiments_Flap2/Sin_Velos/StrainR.dat");
	//infile.open("c:/users/bpoll/source/repos/test1/test1/Strainp.dat");
	if (infile.is_open()) {
		cout << "reading from the file" << endl;
		for (int i = 0; i < S_cols*Yin; ++i) {
			infile >> Strain[i];
		}
	}
	infile.close();
	infile.open("D:/bpoll/Experiments_Flap2/Sin_Velos/YtrainR.dat");
	//infile.open("c:/users/bpoll/source/repos/test1/test1/Ytrainp.dat");
	if (infile.is_open()) {
		cout << "reading from the file" << endl;
		for (int i = 0; i < S_cols * 16; ++i) {
			infile >> Ytrain[i];
		}
	}
	infile.close();
	// Bring in the test data
	infile.open("D:/bpoll/Experiments_Flap2/Sin_Velos/StestR.dat");
	//infile.open("c:/users/bpoll/source/repos/test1/test1/Stestp.dat");
	if (infile.is_open()) {
		cout << "reading from the file" << endl;
		for (int i = 0; i < S_cols_val*Yin; ++i) {
			infile >> Stest[i];
		}
	}
	infile.close();
	infile.open("D:/bpoll/Experiments_Flap2/Sin_Velos/YtestR.dat");
	//infile.open("c:/users/bpoll/source/repos/test1/test1/Ytestp.dat");
	if (infile.is_open()) {
		cout << "reading from the file" << endl;
		for (int i = 0; i < S_cols_val * 16; ++i) {
			infile >> Ytest[i];
		}
	}
	infile.close();

	//#pragma omp parallel
	//for (int i = 0; i < 1; i++) {
	//	printf("Hello \n");
	//}
	for (int i = 10; i < 100; ++i) {
		
		NN_Create(i, &Strain[0], &Stest[0], &Ytrain[0], &Ytest[0]);
	}

	return 0;
}