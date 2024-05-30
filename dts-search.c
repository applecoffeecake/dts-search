/*
	MIT License

	Copyright (c) 2024 Mohannad Shehadeh

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
*/
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>

// search for (n,k)-DTSs
#define n 13
#define k 4

// M is largest scope to consider, at most 255
#define M 140
#define TRAINING_M 160
#define TRAINING_TRIALS 100

// terminate when scope TS solution is found
#define TS 137

// number of trials and how frequently to print status
#define TRIALS (ULLONG_MAX)
#define STATUS_FREQ (1)

// number of attempt thresholds for generating a mark, a row, a full DTS
#define BLOCK_GEN_THRESH (100)
#define BLOCK_RESTART_THRESH (3)
#define DTS_GEN_THRESH (100*1000)

#define get_Hamming_weight(x) ((__builtin_popcountll(x.q3)+__builtin_popcountll(x.q2)+__builtin_popcountll(x.q1)+__builtin_popcountll(x.q0)))
#define iszero_qull(x) ((!x.q3 && !x.q2 && !x.q1 && !x.q0))
#define isequal_qull(x,y) ((x.q3 == y.q3 && x.q2 == y.q2 && x.q1 == y.q1 && x.q0 == y.q0))

#define OR_qull(y,z,x) x.q3 = y.q3 | z.q3; x.q2 = y.q2 | z.q2; x.q1 = y.q1 | z.q1; x.q0 = y.q0 | z.q0;
#define XOR_qull(y,z,x) x.q3 = y.q3 ^ z.q3; x.q2 = y.q2 ^ z.q2; x.q1 = y.q1 ^ z.q1; x.q0 = y.q0 ^ z.q0;
#define AND_qull(y,z,x) x.q3 = y.q3 & z.q3; x.q2 = y.q2 & z.q2; x.q1 = y.q1 & z.q1; x.q0 = y.q0 & z.q0;

static inline __attribute__((always_inline)) double gaussrand() {
	static int state = 0;
	static double next;
	double v1, v2, R, fac;
	if (state) {
		state = 0;
		return(next);
	} else {
		while (1) {
			v1 = 2.0 * (double)rand()/RAND_MAX - 1.0;
			v2 = 2.0 * (double)rand()/RAND_MAX - 1.0;
			R = v1*v1 + v2*v2;
			if (R < 1.0 && R != 0.0) break;
		}
		fac = sqrt(-2.0 * log(R)/R);
		next = v1 * fac;
		state = 1;
		return (v2 * fac);
	}
}

struct quad_ull {
	unsigned long long int q3;
	unsigned long long int q2;
	unsigned long long int q1;
	unsigned long long int q0;
};

static inline __attribute__((always_inline)) int get_largest_mark_byref(struct quad_ull* x) {
	if (x->q3 != 0) {
		return (63-__builtin_clzll(x->q3)) + 192;
	} else if (x->q2 != 0) {
		return (63-__builtin_clzll(x->q2)) + 128;
	} else if (x->q1 != 0) {
		return (63-__builtin_clzll(x->q1)) + 64;
	} else if (x->q0 != 0) {
		return (63-__builtin_clzll(x->q0));
	} else {
		return -1;
	}
}

void printf_qull(struct quad_ull x) {
	printf("%016llx %016llx %016llx %016llx\n", x.q3, x.q2, x.q1, x.q0);
}

static inline __attribute__((always_inline)) struct quad_ull shift_left(struct quad_ull x, int bit_shift) {
	struct quad_ull y;
	int word_shift = bit_shift >> 6;
	int bit_shift_remainder = bit_shift & 0x3F;
	int word_minus_bit_shift_remainder = 64 - bit_shift_remainder; 
	switch (bit_shift_remainder) {
		case 0:
			switch (word_shift) {
				case 0:
					y.q0 = x.q0;
					y.q1 = x.q1;
					y.q2 = x.q2;
					y.q3 = x.q3;
					break;
				case 1:
					y.q1 = x.q0;
					y.q2 = x.q1;
					y.q3 = x.q2;
					y.q0 = 0ULL;
					break;
				case 2:
					y.q2 = x.q0;
					y.q3 = x.q1;
					y.q0 = 0ULL;
					y.q1 = 0ULL;
					break;
				case 3:
					y.q3 = x.q0;
					y.q0 = 0ULL;
					y.q1 = 0ULL;
					y.q2 = 0ULL;
					break;
			}
			break;
		default:
			switch (word_shift) {
				case 0:
					y.q0 = x.q0 << bit_shift_remainder;
					y.q1 = (x.q1 << bit_shift_remainder) | (x.q0 >> word_minus_bit_shift_remainder);
					y.q2 = (x.q2 << bit_shift_remainder) | (x.q1 >> word_minus_bit_shift_remainder);
					y.q3 = (x.q3 << bit_shift_remainder) | (x.q2 >> word_minus_bit_shift_remainder);
					break;
				case 1:
					y.q1 = x.q0 << bit_shift_remainder;
					y.q2 = (x.q1 << bit_shift_remainder) | (x.q0 >> word_minus_bit_shift_remainder);
					y.q3 = (x.q2 << bit_shift_remainder) | (x.q1 >> word_minus_bit_shift_remainder);
					y.q0 = 0ULL;
					break;
				case 2:
					y.q2 = x.q0 << bit_shift_remainder;
					y.q3 = (x.q1 << bit_shift_remainder) | (x.q0 >> word_minus_bit_shift_remainder);
					y.q0 = 0ULL;
					y.q1 = 0ULL;
					break;
				case 3:
					y.q3 = x.q0 << bit_shift_remainder;
					y.q0 = 0ULL;
					y.q1 = 0ULL;
					y.q2 = 0ULL;
					break;
			}
	}
	return y;
}

static inline __attribute__((always_inline)) struct quad_ull shift_right(struct quad_ull x, int bit_shift) {
	struct quad_ull y;
	int word_shift = bit_shift >> 6;
	int bit_shift_remainder = bit_shift & 0x3F;
	int word_minus_bit_shift_remainder = 64 - bit_shift_remainder; 
	switch (bit_shift_remainder) {
		case 0:
			switch (word_shift) {
				case 0:
					y.q0 = x.q0;
					y.q1 = x.q1;
					y.q2 = x.q2;
					y.q3 = x.q3;
					break;
				case 1:
					y.q3 = 0ULL;
					y.q0 = x.q1;
					y.q1 = x.q2;
					y.q2 = x.q3;
					break;
				case 2:
					y.q2 = 0ULL;
					y.q3 = 0ULL;
					y.q0 = x.q2;
					y.q1 = x.q3;
					break;
				case 3:
					y.q1 = 0ULL;
					y.q2 = 0ULL;
					y.q3 = 0ULL;
					y.q0 = x.q3;
					break;
			}
			break;
		default:
			switch (word_shift) {
				case 0:
					y.q3 = x.q3 >> bit_shift_remainder;
					y.q2 = (x.q2 >> bit_shift_remainder) | (x.q3 << word_minus_bit_shift_remainder);
					y.q1 = (x.q1 >> bit_shift_remainder) | (x.q2 << word_minus_bit_shift_remainder);
					y.q0 = (x.q0 >> bit_shift_remainder) | (x.q1 << word_minus_bit_shift_remainder);
					break;
				case 1:
					y.q2 = x.q3 >> bit_shift_remainder;
					y.q1 = (x.q2 >> bit_shift_remainder) | (x.q3 << word_minus_bit_shift_remainder);
					y.q0 = (x.q1 >> bit_shift_remainder) | (x.q2 << word_minus_bit_shift_remainder);
					y.q3 = 0ULL;
					break;
				case 2:
					y.q1 = x.q3 >> bit_shift_remainder;
					y.q0 = (x.q2 >> bit_shift_remainder) | (x.q3 << word_minus_bit_shift_remainder);
					y.q3 = 0ULL;
					y.q2 = 0ULL;
					break;
				case 3:
					y.q0 = x.q3 >> bit_shift_remainder;
					y.q3 = 0ULL;
					y.q2 = 0ULL;
					y.q1 = 0ULL;
					break;
			}
	}
	return y;
}

static inline __attribute__((always_inline)) void toggle_qull_byref(struct quad_ull* x, int bit) {
	int word = bit >> 6;
	int rem = bit & 0x3F;
	switch (word) {
		case 3:
			x->q3 = x->q3 ^ (1ULL << rem);
			break;
		case 2:
			x->q2 = x->q2 ^ (1ULL << rem);
			break;
		case 1:
			x->q1 = x->q1 ^ (1ULL << rem);
			break;
		case 0:
			x->q0 = x->q0 ^ (1ULL << rem);
			break;
	}
}

int main() {

	printf("M, TS %d, %d\n", M, TS );
	
	time_t then, now;
	then = time(NULL);
	unsigned long long int seed = time(NULL) ^ getpid();
	srand(seed);
	printf("seed = %llu\n", seed);
	printf("RAND_MAX = %d\n", RAND_MAX);
	
	if (CHAR_BIT*sizeof(unsigned long long int) != 64 || M >= 256 || TRAINING_M >= 256) {
		return 1;
	}

	struct quad_ull ZERO_QULL = {.q3 = 0ULL, .q2 = 0ULL, .q1 = 0ULL, .q0 = 0ULL}; 
	struct quad_ull ONE_QULL = {.q3 = 0ULL, .q2 = 0ULL, .q1 = 0ULL, .q0 = 1ULL};
	struct quad_ull best_cumulative_spectrum;
	struct quad_ull best_ruler[n];
	struct quad_ull best_ruler_nat[n];
	struct quad_ull best_spectrum[n];
	int best_scope = INT_MAX;
	for (int i = 0; i < n; i++) {
		best_ruler[i] = ONE_QULL;
		best_ruler_nat[i] = ONE_QULL;
		best_spectrum[i] = ZERO_QULL;
	}
	
	double second_moments[k];
	double means[k];
	int sample_size = 0;
	for (int q = 0; q < k; q++) {
		second_moments[q] = 0.0;
		means[q] = 0.0;
	}
	
	// TRAINING PHASE
	// comments omitted since same code is repeated later
	unsigned long long int training_trial = 0;
	while (training_trial < TRAINING_TRIALS) {
		struct quad_ull ruler[n];
		struct quad_ull ruler_nat[n];
		struct quad_ull spectrum[n];
		for (int q = 0; q < n; q++) {
			ruler[q] = ONE_QULL;
			ruler_nat[q] = ONE_QULL;
			spectrum[q] = ZERO_QULL;
		}
		struct quad_ull cumulative_spectrum = ZERO_QULL;
		int i = 0;
		unsigned long long int dts_gen_iters = 0;
		while (i < n && dts_gen_iters < DTS_GEN_THRESH) {
			dts_gen_iters++;
			struct quad_ull old_cumulative_spectrum = cumulative_spectrum;
			int j = 0;
			unsigned long long int block_restarts = 0; 
			while (j < k && block_restarts < BLOCK_RESTART_THRESH) {
				block_restarts++; 
				ruler[i] = ONE_QULL;
				ruler_nat[i] = ONE_QULL;
				spectrum[i] = ZERO_QULL;
				int largest_mark = 0;
				j = 0;
				cumulative_spectrum = old_cumulative_spectrum; 
				unsigned long long int block_gen_iters = 0;
				while (j < k && block_gen_iters < BLOCK_GEN_THRESH) {
					block_gen_iters++;
					int mark = rand()%TRAINING_M + 1;
					struct quad_ull spectrum_update_L, spectrum_update_R, spectrum_update_LR_OR, spectrum_update_LR_AND;
					struct quad_ull intersection_with_past;
					if (mark > largest_mark) {
						spectrum_update_L = shift_left(ruler[i], mark-largest_mark);
						spectrum_update_R = ZERO_QULL;
					} else {
						spectrum_update_L = shift_right(ruler[i], largest_mark-mark);
						spectrum_update_R = shift_right(ruler_nat[i], mark);
					}
					OR_qull(spectrum_update_L, spectrum_update_R, spectrum_update_LR_OR)
					AND_qull(spectrum_update_L, spectrum_update_R, spectrum_update_LR_AND)
					AND_qull(cumulative_spectrum, spectrum_update_LR_OR, intersection_with_past)
					if (iszero_qull(spectrum_update_LR_AND) && iszero_qull(intersection_with_past)) {
						toggle_qull_byref(&ruler_nat[i], mark);
						if (mark > largest_mark) {
							ruler[i] = spectrum_update_L;
							ruler[i].q0 = ruler[i].q0 | 1ULL;
							largest_mark = mark;
						} else {
							toggle_qull_byref(&ruler[i], largest_mark-mark);
						}
						OR_qull(spectrum[i], spectrum_update_LR_OR, spectrum[i])
						OR_qull(cumulative_spectrum, spectrum_update_LR_OR, cumulative_spectrum)
						j++;
					}
				}
			}
			if (j == k) {
				i++; 
			} else {
				cumulative_spectrum = old_cumulative_spectrum;
				struct quad_ull old_cumulative_spectrum_with_deletion;
				for (int t = 0; t < i; t++) {
					XOR_qull(cumulative_spectrum, spectrum[t], cumulative_spectrum)
					old_cumulative_spectrum_with_deletion = cumulative_spectrum;
					int j = 0;
					unsigned long long int block_restarts = 0; 
					while (j < k && block_restarts < BLOCK_RESTART_THRESH) {
						block_restarts++; 
						ruler[i] = ONE_QULL;
						ruler_nat[i] = ONE_QULL;
						spectrum[i] = ZERO_QULL;
						int largest_mark = 0;
						j = 0;
						cumulative_spectrum = old_cumulative_spectrum_with_deletion; 
						unsigned long long int block_gen_iters = 0;
						while (j < k && block_gen_iters < BLOCK_GEN_THRESH) {
							block_gen_iters++;
							int mark = rand()%TRAINING_M + 1;
							struct quad_ull spectrum_update_L, spectrum_update_R, spectrum_update_LR_OR, spectrum_update_LR_AND;
							struct quad_ull intersection_with_past;
							if (mark > largest_mark) {
								spectrum_update_L = shift_left(ruler[i], mark-largest_mark);
								spectrum_update_R = ZERO_QULL;
							} else {
								spectrum_update_L = shift_right(ruler[i], largest_mark-mark);
								spectrum_update_R = shift_right(ruler_nat[i], mark);
							}
							OR_qull(spectrum_update_L, spectrum_update_R, spectrum_update_LR_OR)
							AND_qull(spectrum_update_L, spectrum_update_R, spectrum_update_LR_AND)
							AND_qull(cumulative_spectrum, spectrum_update_LR_OR, intersection_with_past)
							if (iszero_qull(spectrum_update_LR_AND) && iszero_qull(intersection_with_past)) {
								toggle_qull_byref(&ruler_nat[i], mark);
								if (mark > largest_mark) {
									ruler[i] = spectrum_update_L;
									ruler[i].q0 = ruler[i].q0 | 1ULL;
									largest_mark = mark;
								} else {
									toggle_qull_byref(&ruler[i], largest_mark-mark);
								}
								OR_qull(spectrum[i], spectrum_update_LR_OR, spectrum[i])
								OR_qull(cumulative_spectrum, spectrum_update_LR_OR, cumulative_spectrum)
								j++;
							}
						}
					}
					if (j == k) {
						ruler[t] = ruler[i];
						ruler_nat[t] = ruler_nat[i];
						spectrum[t] = spectrum[i];
						break;
					} else {
						cumulative_spectrum = old_cumulative_spectrum;
					}
				}
			}
		}
		if (i == n) {
			int scope = -1;
			for (int q = 0; q < n; q++) {
				int length = get_largest_mark_byref(&ruler[q]);
				if (length > scope) {
					scope = length;
				}
			}
			printf("successful training trial: training_trial = %llu, scope = %d \n", training_trial, scope);
			int decoded_ruler[n][k+1];
			for (int block = 0; block < n; block++) {
				for (int q = k; q >= 0; q--) {
					decoded_ruler[block][q] = get_largest_mark_byref(&ruler_nat[block]);
					struct quad_ull one_hot;
					one_hot = shift_left(ONE_QULL, get_largest_mark_byref(&ruler_nat[block]));
					XOR_qull(ruler_nat[block], one_hot, ruler_nat[block])
				}
			}
			for (int qq = 0; qq < n; qq++) {
				for (int q = 0; q < k; q++) {
					means[q] += (double)decoded_ruler[qq][q+1];
					second_moments[q] += (double)decoded_ruler[qq][q+1]*decoded_ruler[qq][q+1];
					
				}
				sample_size += 1;
			}
		}
		training_trial++;
	}
	for (int q = 0; q < k; q++) {
		means[q] *= 1.0/sample_size;
		printf("mean %d = %.5lf \n", q, means[q]);
	}
	double sigmas[k];
	for (int q = 0; q < k; q++) {
		sigmas[q] = sqrt(second_moments[q]*1.0/sample_size - means[q]*means[q]);
		printf("sigma %d = %.5lf \n", q, sigmas[q]);
	}
	printf("normalized:\n");
	for (int q = 0; q < k; q++) {
		means[q] *= 1.0/TRAINING_M;
		printf("mean %d = %.5lf \n", q, means[q]);
	}
	for (int q = 0; q < k; q++) {
		sigmas[q] *= 1.0/TRAINING_M;
		printf("sigma %d = %.5lf \n", q, sigmas[q]);
	}
	
	// MAIN PHASE
	unsigned long long int trial = 0;
	while (trial < TRIALS && best_scope > TS) {
		struct quad_ull ruler[n];
		struct quad_ull ruler_nat[n];
		struct quad_ull spectrum[n];
		for (int q = 0; q < n; q++) {
			ruler[q] = ONE_QULL;
			ruler_nat[q] = ONE_QULL;
			spectrum[q] = ZERO_QULL;
		}
		struct quad_ull cumulative_spectrum = ZERO_QULL;
		int mark_lim = M;
		if (best_scope <= mark_lim) {
			mark_lim = best_scope-1;
		}
		// START OF DTS GENERATION PROCEDURE
		int i = 0; // blocks populated
		unsigned long long int dts_gen_iters = 0;
		while (i < n && dts_gen_iters < DTS_GEN_THRESH) {
			dts_gen_iters++;
			struct quad_ull old_cumulative_spectrum = cumulative_spectrum;
			// START OF BLOCK GENERATION PROCEDURE
			int j = 0; // marks populated excluding zero mark
			unsigned long long int block_restarts = 0; 
			while (j < k && block_restarts < BLOCK_RESTART_THRESH) { 
				block_restarts++; 
				ruler[i] = ONE_QULL;
				ruler_nat[i] = ONE_QULL;
				spectrum[i] = ZERO_QULL;
				int largest_mark = 0;
				j = 0;
				cumulative_spectrum = old_cumulative_spectrum; 
				unsigned long long int block_gen_iters = 0;
				while (j < k && block_gen_iters < BLOCK_GEN_THRESH) {
					block_gen_iters++;
					int mark = (int)round((double)mark_lim*means[j] + (double)mark_lim*sigmas[j]*gaussrand());
					while (mark < 1 || mark > mark_lim) {
						mark = (int)round((double)mark_lim*means[j] + (double)mark_lim*sigmas[j]*gaussrand());
					}
					struct quad_ull spectrum_update_L, spectrum_update_R, spectrum_update_LR_OR, spectrum_update_LR_AND;
					struct quad_ull intersection_with_past;
					if (mark > largest_mark) {
						spectrum_update_L = shift_left(ruler[i], mark-largest_mark);
						spectrum_update_R = ZERO_QULL;
					} else {
						spectrum_update_L = shift_right(ruler[i], largest_mark-mark);
						spectrum_update_R = shift_right(ruler_nat[i], mark);
					}
					OR_qull(spectrum_update_L, spectrum_update_R, spectrum_update_LR_OR)
					AND_qull(spectrum_update_L, spectrum_update_R, spectrum_update_LR_AND)
					AND_qull(cumulative_spectrum, spectrum_update_LR_OR, intersection_with_past)
					if (iszero_qull(spectrum_update_LR_AND) && iszero_qull(intersection_with_past)) {
						toggle_qull_byref(&ruler_nat[i], mark);
						if (mark > largest_mark) {
							ruler[i] = spectrum_update_L;
							ruler[i].q0 = ruler[i].q0 | 1ULL;
							largest_mark = mark;
						} else {
							toggle_qull_byref(&ruler[i], largest_mark-mark);
						}
						OR_qull(spectrum[i], spectrum_update_LR_OR, spectrum[i]) // update spectrum
						OR_qull(cumulative_spectrum, spectrum_update_LR_OR, cumulative_spectrum) // update cumulative spectrum
						j++;
					}
				}
			} // END OF BLOCK GENERATION PROCEDURE
			if (j == k) {
				i++; 
			} else {
				cumulative_spectrum = old_cumulative_spectrum;
				struct quad_ull old_cumulative_spectrum_with_deletion;
				int offset = rand()%i;
				for (int tt = 0; tt < i; tt++) {
					int t = (tt+offset)%i;
					XOR_qull(cumulative_spectrum, spectrum[t], cumulative_spectrum)
					old_cumulative_spectrum_with_deletion = cumulative_spectrum;
					// START OF BLOCK GENERATION PROCEDURE
					int j = 0; // marks populated excluding zero mark
					unsigned long long int block_restarts = 0; 
					while (j < k && block_restarts < BLOCK_RESTART_THRESH) {
						block_restarts++;
						ruler[i] = ONE_QULL;
						ruler_nat[i] = ONE_QULL;
						spectrum[i] = ZERO_QULL;
						int largest_mark = 0;
						j = 0;
						cumulative_spectrum = old_cumulative_spectrum_with_deletion;
						unsigned long long int block_gen_iters = 0;
						while (j < k && block_gen_iters < BLOCK_GEN_THRESH) {
							block_gen_iters++;
							int mark = (int)round((double)mark_lim*means[j] + (double)mark_lim*sigmas[j]*gaussrand());
							while (mark < 1 || mark > mark_lim) {
								mark = (int)round((double)mark_lim*means[j] + (double)mark_lim*sigmas[j]*gaussrand());
							}
							struct quad_ull spectrum_update_L, spectrum_update_R, spectrum_update_LR_OR, spectrum_update_LR_AND;
							struct quad_ull intersection_with_past;
							if (mark > largest_mark) {
								spectrum_update_L = shift_left(ruler[i], mark-largest_mark);
								spectrum_update_R = ZERO_QULL;
							} else {
								spectrum_update_L = shift_right(ruler[i], largest_mark-mark);
								spectrum_update_R = shift_right(ruler_nat[i], mark);
							}
							OR_qull(spectrum_update_L, spectrum_update_R, spectrum_update_LR_OR)
							AND_qull(spectrum_update_L, spectrum_update_R, spectrum_update_LR_AND)
							AND_qull(cumulative_spectrum, spectrum_update_LR_OR, intersection_with_past)
							if (iszero_qull(spectrum_update_LR_AND) && iszero_qull(intersection_with_past)) {
								toggle_qull_byref(&ruler_nat[i], mark);
								if (mark > largest_mark) {
									ruler[i] = spectrum_update_L;
									ruler[i].q0 = ruler[i].q0 | 1ULL;
									largest_mark = mark;
								} else {
									toggle_qull_byref(&ruler[i], largest_mark-mark);
								}
								OR_qull(spectrum[i], spectrum_update_LR_OR, spectrum[i]) // update spectrum
								OR_qull(cumulative_spectrum, spectrum_update_LR_OR, cumulative_spectrum) // update cumulative spectrum
								j++;
							}
						}
					} // END OF BLOCK GENERATION PROCEDURE
					if (j == k) {
						ruler[t] = ruler[i];
						ruler_nat[t] = ruler_nat[i];
						spectrum[t] = spectrum[i];
						break;
					} else {
						cumulative_spectrum = old_cumulative_spectrum;
					}
				} //block deletion candidate loop
			}

		} // END OF DTS GENERATION PROCEDURE
		if (i == n) {
			// This is infrequent so doesn't have to be fast:
			int scope = -1;
			for (int q = 0; q < n; q++) {
				int length = get_largest_mark_byref(&ruler[q]);
				if (length > scope) {
					scope = length;
				}
			}
			if (scope < best_scope) {
				for (int q = 0; q < n; q++) {
					best_ruler[q] = ruler[q];
					best_ruler_nat[q] = ruler_nat[q];
					best_spectrum[q] = spectrum[q]; 
				}
				best_scope = scope;
				best_cumulative_spectrum = cumulative_spectrum;
				printf("successful: trial = %llu, best_scope = %d \n", trial, best_scope);
			}
		} else if (trial%STATUS_FREQ == 0) {
			printf("status: best_scope = %d, trial = %llu, i = %d\n", best_scope, trial, i);
		}
		trial++;
	} // trial loop
	
	printf("best (%d,%d)-DTS, scope = %d:\n", n, k, best_scope);
	for (int block = 0; block < n; block++) {
		printf_qull(best_ruler[block]);
	}
	struct quad_ull ref_cumulative_spectrum = ZERO_QULL;
	printf("spectrum:\n");
	for (int block = 0; block < n; block++) {
		printf_qull(best_spectrum[block]);
		OR_qull(best_spectrum[block], ref_cumulative_spectrum, ref_cumulative_spectrum);
	}
	printf("purported cumulative spectrum:\n");
	printf_qull(best_cumulative_spectrum);
	printf("actual cumulative spectrum:\n");
	printf_qull(ref_cumulative_spectrum);
	
	// decode DTS
	int decoded_ruler[n][k+1];
	for (int block = 0; block < n; block++) {
		for (int i = k; i >= 0; i--) {
			decoded_ruler[block][i] = get_largest_mark_byref(&best_ruler_nat[block]);
			struct quad_ull one_hot;
			one_hot = shift_left(ONE_QULL, get_largest_mark_byref(&best_ruler_nat[block]));
			XOR_qull(best_ruler_nat[block], one_hot, best_ruler_nat[block])
		}
	}
	
	// decode spectrum
	int decoded_spectrum[n][k*(k+1)/2];
	for (int block = 0; block < n; block++) {
		int i = k*(k+1)/2-1;
		while (i >= 0) {
			decoded_spectrum[block][i] = get_largest_mark_byref(&best_spectrum[block]);
			struct quad_ull one_hot;
			one_hot = shift_left(ONE_QULL, get_largest_mark_byref(&best_spectrum[block]));
			XOR_qull(best_spectrum[block], one_hot, best_spectrum[block])
			i--;
		}
	}
	
	printf("decoded DTS: \n");
	for (int block = 0; block < n; block++) {
		for (int i = 0; i < k+1; i++) {
			printf("%d ", decoded_ruler[block][i]);
		}
		printf("\n");
	}
	printf("decoded spectrum: \n");
	for (int block = 0; block < n; block++) {
		for (int i = 0; i < k*(k+1)/2; i++) {
			printf("%d ", decoded_spectrum[block][i]);
		}
		printf("\n");
	}
	
	now = time(NULL);
	printf("time elapsed = %ld seconds \n", now-then);
	
	printf("BLOCK_GEN_THRESH %d BLOCK_RESTART_THRESH %d DTS_GEN_THRESH %d\n", 
			BLOCK_GEN_THRESH, BLOCK_RESTART_THRESH, DTS_GEN_THRESH);
	
	return 0;
} // main
