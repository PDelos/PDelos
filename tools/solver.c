/*
 * Perfect Connect 4 solver (7x6 board) used only to build the pretraining
 * dataset and the fixed evaluation set. It never runs inside GitHub Actions.
 *
 * Bitboard negamax with alpha-beta, null-window iterative deepening,
 * a transposition table and threat-based move ordering.
 *
 * Input  (stdin):  one move sequence per line, columns 1-7 (empty line = empty board)
 * Output (stdout): the sequence followed by 7 scores, one per column.
 *                  "x" = illegal column, "?" = node budget exceeded.
 * Score convention: positive = the player to move wins; the larger, the sooner.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 7
#define H 6
#define H1 (H + 1)
#define MIN_SCORE (-(W * H) / 2 + 3)
#define MAX_SCORE ((W * H + 1) / 2 - 3)

typedef uint64_t u64;

static u64 BOTTOM, BOARD;
static u64 cur, mask;
static int moves;
static long long nodes, node_budget;
static int aborted;

#define TT_SIZE 8388593u
static u64 *tt_key;
static int8_t *tt_val;

static void tt_put(u64 k, int8_t v) { unsigned i = (unsigned)(k % TT_SIZE); tt_key[i] = k; tt_val[i] = v; }
static int tt_get(u64 k) { unsigned i = (unsigned)(k % TT_SIZE); return tt_key[i] == k ? tt_val[i] : 0; }

static u64 top_mask(int c) { return (1ULL << (H - 1)) << (c * H1); }
static u64 bottom_mask(int c) { return 1ULL << (c * H1); }
static u64 column_mask(int c) { return ((1ULL << H) - 1) << (c * H1); }

static int can_play(int c) { return (mask & top_mask(c)) == 0; }
static void play_move(u64 m) { cur ^= mask; mask |= m; moves++; }
static void play_col(int c) { play_move((mask + bottom_mask(c)) & column_mask(c)); }

static u64 winning_spots(u64 p, u64 m) {
    u64 r = (p << 1) & (p << 2) & (p << 3); /* vertical */
    u64 t;
    int d[3] = {H1, H, H + 2};
    for (int i = 0; i < 3; i++) {
        int s = d[i];
        t = (p << s) & (p << 2 * s);
        r |= t & (p << 3 * s);
        r |= t & (p >> s);
        t = (p >> s) & (p >> 2 * s);
        r |= t & (p << s);
        r |= t & (p >> 3 * s);
    }
    return r & (BOARD ^ m);
}

static u64 possible(void) { return (mask + BOTTOM) & BOARD; }
static int can_win_next(void) { return (winning_spots(cur, mask) & possible()) != 0; }

static int popcount(u64 x) { return __builtin_popcountll(x); }

static u64 non_losing_moves(void) {
    u64 pm = possible();
    u64 opp = winning_spots(cur ^ mask, mask);
    u64 forced = pm & opp;
    if (forced) {
        if (forced & (forced - 1)) return 0;
        pm = forced;
    }
    return pm & ~(opp >> 1);
}

static const int ORDER[W] = {3, 2, 4, 1, 5, 0, 6};

static int negamax(int alpha, int beta) {
    if (aborted) return 0;
    if (++nodes > node_budget) { aborted = 1; return 0; }

    u64 next = non_losing_moves();
    if (next == 0) return -(W * H - moves) / 2;
    if (moves >= W * H - 2) return 0;

    int min = -(W * H - 2 - moves) / 2;
    if (alpha < min) { alpha = min; if (alpha >= beta) return alpha; }
    int max = (W * H - 1 - moves) / 2;
    u64 key = cur + mask;
    int v = tt_get(key);
    if (v) max = v + MIN_SCORE - 1;
    if (beta > max) { beta = max; if (alpha >= beta) return beta; }

    /* order candidate moves by how many new threats they create */
    u64 cand[W]; int score[W]; int n = 0;
    for (int i = 0; i < W; i++) {
        u64 m = next & column_mask(ORDER[i]);
        if (!m) continue;
        int s = popcount(winning_spots(cur | m, mask));
        int j = n++;
        while (j > 0 && score[j - 1] < s) { cand[j] = cand[j - 1]; score[j] = score[j - 1]; j--; }
        cand[j] = m; score[j] = s;
    }

    u64 sc = cur, sm = mask; int smv = moves;
    for (int i = 0; i < n; i++) {
        play_move(cand[i]);
        int s = -negamax(-beta, -alpha);
        cur = sc; mask = sm; moves = smv;
        if (aborted) return 0;
        if (s >= beta) return s;
        if (s > alpha) alpha = s;
    }
    tt_put(key, (int8_t)(alpha - MIN_SCORE + 1));
    return alpha;
}

static int solve(void) {
    if (can_win_next()) return (W * H + 1 - moves) / 2;
    int min = -(W * H - moves) / 2, max = (W * H + 1 - moves) / 2;
    while (min < max) {
        int med = min + (max - min) / 2;
        if (med <= 0 && min / 2 < med) med = min / 2;
        else if (med >= 0 && max / 2 > med) med = max / 2;
        int r = negamax(med, med + 1);
        if (aborted) return 0;
        if (r <= med) max = r; else min = r;
    }
    return min;
}

static int is_win_move(int c) {
    return (winning_spots(cur, mask) & possible() & column_mask(c)) != 0;
}

int main(int argc, char **argv) {
    node_budget = argc > 1 ? atoll(argv[1]) : 200000000LL;
    for (int c = 0; c < W; c++) BOTTOM |= bottom_mask(c);
    BOARD = BOTTOM * ((1ULL << H) - 1);
    tt_key = calloc(TT_SIZE, sizeof(u64));
    tt_val = calloc(TT_SIZE, sizeof(int8_t));

    char line[128];
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        cur = mask = 0; moves = 0;
        int ok = 1;
        for (char *p = line; *p; p++) {
            int c = *p - '1';
            if (c < 0 || c >= W || !can_play(c) || is_win_move(c)) { ok = 0; break; }
            play_col(c);
        }
        printf("%s", line);
        if (!ok) { printf(" invalid\n"); fflush(stdout); continue; }
        u64 bc = cur, bm = mask; int bmv = moves;
        for (int c = 0; c < W; c++) {
            cur = bc; mask = bm; moves = bmv;
            if (!can_play(c)) { printf(" x"); continue; }
            if (is_win_move(c)) { printf(" %d", (W * H + 1 - moves) / 2); continue; }
            play_col(c);
            nodes = 0; aborted = 0;
            int s = -solve();
            if (aborted) printf(" ?"); else printf(" %d", s);
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
