# The Attention series: from "Attention Is All You Need" to CUDA vs. JAX

This is a living document — one episode gets fully scripted at a time, in
the order the user is building understanding, not necessarily tex order.
Full math source: `Documents/FlashAttention/FlashAttention.tex`. Finale
(already scripted): `CuLLM/Documents/AttentionBenchmarkShortForm.md`.

**Series arc, tentative:**
1. **This episode** — the attention mechanism itself: Q/K/V, softmax,
   scaling, and *why* positional encoding has to exist (permutation
   equivariance). Stops right before multi-head attention.
2. Multi-head attention — next, once episode 1 lands. The user's stated
   confusion: relating single-head attention to MHA. Don't script this yet;
   wait for the user.
3. (later, unscripted) Safe-softmax / online normalizer (Paper II) → the
   IO-awareness argument (Paper III) → tensor cores → **finale**: CUDA vs.
   JAX/XLA/cuDNN (`AttentionBenchmarkShortForm.md`, already done).

---

# Episode 1: What Attention Actually Computes

Tex source, **in tex order** (confirmed against the compiled PDF, 2026-07-03):
§4 Setup (line 549) → §5 The Softmax Map (581) → §6 Scaled Dot-Product
Attention (663) → §7 The Scaling Factor (815) → §8 Permutation Equivariance
(842) → §9 Positional Encoding (932). Stops before §10 Multi-Head Attention
(993) — **this stopping point is confirmed correct, don't change it.**

The tex itself now has one added sentence (§6, right after the score-matrix
definition) making explicit what clicked for the user: $QK^\top$ is a matrix
product, but entrywise it's a dot product — $S_{ij} = \langle q_i,
k_j\rangle/\sqrt{d_k}$, because column $j$ of $K^\top$ *is* row $j$ of $K$.
One line, no proof needed, and it's now the source of truth both documents
below quote from.

## Video visuals: screenshot *this doc's* rendered preview, not the PDF

Correction from an earlier draft of this file: the video screenshots come
from **this markdown file's rendered preview** (Cursor, VS Code, GitHub —
anything that renders KaTeX/MathJax in markdown), specifically the ON
SCREEN column of the Part B table below, or the isolated large-format
blocks in the next section. **`FlashAttention.pdf` is reading/reference
material only** — it's what backs Part A and lets you verify a beat's
equation is stated exactly right, but it was never meant to double as
video-presentable content, and it shouldn't have to: it's a 37-page paper
with full proofs and surrounding context, not a set of cropped visuals.
Screenshotting it would mean cropping dense paragraphs down to one
equation each time; screenshotting this doc's preview means the equation
is already isolated.

If you do want to cross-check a beat against the formal source (e.g. to
confirm Proposition 8.3's wording exactly), the page map is: §4 Setup +
§5 Softmax Map on p.7, §6 Scaled Dot-Product Attention (with the new
dot-product sentence) on p.8, §7 Scaling + §8 Equivariance on p.10, §9
Positional Encoding on p.11 — but that's a verification step, not a
screenshot source.

## Screenshot-ready equations (large format, one per beat)

Isolated so each renders as a clean, full-width block in preview — no
table-cell cropping needed. Same order and labels as the Part B script
below.

**Hook** — no equation, title card only.

**Beat 1 (§4):**
$$X \in \R^{n\times d_{\mathrm{model}}}, \qquad Q = XW^Q$$

**Beat 2 (§5):**
$$\softmax(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}, \qquad \softmax(x)=\softmax(y) \iff x-y\in\R\mathbf{1}$$

**Beat 3 (§6):**
$$S = \frac{QK^\top}{\sqrt{d_k}}, \qquad S_{ij} = \frac{\langle q_i,k_j\rangle}{\sqrt{d_k}}$$

**Beat 4 (§6):**
$$P_i = \softmax(S_i), \qquad O = PV$$

**Beat 5 (§7):**
$$\mathrm{Var}[\langle q,k\rangle] = d_k$$

**Beat 6 (§8) — Proposition 8.3, verbatim:**
$$\forall\,\pi\in S_n,\ \forall\,Q,K,V:\qquad \Att(\pi\cdot Q,\ \pi\cdot K,\ \pi\cdot V) \;=\; \pi\cdot\Att(Q,K,V)$$
$$\Att \text{ is } S_n\text{-equivariant.}$$

**Beat 7 (§9):**
$$\mathrm{PE}_{\mathrm{pos},2i} = \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right), \qquad \mathrm{PE}_{\mathrm{pos},2i+1} = \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right)$$

**Beat 8 (§9) — Proposition 9.2:**
$$
\begin{pmatrix}\mathrm{PE}_{\mathrm{pos}+k,2i}\\\mathrm{PE}_{\mathrm{pos}+k,2i+1}\end{pmatrix}
= R(\omega_i k)
\begin{pmatrix}\mathrm{PE}_{\mathrm{pos},2i}\\\mathrm{PE}_{\mathrm{pos},2i+1}\end{pmatrix},
\qquad R(\theta)\in SO(2)
$$

**Bridge** — no equation, title card only.

## Part A — understand it first (for you, not the camera)

Renumbered to match tex section order exactly, since that's what you asked
to preserve.

**1. (§4) A sequence is a matrix, and $Q,K,V$ are it, linearly mapped.**
$n$ tokens stacked as rows: $X \in \R^{n\times d_{\mathrm{model}}}$. The
tex's own example of "projected/learned" is $Q = XW^Q$ — a genuinely loose
use of "projection" (a real linear-algebra projection satisfies $P^2=P$;
these learned $W$'s don't have to). $K$ and $V$ follow the same pattern,
formalized properly in §6 (next).

**2. (§5) The softmax map, briefly — and its one truly algebraic fact.**
$\softmax(x)_i = e^{x_i}/\sum_j e^{x_j}$, landing in the open simplex
$\operatorname{int}(\Delta^{n-1})$. The fact worth carrying: its fibers are
exactly the cosets of the line $\R\mathbf{1}\subset\R^n$ — $\softmax(x) =
\softmax(y) \iff x - y \in \R\mathbf{1}$. Since $(\R^n,+)$ is a group and
$\R\mathbf{1}$ is a subgroup (a 1-dimensional subspace, closed under
addition), this says softmax literally factors through the quotient group
$\R^n/\R\mathbf{1}$, and descends there to a diffeomorphism onto
$\operatorname{int}(\Delta^{n-1})$. This is genuinely a quotient-by-a-subgroup
statement, not just "shift-invariance" dressed up — it's the first place the
abstract-algebra lens pays off.

**3. (§6, Def 6.1–6.2) The score matrix, entrywise.** Formally now:
$Q,K\in\R^{n\times d_k}$, $V\in\R^{n\times d_v}$. Score matrix
$S = QK^\top/\sqrt{d_k}$. **The clarification that made this click**, now
also in the tex: entrywise, $S_{ij} = \langle q_i, k_j\rangle/\sqrt{d_k}$ —
column $j$ of $K^\top$ *is* row $j$ of $K$, so the $(i,j)$ entry of the
matrix product $QK^\top$ is exactly the dot product of row $i$ of $Q$ with
row $j$ of $K$. Matrix multiplication and "a table of pairwise dot products"
are the same statement here, not two different things to reconcile.

**4. (§6, Remark 6.3) Output = weighted average.** $P_i = \softmax(S_i)$,
$O = PV$. Row $i$: $O_i = \sum_j P_{ij}v_j$, a convex combination of the
value vectors — a soft nearest-neighbor lookup.

**5. (§7) Why $\sqrt{d_k}$: a variance argument, not a magic constant.**
Independent mean-0/variance-1 coordinates give $\mathrm{Var}[\langle
q,k\rangle] = d_k$ — the raw dot product's spread *grows with dimension*.
Large scores push softmax toward a simplex vertex, where the gradient is
nearly zero. Dividing by $\sqrt{d_k}$ pins the variance back to 1 regardless
of $d_k$.

**6. (§8) The theorem, stated precisely: $\Att$ is $S_n$-equivariant.**
For every $\pi \in S_n$ and every $Q,K,V$:
$$\Att(\pi\cdot Q,\ \pi\cdot K,\ \pi\cdot V) \;=\; \pi\cdot\Att(Q,K,V).$$
Proof sketch: $S$ transforms by conjugation with the permutation matrix
($S' = P_\pi S P_\pi^\top$), row-wise softmax commutes with row permutation,
and the rest follows. **Corollary (set-valued):** attention treats its input
as an unordered multiset — permute every token identically and the output
permutes identically. There is no mechanism anywhere in $S_{ij} = \langle
q_i,k_j\rangle/\sqrt{d_k}$ that distinguishes position 3 from position 30
except through the *content* sitting there. This is *the* pivot of the
episode: it's the reason positional encoding must exist at all, not a
design choice someone made.

**7. (§9, Def 9.1) Positional encoding, defined.** Add a fixed
$\mathrm{PE}\in\R^{n\times d_{\mathrm{model}}}$ to the embeddings before any
attention layer runs: $\mathrm{PE}_{\mathrm{pos},2i} =
\sin(\mathrm{pos}/10000^{2i/d})$, $\mathrm{PE}_{\mathrm{pos},2i+1} =
\cos(\mathrm{pos}/10000^{2i/d})$ — $d/2$ sine/cosine pairs at geometrically
spaced frequencies.

**8. (§9, Prop 9.2) Why the rotation fact belongs here — this directly
answers "why did you mention rotation?".** It's not decorative; it's the
mathematical payload the whole hook is building toward. Fix a frequency
pair $(2i, 2i+1)$. Proposition 9.2 says shifting position by any offset $k$
acts on that pair as an honest linear map — and that linear map is a
rotation, an element of $SO(2)$:
$$
\begin{pmatrix}\mathrm{PE}_{\mathrm{pos}+k,2i}\\\mathrm{PE}_{\mathrm{pos}+k,2i+1}\end{pmatrix}
= R(\omega_i k)
\begin{pmatrix}\mathrm{PE}_{\mathrm{pos},2i}\\\mathrm{PE}_{\mathrm{pos},2i+1}\end{pmatrix},
\qquad R(\theta)\in SO(2),\ \ \omega_i = 10000^{-2i/d}.
$$
(It's the angle-addition formula, nothing more exotic — but that's exactly
the point: a genuine identity, not an analogy.) Stack all $d/2$ frequency
pairs and this says: **position-shift acts on positional encoding as a
representation of the additive group of shifts, landing in a product of
$SO(2)$'s.** Positional encoding isn't just "add some waves so the symmetry
breaks" — it's constructed so that *relative* position is recoverable from
*absolute* position by a fixed linear map, because shift really is being
represented as rotation. That's the single fact in this episode that most
rewards a viewer who already knows what a group representation is — which
is exactly the audience the hook promises.

**Where this leaves you:** everything above describes *one* query/key/value
triple — one attention "head." The transformer never uses just one; it
runs several of these in parallel, on different learned subspaces, then
concatenates. That's multi-head attention, and it's genuinely a different
question from anything above (not a bigger version of the same thing) —
which is exactly why it deserves its own episode instead of a tacked-on
paragraph here. **(Confirmed: stopping here is the right cut point.)**

## Part B — the short-form script

**Honest heads-up on length:** adding the softmax properties, the full
equivariance statement, and the dot-product clarification (all requested)
pushed the narration to **~248 words**. At a deliberate technical pace
(~2 words/sec — slower than casual speech, since viewers are also reading
equations) that's **~120–125 seconds**, not the 60–90s of a typical short.
Two honest options; my recommendation is B:

- **Option A — one video, ~2 min.** Everything below, straight through.
  Simpler to produce, one hook, one upload.
- **Option B (recommended) — split at the natural seam, Beat 5 | Beat 6.**
  - *Part 1, "What does attention actually compute?"* — Hook (plain
    mechanism framing) + Beats 1–5 (setup, softmax basics, score matrix,
    output, scaling). ~124 words, ~60s. This is the "standard explainer"
    content — clean, fast, sets up notation for everything downstream.
  - *Part 2, "...now read it like an abstract algebraist."* — a *second*,
    sharper hook, paid off immediately: Beats 6–8 (the $S_n$-equivariance
    theorem, positional encoding, the rotation/representation fact) +
    bridge. ~124 words, ~60s. This is where the abstract-algebra framing
    actually earns its keep — cosets, a stated theorem, a representation —
    so the hook lands with zero delay instead of 60 seconds into a longer
    video.

  Reasoning for the split: the hook you want ("an advanced mathematician
  with an abstract-algebra background reads this paper") is a promise that
  pays off hardest exactly at Beats 6–8. Beats 1–5 are excellent, correct,
  necessary setup — but they're the content *any* attention explainer
  covers. Splitting lets the sharp hook open the video where the payoff
  actually is, instead of asking viewers to sit through five beats of
  standard material first.

Table below is written for **Option A** (the full sequence); if you go with
Option B, cut it into two videos at the marked seam — no rewriting needed,
the rows don't change.

| # | Tex | ON SCREEN | NARRATION (say this, no more) |
|---|---|---|---|
| Hook | — | title card | "What does 'Attention Is All You Need' actually say, if you read it the way an abstract algebraist would? Six definitions. One hidden symmetry theorem." |
| 1 | §4 | $X\in\R^{n\times d_{\mathrm{model}}}$; $Q=XW^Q$ | "A sequence is just a matrix — one row per token. Query, key, value are that matrix, linearly mapped by learned weights. 'Projected' here just means multiplied." |
| 2 | §5 | $\softmax(x)_i=\frac{e^{x_i}}{\sum_j e^{x_j}}$; $\softmax(x)=\softmax(y)\iff x-y\in\R\mathbf 1$ | "Before attention even enters: softmax's level sets are cosets of the line R-one. It's a diffeomorphism from that quotient onto the open simplex." |
| 3 | §6 | $S=\dfrac{QK^\top}{\sqrt{d_k}}$; $S_{ij}=\dfrac{\langle q_i,k_j\rangle}{\sqrt{d_k}}$ | "The score matrix: entry i,j is exactly the dot product of query i with key j — a row of Q times a column of K-transpose, which is just row j of K." |
| 4 | §6 | $P_i=\softmax(S_i)$; $O=PV$ | "Softmax each row, multiply by V: the output is a weighted average of the values — a soft nearest-neighbor lookup." |
| 5 | §7 | $\mathrm{Var}[\langle q,k\rangle]=d_k$ | "Why divide by root d-k? Raw dot-product variance grows with dimension. Rescaling pins it to 1, so softmax doesn't collapse to a vertex and kill the gradient." |
| 6 | §8 | **Prop 8.3, verbatim:** $\forall\pi\in S_n,\,Q,K,V:\ \Att(\pi{\cdot}Q,\pi{\cdot}K,\pi{\cdot}V)=\pi{\cdot}\Att(Q,K,V)$. $\Att$ is $S_n$-equivariant. | "Here's the theorem: for every permutation π in the symmetric group S-n, Att is S_n-equivariant. Permute the input rows, the output permutes identically — attention sees a set, not a sequence." |
| 7 | §9 | $\mathrm{PE}_{\mathrm{pos},2i}=\sin(\cdot),\ \mathrm{PE}_{\mathrm{pos},2i+1}=\cos(\cdot)$ | "So order has to be injected from outside: a fixed sine-cosine encoding, added before attention runs." |
| 8 | §9 | Prop 9.2 rotation equation, $R(\omega_i k)\in SO(2)$ | "And it's not arbitrary. Shifting position by k acts as an SO(2) rotation on each frequency pair — position isn't just encoded, it's represented, as a genuine group action." |
| Bridge | — | "Next: there's never just one Q, K, V." | "I've described one attention head. There's never just one. Next time: multi-head attention — a genuinely different question." |

## Honesty guardrails

- Don't call $W^Q,W^K,W^V$ "projections" without the aside that it's the
  paper's loose usage, not the idempotent linear-algebra definition.
- Beat 2's "cosets" claim is precise, not loose: $\R\mathbf{1}$ really is a
  subgroup of $(\R^n,+)$, and softmax's fibers really are its cosets — say
  it that way, not "softmax is shift-invariant" (true but weaker, and
  misses the algebra).
- Beat 3's dot-product remark is an identity, not an approximation — "is
  exactly," not "can be thought of as."
- Beat 6 is quoted **verbatim** from Proposition 8.3 — don't paraphrase the
  on-screen text into something looser than what's proven.
- The scaling explanation (Beat 5) is a **variance** argument — say
  "variance" or "grows with dimension," not "gets too big."
- Beat 8's rotation fact holds **per frequency pair**, not globally — it's
  $d/2$ independent $SO(2)$ rotations at different angular speeds, not one
  rotation of the whole PE vector. Say "a representation," not "the
  representation," if there's any doubt about overclaiming precision.
- Don't preview multi-head attention's mechanics in the bridge line beyond
  "there's more than one" — episode 2 owns that explanation.

## Production notes

- Visuals come from this doc's own rendered preview (see "Video visuals"
  section above) — not from `FlashAttention.pdf`, which is reading
  material only and was never meant to be video-presentable.
- Real duration will likely be dominated by how long each equation is held
  on screen for a math-literate viewer to actually parse it, not by
  narration speed — budget 3–5s of hold time per dense equation
  independent of the spoken-word timing above.
- This episode deliberately excludes the Jacobian/gradient material from
  §6 (Remark 6.4, Prop 6.5, page 9 of the PDF) — that's backward-pass
  content for a later gradients episode, not the "what is attention" intro.
