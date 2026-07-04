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

## Recording & editing logistics (series-wide, decided 2026-07-04)

Gear: gaming tower (weak Bluetooth/mic), MacBook Pro (best mic), ASUS
Zephyrus (has OBS). **Decision: decouple audio from visuals — don't make
one device do both.**

1. **Visuals**: for equation-slide content like this, don't screen-record
   live (no typing/scrolling on camera). Take one clean static screenshot
   per beat from this doc's own rendered preview (see "Screenshot-ready
   equations" in each episode) — any machine, no mic or OBS needed for
   this step.
2. **Audio**: record narration separately on the MacBook (best mic),
   reading straight down each episode's Teleprompter section. QuickTime's
   "New Audio Recording" or Voice Memos is enough — no face-on-camera in
   this format, so no eye-line/lighting concerns, just redo a line if
   needed.
3. **Edit**: combine images + audio track in **CapCut** (free,
   vertical-native, auto-captions from the narration — solves muted-viewer
   discovery for free). Skip iMovie for this series: it's landscape-first
   and vertical editing is workable but clunky, with no auto-captions.
   DaVinci Resolve (free tier) is the "grow into it" option if this
   becomes a long-running series and more control is worth the steeper
   learning curve.

Narrate in your own voice — don't add heavy on-screen caption text
alongside the equations (redundant-channel overload; the equations *are*
the visual content) and don't use AI TTS (undercuts the "I derived this
myself" personal-brand framing this series and the CHAOS interview prep
share). Small auto-captions of the speech itself are still worth keeping.

Optional upgrade: a cheap USB mic (roughly 30–50 USD, e.g. Fifine/Samson Q2U) on the
desktop would let future episodes go back to a single live narrated
screen-record pass, removing the audio/visual device split entirely —
worth it only if this series keeps going past a couple more episodes.

Optional, your call: a 3–5s face-cam hook before cutting to screen content
is common for STEM shorts (personal connection before the math starts) —
adds camera/lighting complexity not otherwise needed for this format.

**Update (2026-07-04): the "optional upgrade" above already happened** —
OBS is now set up on the MacBook Pro itself (the best-mic device), so
Episode 1 is being recorded as a single live take: video + narration
together, no audio/visual device split. Setup: the local slide deck
(`Data/Public/Generated/AttentionSeries/Episode1/slides.html`) opened in a
normal browser window, captured via OBS's macOS Screen Capture source,
cropped to just the slide frame via a Crop/Pad filter (the deck computes
and displays the exact crop numbers live, self-correcting for whatever
window size is actually available — no manual pixel math), then scaled to
1080×1920 via Edit Transform's Bounding Box. The narration/teleprompter
text stays visible on-screen next to the frame for reading, but sits
outside the cropped region so it never appears in the recording.

---

# Episode 1: What Attention Actually Computes

Tex source, **in tex order** (line numbers current as of 2026-07-04, after
§4 gained two new remarks and §4/§6 both gained explicit Q/K/V construction
formulas — see below):
§4 Setup (line 549) → §5 The Softmax Map (652) → §6 Scaled Dot-Product
Attention (734) → §7 The Scaling Factor (895) → §8 Permutation Equivariance
(922) → §9 Positional Encoding (1012). Stops before §10 Multi-Head Attention
(1073) — **this stopping point is confirmed correct, don't change it.**

The tex itself has three additions now, all directly requested:
- §6, right after the score-matrix definition: one sentence making explicit
  what clicked for the user — $QK^\top$ is a matrix product, but entrywise
  it's a dot product — $S_{ij} = \langle q_i, k_j\rangle/\sqrt{d_k}$,
  because column $j$ of $K^\top$ *is* row $j$ of $K$.
- §4, right after the "Projected and learned" remark: two new remarks,
  **"From $X$ to the query, key, and value matrices"** (spells out
  $Q:=XW^Q$, $K:=XW^K$, $V:=XW^V$ explicitly and with equal display
  prominence — not just $Q$ with $K,V$ mentioned in passing — states the
  dimensions crisply, notes $W^Q,W^K,W^V$ are three *separate* learned
  parameters, and reminds the reader that $d_k$ is the *query/key*
  dimension — shared, because $q_i\cdot k_j$ needs both in the same
  space) and **"Typical dimensions in practice"** (a small table of real
  $(n, d_{\mathrm{model}}, h, d_k)$ values from Vaswani et al. 2017,
  BERT, GPT-2, GPT-3, and LLaMA, each cited).
- §6's Definition 6.1 ("Attention inputs") now states the construction
  formulas directly too ($Q:=XW^Q$, $K:=XW^K$, $V:=XW^V$, with a forward
  reference to the §4 remark above) instead of only giving shapes —
  so the formal definition and the intuition-building remark both show
  the same explicit construction, not just one of them.

All three are now the source of truth Beat 1 below quotes from.

## Video visuals: screenshot *this doc's* rendered preview, not the PDF

Correction from an earlier draft of this file: the video screenshots come
from **this markdown file's rendered preview** (Cursor, VS Code, GitHub —
anything that renders KaTeX/MathJax in markdown), specifically the ON
SCREEN column of the Part B table below, or the isolated large-format
blocks in the next section. **`FlashAttention.pdf` is reading/reference
material only** — it's what backs Part A and lets you verify a beat's
equation is stated exactly right, but it was never meant to double as
video-presentable content, and it shouldn't have to: it's a 38-page paper
with full proofs and surrounding context, not a set of cropped visuals.
Screenshotting it would mean cropping dense paragraphs down to one
equation each time; screenshotting this doc's preview means the equation
is already isolated.

If you do want to cross-check a beat against the formal source (e.g. to
confirm Proposition 8.3's wording exactly), the page map (recompiled
2026-07-04, after Definition 6.1 grew to state the Q/K/V construction
formulas directly — §7 shifted down one more page as a result) is: §4
Setup (incl. "From $X$ to the query, key, and value matrices" and
"Typical dimensions" remarks) on p.7, §5 Softmax Map on p.8, §6 Scaled
Dot-Product Attention (Definition 6.1 now explicit about $Q,K,V$'s
construction, plus the dot-product sentence) on p.9, §7 Scaling on p.11,
§8 Equivariance also p.11, §9 Positional Encoding on p.12 — but that's a
verification step, not a screenshot source.

## Screenshot-ready equations (large format, one per beat)

Isolated so each renders as a clean, full-width block in preview — no
table-cell cropping needed. Same order and labels as the Part B script
below.

**Hook** — no equation, title card only.

**Beat 1 (§4):**
$$X \in \mathbb{R}^{n\times d}, \quad \text{row } i = \text{element } i; \qquad d = d_{\mathrm{model}}; \qquad Q = XW^Q \in \mathbb{R}^{n\times d_k}$$

**Beat 2 (§5):**
$$x=(x_1,\ldots,x_n)\in\mathbb{R}^n; \qquad Z := \sum_j e^{x_j} \ \text{(partition function, à la Gibbs)}; \qquad \operatorname{softmax}(x)_i := \frac{e^{x_i}}{Z}$$

**Beat 3 (§6):**
$$S = \frac{QK^\top}{\sqrt{d_k}}, \qquad S_{ij} = \frac{\langle q_i,k_j\rangle}{\sqrt{d_k}}$$

**Beat 4 (§6):**
$$P_i = \operatorname{softmax}(S_i), \qquad O = PV$$

**Beat 5 (§7):**
$$\mathrm{Var}[\langle q,k\rangle] = d_k$$

**Beat 6 (§8) — Proposition 8.3, verbatim:**
$$\forall\,\pi\in S_n,\ \forall\,Q,K,V:\qquad \operatorname{Att}(\pi\cdot Q,\ \pi\cdot K,\ \pi\cdot V) \;=\; \pi\cdot\operatorname{Att}(Q,K,V)$$
$$\operatorname{Att} \text{ is } S_n\text{-equivariant.}$$

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

**1. (§4) A sequence is a matrix — tex Definition (Sequence), verbatim in
spirit.** "A sequence of length $n$ in $\mathbb{R}^d$ is an element of
$\mathbb{R}^{n\times d}$" — a matrix whose $i$-th row is the $i$-th element
of the sequence. **What is $d$ here?** In this general definition $d$ is
just a placeholder; the moment we apply it to attention's input, $d =
d_{\mathrm{model}}$, the *embedding dimension* — how many numbers represent
one token before anything else happens to it. So: $X \in
\mathbb{R}^{n\times d_{\mathrm{model}}}$.

**Where does $Q = XW^Q$ come from, and is it really in Paper I?** Yes —
same section (§4), immediately after the sequence definition, in the
Remark "Projected and learned" (`rem:projection`). It's not our tex's
invention: that remark explicitly cites **§3.2 of the original "Attention
Is All You Need"** as the source — the paper's own words are "projected by
learned weight matrices," and it calls $W^Q,W^K,W^V$ "projection
matrices." Our tex uses $Q=XW^Q$ as the illustrative example of what that
means concretely: $X\in\mathbb{R}^{n\times d_{\mathrm{model}}}$,
$W^Q\in\mathbb{R}^{d_{\mathrm{model}}\times d_k}$, so $Q\in\mathbb{R}^{n\times
d_k}$ — right-multiplication, i.e. the linear map $X\mapsto XW^Q$. Same
remark flags that "projection" here is the paper's loose usage: a real
(linear-algebra) projection satisfies $P^2=P$, and these learned $W$'s
don't have to (squaring isn't even defined once $d_k\neq
d_{\mathrm{model}}$).

**And $K,V$?** As of 2026-07-04, no longer "follow the same pattern, left
implicit" — the very next remark (`rem:query-matrix`, retitled "From $X$
to the query, key, and value matrices") spells out $K:=XW^K$ and
$V:=XW^V$ with the *same* display prominence as $Q$, not mentioned only
in passing, and notes explicitly that $W^Q,W^K,W^V$ are three *separate*
learned parameters — nothing ties them together, which is exactly what
lets $Q$ and $K$ end up in geometrically different "views" of the same
token. §6's Definition 6.1 ("Attention inputs") — that's next — now
*also* states $Q:=XW^Q,K:=XW^K,V:=XW^V$ directly instead of only giving
their shapes, cross-referencing back to this remark for the fuller
derivation.

**2. (§5) The softmax map, briefly — what the $x_i$'s are, and the reading
straight from the tex's own remark.** $\operatorname{softmax}(x)_i =
e^{x_i}/\sum_j e^{x_j}$ for $i=1,\ldots,n$. At this point in the tex, $x$
is fully generic — just $n$ real numbers, nothing to do with attention
yet (that binding happens in §6: $x$ becomes one row of the score matrix,
$x_i$ = how well one query matches key $i$). Write $Z := \sum_j e^{x_j}$.
The tex's own remark says exactly this: $Z$ (its log is called the
log-sum-exp, or log-partition function) is the partition function of the
**Gibbs measure** on outcomes $\{1,\ldots,n\}$ with energy $-x_i$, and
$\operatorname{softmax}(x)_i = e^{x_i}/Z$ is precisely the Gibbs (Boltzmann)
probability of outcome $i$ under that measure. That's the whole beat — no
diffeomorphism, no fibers, no cosets needed to use softmax; those are real
facts in the tex (Prop. 5.5) but not ones this episode needs.

**3. (§6, Def 6.1–6.2) The score matrix, entrywise.** Formally now:
$Q,K\in\mathbb{R}^{n\times d_k}$, $V\in\mathbb{R}^{n\times d_v}$. Score matrix
$S = QK^\top/\sqrt{d_k}$. **The clarification that made this click**, now
also in the tex: entrywise, $S_{ij} = \langle q_i, k_j\rangle/\sqrt{d_k}$ —
column $j$ of $K^\top$ *is* row $j$ of $K$, so the $(i,j)$ entry of the
matrix product $QK^\top$ is exactly the dot product of row $i$ of $Q$ with
row $j$ of $K$. Matrix multiplication and "a table of pairwise dot products"
are the same statement here, not two different things to reconcile.

**4. (§6, Remark 6.3) Output = weighted average.** $P_i = \operatorname{softmax}(S_i)$,
$O = PV$. Row $i$: $O_i = \sum_j P_{ij}v_j$, a convex combination of the
value vectors — a soft nearest-neighbor lookup.

**5. (§7) Why $\sqrt{d_k}$: a variance argument, not a magic constant.**
Independent mean-0/variance-1 coordinates give $\mathrm{Var}[\langle
q,k\rangle] = d_k$ — the raw dot product's spread *grows with dimension*.
Large scores push softmax toward a simplex vertex, where the gradient is
nearly zero. Dividing by $\sqrt{d_k}$ pins the variance back to 1 regardless
of $d_k$.

**6. (§8) The theorem, stated precisely: $\operatorname{Att}$ is $S_n$-equivariant.**
For every $\pi \in S_n$ and every $Q,K,V$:
$$\operatorname{Att}(\pi\cdot Q,\ \pi\cdot K,\ \pi\cdot V) \;=\; \pi\cdot\operatorname{Att}(Q,K,V).$$
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
$\mathrm{PE}\in\mathbb{R}^{n\times d_{\mathrm{model}}}$ to the embeddings before any
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

**Decision (2026-07-03): Option A, one video.** Length (recomputed from the
actual final teleprompter text, **including Beat 4** — see the note on
Beat 4 below): **331 words, ~165s** at a deliberate technical pace. That's
grown from an earlier ~140s estimate as Beats 1, 2, 6, and 7 picked up
more explicit framing (the query-matrix construction, the Gibbs-measure
setup, the theorem statement, the causal link to positional encoding).
No format constraint is pushing back on this — TikTok/Shorts/Reels all
support much longer "short-form" uploads now — so length isn't a reason to
trim; trim only if something reads as genuinely unnecessary on a dry read.

**Revision (2026-07-04), verbatim pass for live recording:** the user
rewrote the script beat-by-beat for a live single-take OBS recording
(video + mic together, MacBook Pro's mic — no more decoupled audio/visual
split for this episode). Two structural changes: (1) a new **Cold Open**
beat now precedes the Hook — a screenshot of the actual paper's title page
with a two-line intro (paper name, 2017, "defines the Transformer
architecture underlying generative AI... ChatGPT, and beyond") before the
"read like an abstract algebraist" framing; (2) the Bridge now names the
actual head count. New length: **418 words, ~208s (~3:28)** at the same
pace — still no format constraint pushing back on this. The live
production deck (screenshot-ready frames + this teleprompter, kept in
sync) lives outside this repo at
`Data/Public/Generated/AttentionSeries/Episode1/slides.html` — this repo
stays code/tex/docs only; produced video assets don't belong here.

The Beat 5|6 seam is still marked in the teleprompter below, in case a
*content* reason to split ever comes up later (e.g. the video runs long
in a dry read for pacing reasons, not format reasons) — but it's no longer
the plan, just a reference point.

**Open decision: Beat 4 ("the output is a weighted average").** Question
raised 2026-07-04: is this actually shown in the tex, or just asserted?
Checked — it's real, cited content: **Remark 6.3 ("Interpretation")**,
right after the score/output definitions, states verbatim "a convex
combination of the value vectors." The one-line "why," if it's worth
saying out loud: softmax's outputs are non-negative and sum to 1 (that's
its codomain, the simplex), so $O_i=\sum_j P_{ij}v_j$ is a convex
combination *by definition*, not by a separate argument. It currently
stays in the script as-is (unchanged below) — cut it if it still feels
like dead weight on a read-through, but it isn't an invented claim.

### Teleprompter — read this straight down, nothing else

Narration only, no table, no equations, no tex references — just the words,
in order, read straight through as one video. The `⸻ split here ⸻` marker
is left in as a reference point only (see the note above) — ignore it
unless a real content reason to split comes up later.

> "'Attention Is All You Need' came out in 2017. It's the landmark paper
> that defines the Transformer architecture underlying generative AI
> today — ChatGPT, and beyond."
>
> "But as I read it, I tried to understand it from the perspective of a
> pure mathematician — in particular, someone with a background in
> abstract algebra. We'll walk through the paper's definitions, and show
> one hidden symmetry theorem underlying the Transformer architecture."
>
> "Define a sequence of length n in R-d as a matrix X in R to the n-by-d,
> where the ith row is the ith element in that sequence of length n. Here,
> d is d-model: the embedding dimension. Operate on X by right-multiplying
> by learned weights W-Q to get the query matrix, Q in R to the n-by-d-k."
>
> "Consider a general array of numbers, x-1 through x-n. Define a
> partition function Z, just like a Gibbs distribution — the sum of e to
> the x-j. Then softmax of x, at i, is simply e to the x-i, over Z."
>
> "Define the score matrix S to be Q times K-transpose, over the square
> root of d-k — the key dimension. So entry S-i-j is exactly the dot
> product of row i of Q with column j of K-transpose, which is just row j
> of K."
>
> "Softmax each row to get P sub i, and right-multiply by V — the output
> is then a weighted average of the values."
>
> "Why did we divide by the square root of d-k in the score matrix S? Raw
> dot-product variance grows with dimension. Rescaling pins it to 1, so
> softmax doesn't collapse to a vertex and kill the gradient."
>
> **⸻ (reference marker only — read straight through, don't actually pause here) ⸻**
>
> "It can be shown that, using the symmetric group S-n: for every
> permutation π in S-n, and every Q, K, V — where Att of Q, K, V is just
> the whole computation we've built, from score matrix through softmax to
> the output — Att is S_n-equivariant. Permute the input rows, and the
> output permutes identically."
>
> "So, because of that S_n-equivariance, order has to be injected from
> outside — a fixed sine-cosine encoding, added before attention ever
> runs."
>
> "And it's not arbitrary. Shifting position by k acts as an SO(2)
> rotation on each frequency pair — position isn't just encoded, it's
> represented, as a genuine group action."
>
> "I've described only one attention head. In the paper, the authors used
> h equals 8, and called this multi-head attention, which I'll describe
> in the next video. Check the comments or my bio for the paper and code."

(Kept for reference only: if a split ever becomes useful later, Part 2
would need its *own* hook line in place of a cold open — *"...now read the
same paper like an advanced mathematician with a background in abstract
algebra would."* — a short callback, not a full re-hook, since Part 1
already did the setup work.)

### Full table (context: on-screen cues + tex refs, for reference while producing — not for reading aloud)

| # | Tex | ON SCREEN | NARRATION (say this, no more) |
|---|---|---|---|
| Cold Open | — | screenshot of the paper's actual title page | "'Attention Is All You Need' came out in 2017. It's the landmark paper that defines the Transformer architecture underlying generative AI today — ChatGPT, and beyond." |
| Hook | — | title card | "But as I read it, I tried to understand it from the perspective of a pure mathematician — in particular, someone with a background in abstract algebra. We'll walk through the paper's definitions, and show one hidden symmetry theorem underlying the Transformer architecture." |
| 1 | §4 | $X\in\mathbb{R}^{n\times d}$, row $i$ = element $i$; $d=d_{\mathrm{model}}$; $Q=XW^Q\in\mathbb{R}^{n\times d_k}$ *(§3.2, original paper)* | "Define a sequence of length n in R-d as a matrix X in R to the n-by-d, where the ith row is the ith element in that sequence of length n. Here, d is d-model: the embedding dimension. Operate on X by right-multiplying by learned weights W-Q to get the query matrix, Q in R to the n-by-d-k." |
| 2 | §5 | $x=(x_1,\ldots,x_n)\in\mathbb{R}^n$; $Z:=\sum_j e^{x_j}$ (partition function, à la Gibbs); $\operatorname{softmax}(x)_i:=e^{x_i}/Z$ | "Consider a general array of numbers, x-1 through x-n. Define a partition function Z, just like a Gibbs distribution — the sum of e to the x-j. Then softmax of x, at i, is simply e to the x-i, over Z." |
| 3 | §6 | $S=\dfrac{QK^\top}{\sqrt{d_k}}$; $S_{ij}=\dfrac{\langle q_i,k_j\rangle}{\sqrt{d_k}}$ | "Define the score matrix S to be Q times K-transpose, over the square root of d-k — the key dimension. So entry S-i-j is exactly the dot product of row i of Q with column j of K-transpose, which is just row j of K." |
| 4 | §6 | $P_i=\operatorname{softmax}(S_i)$; $O=PV$ | "Softmax each row to get P sub i, and right-multiply by V — the output is then a weighted average of the values." |
| 5 | §7 | $\mathrm{Var}[\langle q,k\rangle]=d_k$ | "Why did we divide by the square root of d-k in the score matrix S? Raw dot-product variance grows with dimension. Rescaling pins it to 1, so softmax doesn't collapse to a vertex and kill the gradient." |
| 6 | §8 | **Prop 8.3, verbatim:** $\forall\pi\in S_n,\,Q,K,V:\ \operatorname{Att}(\pi{\cdot}Q,\pi{\cdot}K,\pi{\cdot}V)=\pi{\cdot}\operatorname{Att}(Q,K,V)$. $\operatorname{Att}$ is $S_n$-equivariant. | "It can be shown that, using the symmetric group S-n: for every permutation π in S-n, and every Q, K, V — where Att of Q, K, V is just the whole computation we've built, from score matrix through softmax to the output — Att is S_n-equivariant. Permute the input rows, and the output permutes identically." |
| 7 | §9 | $\mathrm{PE}_{\mathrm{pos},2i}=\sin(\cdot),\ \mathrm{PE}_{\mathrm{pos},2i+1}=\cos(\cdot)$ | "So, because of that S_n-equivariance, order has to be injected from outside — a fixed sine-cosine encoding, added before attention ever runs." |
| 8 | §9 | Prop 9.2 rotation equation, $R(\omega_i k)\in SO(2)$ | "And it's not arbitrary. Shifting position by k acts as an SO(2) rotation on each frequency pair — position isn't just encoded, it's represented, as a genuine group action." |
| Bridge | — | "Next: there's never just one Q, K, V." | "I've described only one attention head. In the paper, the authors used h equals 8, and called this multi-head attention, which I'll describe in the next video. Check the comments or my bio for the paper and code." |

## Honesty guardrails

- Don't call $W^Q,W^K,W^V$ "projections" without the aside that it's the
  paper's loose usage, not the idempotent linear-algebra definition.
- Beat 2's Gibbs-measure claim is precise, not decorative: the tex's own
  remark states the energy as $-x_i$ (so $e^{x_i} = e^{-(-x_i)}$), matching
  the usual Boltzmann-factor sign convention. The spoken line skips the
  sign for brevity ("softmax is the Gibbs distribution built from Z") —
  that's fine for the video, but if asked to elaborate, say "energy minus
  x-i," not "energy x-i."
- Beat 1's $Q=XW^Q$ is real, cited content (§4, Remark "Projected and
  learned," citing §3.2 of the original paper) — not an invented bridge
  from setup to attention. As of 2026-07-04, $K:=XW^K$ and $V:=XW^V$ are
  *also* explicit in §4 (the very next remark, with equal display
  prominence to $Q$) — §6's Definition 6.1 is no longer the first place
  their construction appears, only the first place they're given their
  formal name as "the query, key, and value matrices" as a triple. Beat 1
  deliberately still doesn't say K, V out loud in the narration — that's
  a script-economy choice (one beat, one construction, shown once,
  generalizes visibly on screen without needing three sentences of
  narration to say so), not a claim that §4 leaves them undefined.
- Beat 2's array $x=(x_1,\ldots,x_n)$ is still fully generic at this point
  in the tex — don't let "soon it'll be attention scores" slip into the
  spoken line itself; that binding happens in §6, one beat later.
- Beat 6's opening ("you can prove a genuine theorem") is doing real
  work, not just rhetorical flourish — say it before the theorem
  statement, not after, so the viewer knows a proof-grade claim is coming.
- Beat 7's "because of that S_n-equivariance" is a genuine callback, not
  a rhetorical connective — the tex's own text says almost exactly this
  ("This is the mathematical reason positional encodings must be added,"
  directly after the equivariance corollary in §8). Keep the causal
  wording; don't soften it to "so" or "now."
- Beat 3's dot-product remark is an identity, not an approximation — "is
  exactly," not "can be thought of as."
- Beat 6 is quoted **verbatim** from Proposition 8.3 — don't paraphrase the
  on-screen text into something looser than what's proven.
- Beat 6's narration (2026-07-04 revision) now defines Att inline ("the
  whole computation we've built, from score matrix through softmax to the
  output") instead of leaving it unstated, and drops the closing "Attention
  sees a set, not a sequence" line — Beat 7's own "because of that
  S_n-equivariance" callback still carries the causal link without it.
- Bridge's head count is **h=8**, not "N=6" — in the original paper, N=6
  is the number of stacked encoder/decoder *layers*, a different parameter
  from the number of attention heads. Don't conflate the two if either
  comes up again.
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

---

# Episode 2: Why One Attention Head Isn't Enough

Started 2026-07-04, directly requested (Episode 1's stated confusion: how
does single-head attention relate to MHA). Tex source: §10 Multi-Head
Attention only (line 1073, pages 12–13) — Definition 10.1, Remark 10.2
(column-wise concatenation, with a fully worked $n{=}2,h{=}2,d_v{=}2$
numerical example), Remark 10.3 (dimension bookkeeping), Remark 10.4
(subspace decomposition — **this is the remark that answers the stated
confusion directly**). Stops before §11 (Feed-Forward/LayerNorm) — same
"stop at a clean section boundary" pattern as Episode 1.

## Part A — understand it first

**1. The setup: $h$ heads, each with its *own* three weight matrices.**
This is the one fact that resolves the confusion before anything else
does: multi-head attention is **not** "run the same $\operatorname{Att}$ twice" and
it's **not** "make $d_k$ bigger." For each head $\ell = 1,\ldots,h$, fix
*separate* learned matrices $W^Q_\ell, W^K_\ell \in
\mathbb{R}^{d_{\mathrm{model}}\times d_k}$, $W^V_\ell \in
\mathbb{R}^{d_{\mathrm{model}}\times d_v}$ — $3h$ independent parameters
total, none of them shared across heads. So head $\ell$ doesn't just
re-run attention on the same $Q,K,V$ from Episode 1 — it builds its
*own* $Q_\ell := X W^Q_\ell$, $K_\ell := XW^K_\ell$, $V_\ell := XW^V_\ell$
first (the exact §4 construction from Episode 1, now run $h$ times with
$h$ different weight matrices).

**2. Each head is just Episode 1's $\operatorname{Att}$, unchanged.** $\mathrm{head}_\ell
:= \operatorname{Att}(XW^Q_\ell, XW^K_\ell, XW^V_\ell) \in \mathbb{R}^{n\times d_v}$ —
literally the same map from §6, called $h$ times with $h$ different
inputs. Nothing about $\operatorname{Att}$ itself changes; MHA is a composition
*around* it, not a modification *of* it.

**3. Concatenate the heads — side by side, not stacked.** $[\mathrm{head}_1
\| \cdots \| \mathrm{head}_h] \in \mathbb{R}^{n \times hd_v}$: **column-wise**
means wider, not taller — $n$ rows stay $n$ rows, the column count grows
from $d_v$ to $hd_v$. The tex's own worked example makes this concrete
rather than notation-only: with $n=2, h=2, d_v=2$,
$$\mathrm{head}_1 = \begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix},\ \
\mathrm{head}_2 = \begin{pmatrix}b_{11}&b_{12}\\b_{21}&b_{22}\end{pmatrix}
\ \Rightarrow\ 
[\mathrm{head}_1\|\mathrm{head}_2] = \left(\begin{array}{cc|cc}a_{11}&a_{12}&b_{11}&b_{12}\\a_{21}&a_{22}&b_{21}&b_{22}\end{array}\right).$$
Row $i$ of the concatenation is the *horizontal join* of row $i$ from
every head, in order — entrywise, $C_{i,(\ell-1)d_v+j} :=
(\mathrm{head}_\ell)_{ij}$.

**4. One more linear map merges the heads back down.**
$\operatorname{MHA}(Q,K,V) := [\mathrm{head}_1\|\cdots\|\mathrm{head}_h]\,W^O \in
\mathbb{R}^{n\times d_{\mathrm{model}}}$, with $W^O \in
\mathbb{R}^{hd_v\times d_{\mathrm{model}}}$ — the *only* place information
from different heads gets combined into one representation again.
Everything before this point kept the $h$ heads' computations completely
independent.

**5. Real numbers, and a FLOP-neutrality fact worth having ready.** The
original paper: $d_{\mathrm{model}}=512$, $h=8$, $d_k=d_v=64$ (matches
Episode 1's "Typical dimensions" table exactly — this *is* the
Transformer-base row). Note $h\cdot d_v = 8\cdot 64 = 512 = d_{\mathrm{model}}$
— the concatenated output lands back at the *same* width the input
started at, so $W^O\in\mathbb{R}^{512\times 512}$ is square. A fact worth
having ready for a technical follow-up question: the tex's own Remark 10.3
notes the total FLOPs of 8-head attention at $d_k=64$ equal a *single*
head running at $d_k=d_{\mathrm{model}}=512$ — multi-head attention isn't
"8× the compute of one head," it's the *same* compute, split $h$ ways.

**6. The payoff — why one (bigger) head can't do this.** Remark 10.4,
almost verbatim: a single head, whatever its width, produces **one**
convex combination per row — one weighted average, one notion of
"what matched what." Multiple independent relationships between the
*same* pair of positions (e.g. "these two tokens are the same part of
speech" *and*, separately, "these two tokens are 3 positions apart") would
have to be blended into that single average, and averaging destroys the
distinction between them. $h$ heads means $h$ *independent* attention
computations running in parallel, each in its own learned subspace (the
image of $W^Q_\ell$), each free to discover a *different* relationship —
then $W^O$ combines the results, rather than having to average them
together from the start. That's the entire argument, and it's the one to
have ready if this comes up: **multi-head isn't about attending "more,"
it's about attending to several *different things* at once, which a
single softmax-weighted average structurally cannot represent.**

**Bridge to what's next:** every head above still computes the *full*
$n\times n$ score matrix $S_\ell = Q_\ell K_\ell^\top/\sqrt{d_k}$ and
writes it to memory before softmax ever runs — for long sequences, that
matrix gets enormous, and there are now $h$ of them per layer. That's
where the series goes next: Paper II/III's actual subject, computing
$\operatorname{Att}$ without ever materializing that matrix — which is the direct road
to the CUDA-vs-JAX finale.

## Part B — the short-form script

**Not yet written.** Given the interview timeline, Part A above is
complete and ready whenever there's time to distill it into a teleprompter
script (following the exact template used for Episode 1: hook → numbered
beats → bridge, each beat an ON SCREEN equation + one narration line, a
teleprompter blockquote, a full table, honesty guardrails, production
notes). Do this the same way Episode 1 was built — iteratively, beat by
beat, with the user reviewing each — rather than generating a finished
script in one pass; that's what made Episode 1's script actually correct.
