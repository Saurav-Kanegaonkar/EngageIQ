# How People Contribute to Open Source: Evidence Review (for EngageIQ ranking)

> Literature + web research synthesis (2026-06-02) to inform how EngageIQ should rank
> contribution opportunities. Most evidence is descriptive/correlational from GitHub mining +
> developer surveys; two causal-ish exceptions (Lakhani-Wolf motivation survey; ifo 2024
> signaling DiD). Skeptical flags kept inline.

## Q1. What repos are worth contributing to?
No single validated rubric, but consensus "project health" signals converge:
- **Responsiveness** (time-to-first-response on issues/PRs; PR age) = the single strongest driver of *repeat* contribution (Mozilla/CHAOSS).
- **Contributor health** (>=2 active maintainers, newcomers who return, bus-factor).
- **Release cadence**, **governance/roadmap**, contribution guide + code of conduct.
- (opensource.guide/metrics, CHAOSS, Red Hat 12-factors, Nadia Eghbal project-health.)

Popularity/stars != good target:
- Stars proxy *visibility*, not contributor-friendliness.
- Big/mature projects are demonstrably HARDER to enter (code complexity strongly reduces newcomer contributions; custom infra/CLAs raise entry cost).
- On famous repos, newcomer issues get **preempted by experienced devs**: GFIs solved by newcomers only **21.2%** of the time; **40.9%** of GFIs not solved by newcomers at all (RecGFI / Tan et al.).

"Good first issue" label is a WEAK signal:
- Scarce: only **~1.93%** of issues labeled (median ~4%/project).
- Mis-targeted: **31.2%** of newcomers fail a labeled GFI.
- The strong signal is **issue write-up quality** (clear repro, expected behavior, code snippets, links). Many genuinely-suitable issues are **unlabeled**. RecGFI text features -> **0.85 AUC** for true newcomer-suitability.

## Q2. Why do people contribute? (Lakhani-Wolf 2005, n=684)
Top motivations: user-need (>58%), intellectually stimulating / fun (44.9%, top single), **improves skills (41.8%)**, ideology (~33%), give-back (28.6%), belonging (~20%). Reputation self-reported LOW, but revealed-preference (career signaling) says otherwise. ~40% paid; 73% "lose track of time" (flow). Three buckets: enjoyment-intrinsic (strongest), community/obligation, extrinsic (own-use, skill-building, paid, career signaling).

## Q3. Learning trajectory + barriers
Onion model: user -> bug-reporter -> occasional -> regular -> committer -> leader. Inward movement needs technical skill AND social reputation. **~70% of occasional contributors make exactly one contribution and never return.** Learn: real code review, conventions/tooling, async collaboration with strangers (social challenges > code challenges, Gousios).
Barriers (Steinmacher SLR, 21 studies): social interaction + timely response (Jensen: ~80% of newbie posts got replies; **<48h responses correlate with future participation**); previous knowledge; finding a way to start (only 16.7% given a task); code complexity/setup; docs. Toxicity: 2017 GitHub survey, **21% stopped contributing** after a negative interaction. BUT response *sentiment* doesn't predict retention (blunt critique is fine); *responsiveness + non-toxicity* do.

## Q4. Leverage for learning AND recognition
Signaling is causally real: ifo 2024 DiD (~22,900 devs) -> OSS activity **+16% during job search**; ~6.8% of all OSS activity is signaling-driven, concentrated in high-visibility + top-paying-language repos (aimed at signal > community need). 50% of contributors said OSS was important to getting their current role. First PRs judged mostly on **code quality + project-fit** (follow CONTRIBUTING.md, run tests, minimize deviation); a track record then buys faster reviews. Target repos with public credit (CONTRIBUTORS, release notes, all-contributors). Popular-recognition vs small-responsive-learning PULL APART.

## Q5. Recommender signals (prioritized)
- **Tier 1 (responsiveness/liveness):** median time-to-first-response (48h), PR merge rate + median PR age, recent commits/releases, >=2 maintainers. Down-rank abandoned + single-owner pet repos.
- **Tier 2 (issue quality + fit):** don't trust GFI label alone; score issue write-up quality (repro/expected/code/links/readability); surface UNLABELED suitable issues; calibrate difficulty to user skill (flow / zone of proximal development); penalize likely-preempted issues on hyper-popular repos.
- **Tier 3 (learnability vs recognition = user-tunable axis):** complexity/onboarding-cost penalty for huge/old/custom-infra repos; separate recognition score = visibility (stars/forks/dependents) + in-demand language + public credit mechanisms.
- **Tier 4 (guardrails):** toxicity/civility screen (not sentiment); mentorship proxies (help-wanted triage, onboarding docs, maintainer comments on newcomer PRs).

Default score sketch:
`score = w1*responsiveness + w2*issue_quality + w3*skill_fit - w4*complexity_penalty - w5*toxicity_penalty + (user_recognition_weight)*visibility`

## Load-bearing takeaways
1. **Responsiveness > popularity** for landing + returning.
2. **GFI label weak; issue write-up quality strong; most suitable issues unlabeled.**
3. **Recognition and learnability pull apart** -> make it an explicit, user-tunable axis, not a hidden popularity sort.

Treat as opinion/unverified: "70% employers / 38% more interviews" hiring stats (untraceable); Raymond's "many eyeballs" law.

## Key sources
- Steinmacher et al., Barriers Faced by Newcomers (SLR 2014): https://www.ime.usp.br/~gerosa/papers/Steinmacher2014_Chapter_BarriersFacedByNewcomersToOpen.pdf
- Xiao/He et al., Recommending Good First Issues (ICSE 2022): https://hehao98.github.io/files/2022-recgfi.pdf
- Tan et al., A First Look at Good First Issues (FSE 2020): https://dl.acm.org/doi/abs/10.1145/3368089.3409746
- Lakhani & Wolf, Why Hackers Do What They Do (2005): https://ocw.mit.edu/courses/15-352-managing-innovation-emerging-trends-spring-2005/8733c45a525ebcede867a9fb282398ca_lakhaniwolf.pdf
- Abou El-Komboz & Goldbeck, Career Concerns / Signaling (ifo WP 405, 2024): https://www.ifo.de/DocDL/wp-2024-405_goldbeck_carreer%20concerns.pdf
- Gousios et al., Pull-based Development (ICSE 2014): https://www.gousios.org/bibliography/GPD14.html
- Assavakamhaenghan et al., Does the First Response Matter? (2021): https://arxiv.org/html/2104.02933
- 2017 GitHub Open Source Survey: https://opensourcesurvey.org/2017/
- GitHub Open Source Guides, Metrics: https://opensource.guide/metrics/
- CHAOSS project health: https://opensource.net/measure-open-source-project-health/
- Onion model revisited (2021): https://onlinelibrary.wiley.com/doi/full/10.1111/radm.12428
