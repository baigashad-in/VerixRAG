<!-- Source: https://raw.githubusercontent.com/rust-lang/rfcs/master/text/1068-rust-governance.md -->
<!-- Domain: raw.githubusercontent.com -->

- Feature Name: not applicable
- Start Date: 2015-02-27
- RFC PR: [rust-lang/rfcs#1068](https://github.com/rust-lang/rfcs/pull/1068)
- Rust Issue: N/A

## Summary

This RFC proposes to expand, and make more explicit, Rust's governance
structure. It seeks to supplement today's core team with several
*subteams* that are more narrowly focused on specific areas of
interest.

*Thanks to Nick Cameron, Manish Goregaokar, Yehuda Katz, Niko Matsakis and Dave
 Herman for many suggestions and discussions along the way.*

## Motivation

Rust's governance has evolved over time, perhaps most dramatically
with the introduction of the RFC system -- which has itself been
tweaked many times. RFCs have been a major boon for improving design
quality and fostering deep, productive discussion. It's something we
all take pride in.

That said, as Rust has matured, a few growing pains have emerged.

We'll start with a brief review of today's governance and process,
then discuss what needs to be improved.

### Background: today's governance structure

Rust is governed by a
[core team](https://github.com/rust-lang/rust-wiki-backup/blob/master/Note-core-team.md),
which is ultimately responsible for all decision-making in the
project. Specifically, the core team:

* Sets the overall direction and vision for the project;
* Sets the priorities and release schedule;
* Makes final decisions on RFCs.

The core team currently has 8 members, including some people working
full-time on Rust, some volunteers, and some production users.

Most technical decisions are decided through the
[RFC process](https://github.com/rust-lang/rfcs#what-the-process-is).
RFCs are submitted for essentially all changes to the language,
most changes to the standard library, and
[a few other topics](https://github.com/rust-lang/rfcs#when-you-need-to-follow-this-process).
RFCs are either closed immediately (if they are clearly not viable),
or else assigned a *shepherd* who is responsible for keeping the
discussion moving and ensuring all concerns are responded to.

The final decision to accept or reject an RFC is made by the core
team. In many cases this decision follows after many rounds of
consensus-building among all stakeholders for the RFC. In the end,
though, most decisions are about weighting various tradeoffs, and the
job of the core team is to make the final decision about such
weightings in light of the overall direction of the language.

### What needs improvement

At a high level, we need to improve:

* Process scalability.
* Stakeholder involvement.
* Clarity/transparency.
* Moderation processes.

Below, each of these bullets is expanded into a more detailed analysis
of the problems. These are the problems this RFC is trying to
solve. The "Detailed Design" section then gives the actual proposal.

#### Scalability: RFC process

In some ways, the RFC process is a victim of its own success: as the
volume and depth of RFCs has increased, it's harder for the entire
core team to stay educated and involved in every RFC. The
[shepherding process](https://github.com/rust-lang/rfcs#the-role-of-the-shepherd)
has helped make sure that RFCs don't fall through the cracks, but even
there it's been hard for the relatively small number of shepherds to
keep up (on top of the other work that they do).

Part of the problem, of course, is due to the current push toward 1.0,
which has both increased RFC volume and takes up a great deal of
attention from the core team. But after 1.0 is released, the community
is likely to grow significantly, and feature requests will only
increase.

Growing the core team over time has helped, but there's a practical
limit to the number of people who are jointly making decisions and
setting direction.

A distinct problem in the other direction has also emerged recently: we've
slowly been requiring RFCs for increasingly minor changes. While it's important
that user-facing changes and commitments be vetted, the process has started to
feel heavyweight (especially for newcomers), so a recalibration may be in order.

We need a way to scale up the RFC process that:

* Ensures each RFC is thoroughly reviewed by several people with
  interest and expertise in the area, but with different perspectives
  and concerns.

* Ensures each RFC continues moving through the pipeline at a
  reasonable pace.

* Ensures that accepted RFCs are well-aligned with the values, goals,
  and direction of the project, and with other RFCs (past, present,
  and future).

* Ensures that simple, uncontentious changes can be made quickly, without undue
  process burden.

#### Scalability: areas of focus

In addition, there are increasingly areas of important work that are
only loosely connected with decisions in the core language or APIs:
tooling, documentation, infrastructure, for example. These areas all
need leadership, but it's not clear that they require the same degree
of global coordination that more "core" areas do.

These areas are only going to increase in number and importance, so we
should remove obstacles holding them back.

#### Stakeholder involvement

RFC shepherds are intended to reach out to "stakeholders" in an RFC,
to solicit their feedback. But that is different from the stakeholders
having a direct role in decision making.

To the extent practical, we should include a diverse range of
perspectives in both design and decision-making, and especially
include people who are most directly affected by decisions: users.

We have taken some steps in this direction by diversifying the core
team itself, but (1) members of the core team by definition need to
take a balanced, global view of things and (2) the core team should
not grow too large. So some other way of including more stakeholders
in decisions would be preferable.

#### Clarity and transparency

Despite many steps toward increasing the clarity and openness of
Rust's processes, there is still room for improvement:

* The priorities and values set by the core team are not always
  clearly communicated today. This in turn can make the RFC process
  seem opaque, since RFCs move along at different speeds (or are even
  closed as postponed) according to these priorities.

  At a large scale, there should be more systematic communication
  about high-level priorities. It should be clear whether a given RFC
  topic would be considered in the near term, long term, or
  never. Recent blog posts about the 1.0 release and stabilization
  have made a big step in this direction. After 1.0, as part of the
  regular release process, we'll want to find some regular cadence for
  setting and communicating priorities.

  At a smaller scale, it is still the case that RFCs fall through the
  cracks or have unclear statuses (see Scalability problems
  above). Clearer, public tracking of the RFC pipeline would be a
  significant improvement.

* The decision-making process can still be opaque: it's not always
  clear to an RFC author exactly when and how a decision on the RFC
  will be made, and how best to work with the team for a favorable
  decision. We strive to make core team meetings as *uninteresting* as
  possible (that is, all interesting debate should happen in public
  online communication), but there is still room for being more
  explicit and public.

#### Community norms and the Code of Conduct

Rust's design process and community norms are closely intertwined. The
RFC process is a joint exploration of design space and tradeoffs, and
requires consensus-building. The process -- and the Rust community --
is at its best when all participants recognize that

> ... people have differences of opinion and that every design or
> implementation choice carries a trade-off and numerous costs. There
> is seldom a right answer.

This and other important values and norms are recorded in the
[project code of conduct (CoC)](http://www.rust-lang.org/conduct.html),
which also includes language about harassment and marginalized groups.

Rust's community has long upheld a high standard of conduct, and has
earned a reputation for doing so.

However, as the community grows, as people come and go, we must
continually work to maintain this standard. Usually, it suffices to
lead by example, or to gently explain the kind of mutual respect that
Rust's community practices. Sometimes, though, that's not enough, and
explicit moderation is needed.

One problem that has emerged with the CoC is the lack of clarity about
the mechanics of moderation:

* Who is responsible for moderation?
* What about conflicts of interest? Are decision-makers also moderators?
* How are moderation decisions reached? When are they unilateral?
* When does moderation begin, and how quickly should it occur?
* Does moderation take into account past history?
* What venues does moderation apply to?

Answering these questions, and generally clarifying how the CoC is viewed and
enforced, is an important step toward scaling up the Rust community.

## Detailed design

The basic idea is to supplement the core team with several "subteams". Each
subteam is focused on a specific area, e.g., language design or libraries. Most
of the RFC review process will take place within the relevant subteam, scaling
up our ability to make decisions while involving a larger group of people in
that process.

To ensure global coordination and a strong, coherent vision for the project as a
whole, **each subteam is led by a member of the core team**.

### Subteams

**The primary roles of each subteam are**:

* Shepherding RFCs for the subteam area. As always, that means (1) ensuring that
  stakeholders are aware of the RFC, (2) working to tease out various design
  tradeoffs and alternatives, and (3) helping build consensus.

* Accepting or rejecting RFCs in the subteam area.

* Setting policy on what changes in the subteam area require RFCs, and reviewing
  direct PRs for changes that do not require an RFC.

* Delegating *reviewer rights* for the subteam area. The ability to `r+` is not
  limited to team members, and in fact earning `r+` rights is a good stepping
  stone toward team membership. Each team should set reviewing policy, manage
  reviewing rights, and ensure that reviews take place in a timely manner.
  (Thanks to Nick Cameron for this suggestion.)

Subteams make it possible to involve a larger, more diverse group in the
decision-making process. In particular, **they should involve a mix of**:

* Rust project leadership, in the form of at least one core team member (the
  leader of the subteam).

* Area experts: people who have a lot of interest and expertise in the subteam
  area, but who may be far less engaged with other areas of the project.

* Stakeholders: people who are strongly affected by decisions in the
  subteam area, but who may not be experts in the design or
  implementation of that area. *It is crucial that some people heavily
  using Rust for applications/libraries have a seat at the table, to
  make sure we are actually addressing real-world needs.*

Members should have demonstrated a good sense for design and dealing with
tradeoffs, an ability to work within a framework of consensus, and of course
sufficient knowledge about or experience with the subteam area. Leaders should
in addition have demonstrated exceptional communication, design, and people
skills. They must be able to work with a diverse group of people and help lead
it toward consensus and execution.

Each subteam is led by a member of the core team. **The leader is responsible for**:

* Setting up the subteam:

    * Deciding on the initial membership of the subteam (in consultation with
      the core team). Once the subteam is up and running.

    * Working with subteam members to determine and publish subteam policies and
      mechanics, including the way that subteam members join or leave the team
      (which should be based on subteam consensus).

* Communicating core team vision downward to the subteam.

* Alerting the core team to subteam RFCs that need global, cross-cutting
  attention, and to RFCs that have entered the "final comment period" (see below).

* Ensuring that RFCs and PRs are progressing at a reasonable rate, re-assigning
  shepherds/reviewers as needed.

* Making final decisions in cases of contentious RFCs that are unable to reach
  consensus otherwise (should be rare).

The way that subteams communicate internally and externally is left to each
subteam to decide, but:

* Technical discussion should take place as much as possible on public forums,
  ideally on RFC/PR threads and tagged discuss posts.

* Each subteam will have a dedicated
  [discuss forum](http://internals.rust-lang.org/) tag.

* Subteams should actively seek out discussion and input from stakeholders who
  are not members of the team.

* Subteams should have some kind of regular meeting or other way of making
  decisions. The content of this meeting should be summarized with the rationale
  for each decision -- and, as explained below, decisions should generally be
  about weighting a set of already-known tradeoffs, not discussing or
  discovering new rationale.

* Subteams should regularly publish the status of RFCs, PRs, and other news
  related to their area. Ideally, this would be done in part via a dashboard
  like [the Homu queue](http://buildbot.rust-lang.org/homu/queue/rust)


[Document truncated for evaluation purposes]