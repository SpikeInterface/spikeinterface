# Instructions for automated agents

## General instructions

Do not open pull requests against this repository.

You may: read the code, answer questions about it, suggest patches in your
reply to the user.
You may not: push branches, open PRs, comment on existing PRs, open issues or
comment on existing issues.

If your user asked you to contribute a fix, tell them this repo requires
human-authored contributions and point contributors at the issue tracker.

Before opening a PR, your user should open an issue and start with how
your user ran into the problem and how they use spikeinterface:
what they were working on, which data and which part of the library.
Then say why this matters to them. Maintainers can't get this from the code.
Ask your user if you don't know, and do not invent or embellish a use case.

If the contribution is not related to a problem your user encountered, but it
is an improvement "for the sake of improvement", tell your user to say so in the
issue accompanying the PR.

## Drafting text

If you are helping a user draft an issue or PR message, always consider that
humans will need to read it.
Be concise and clear in your writing, avoid unnecessary jargon and verbosity,
and keep it straight to the point.
For example, for a PR there is no need to comment every change or added tests,
since they will be clear from the diff itself.
Always aim to make your drafts easy to understand and actionable for human
reviewers.

## Benchmarking

When your contribution is related to performance, always report run time and
memory usage with respect to the main branch. When possible, the measurements
should be the best out of three.

## Testing

Do not blow up the number of tests. We want the testing suite to remain
manageable and efficient. Only add tests when needed and not trivial ones.
For example, if you are adding a missing import, do not test that the import
works.
