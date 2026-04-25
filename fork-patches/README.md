# fork-patches

Local patch artifacts prepared against `YGO-ExodAI/ygopro-core` but not
yet pushed. Each file is a standard `git format-patch` output, applicable
to a fresh checkout of the fork via `git am <patch>`.

The patches sit here instead of in the fork because their only consumer
at this stage is the training build's in-flight encoder port (Phase B.3.a
of the edo9300 migration), and we don't want to publish patch shapes
before they've been validated end-to-end by the consumer.

## Push policy (deferred until B.8)

Even once a patch has been validated end-to-end by the consumer, **do
not push it to the fork until B.8 cutover**. The full accumulated patch
set is pushed as a batch at cutover, not incrementally during the
migration. This keeps the fork branch stable for the duration of the
migration (no risk of mid-migration rebase against an evolving fork
tip) and concentrates the "publish fork SHA + bump xmake pin + remove
io.replace" ritual into a single step at the end.

Until B.8:
- Prepare the commit on the local worktree branch.
- Export a `git format-patch -1` artifact into this directory.
- Apply the change in the training build via `io.replace` in
  `src/ygo-agent/repo/packages/e/edopro-core/xmake.lua` at `on_install`
  time, with a comment pointing at the patch file.
- Do not push.

At B.8:
- Push all accumulated patches in one batch to
  `YGO-ExodAI/ygopro-core` (separate branch per patch, or a single
  cumulative branch — whichever fits the review workflow at the time).
- Bump the xmake pin to include them.
- Delete all `io.replace` blocks and preserved-intent comments from
  the package's `on_install`.
- Re-run the full regression suite against the fork-pinned build.

**Current state** (2026-04-20): the training build applies the same
source-level changes as these patches via `io.replace`. B.3.a is
complete; the encoder port has validated
`0001-Add-QUERY_ATTACKED_COUNT-tag-to-card-get_infos.patch` end-to-end,
but push is deferred per the policy above.

## Patches

- `0001-Add-QUERY_ATTACKED_COUNT-tag-to-card-get_infos.patch`
  Adds a new scalar TLV tag to `card::get_infos` emitting
  `card->attacked_count` as a uint8. Needed by training observation col
  42. Commit base: `4d4bb4c` (exodai branch tip). Prepared on local
  branch `b3a-query-attacked-count` in `/tmp/ygopro-core-b3a` worktree.

## Migrating a patch from here to the fork

```
# from any clone of YGO-ExodAI/ygopro-core at the same base SHA:
git am path/to/0001-whatever.patch
git push origin <branch>
```

Then:
1. Update the xmake package's `add_versions(...)` to pin the new SHA.
2. Remove the corresponding `io.replace` block from the package's
   `on_install`.
3. Remove any transitional `#ifndef` guard the wrapper added for the
   define.
4. Re-run regressions against the fork-pinned build to confirm parity.
